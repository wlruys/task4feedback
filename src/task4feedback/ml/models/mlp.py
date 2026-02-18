from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor

from task4feedback.ml.models.common import build_aux_features, get_aux_feature_dim, masked_mean_pool
from task4feedback.ml.models.nn_utils import FeatureDimConfig, _expand_to_batch, _flatten_last_dim, _tiny_last_linear, kaiming_init


def _as_list(channels: int | Sequence[int]) -> list[int]:
    if isinstance(channels, int):
        return [channels]
    return list(channels)


def _make_act(activation: DictConfig | None) -> nn.Module:
    return instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01, inplace=False)


def _get_layer_init(initialization: DictConfig | None):
    return call(initialization) if initialization else kaiming_init


def _build_mlp(
    input_dim: int,
    hidden_channels: Sequence[int],
    *,
    activation: DictConfig | None,
    initialization: DictConfig | None,
    layer_norm: bool,
) -> tuple[nn.Module, int]:
    hidden_list = list(hidden_channels)
    if not hidden_list:
        return nn.Identity(), input_dim

    layer_init = _get_layer_init(initialization)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for out_dim in hidden_list:
        layers.append(layer_init(nn.Linear(in_dim, out_dim)))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(_make_act(activation))
        in_dim = out_dim
    return nn.Sequential(*layers), in_dim


def _flatten_candidates(x: Tensor) -> tuple[Tensor, tuple[int, ...], int, int, int]:
    if x.dim() < 2:
        raise ValueError(f"Expected (..., C, D) tensor, got {tuple(x.shape)}")
    *batch_shape, C, D = x.shape
    x_flat = x.reshape(-1, C, D)
    B = x_flat.size(0)
    return x_flat, tuple(batch_shape), B, C, D


def _prepare_candidate_inputs(
    tensordict: TensorDict,
    *,
    add_device_load: bool,
    add_progress: bool,
) -> tuple[Tensor, tuple[int, ...], int, int, Tensor | None]:
    """Flatten candidate features and build broadcast aux conditioning."""
    task_features = tensordict["nodes", "tasks", "attr"]
    x_flat, batch_shape, B, C, _ = _flatten_candidates(task_features)

    aux_flat, _, _ = build_aux_features(
        tensordict,
        add_device_load=add_device_load,
        add_progress=add_progress,
    )
    if aux_flat is None:
        return x_flat, batch_shape, B, C, None

    aux_flat = _expand_to_batch(aux_flat, B)
    aux_cand = aux_flat.unsqueeze(1).expand(-1, C, -1)
    return x_flat, batch_shape, B, C, aux_cand


class _FiLMGenerator(nn.Module):

    def __init__(self, g_dim: int, out_dim: int, layer_init):
        super().__init__()
        self.out_dim = int(out_dim)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(g_dim, out_dim)),
            nn.SiLU(inplace=False),
            layer_init(nn.Linear(out_dim, 2 * out_dim)),
        )
        _tiny_last_linear(self.mlp, std=1e-4)
        self.rezero = nn.Parameter(torch.tensor(0.0))
        self.gamma_range = nn.Parameter(torch.full((out_dim,), 0.05))

    def forward(self, x: Tensor, x_norm: Tensor, g: Tensor | None, B: int, C: int) -> Tensor:
        if g is None:
            return x
        H = x.size(-1)
        g_dim = g.size(-1)

        # Fast path: broadcast conditioning (common case, produced by `.expand`)
        # Avoids materializing a (B*C, g_dim) tensor and re-running the MLP C times.
        if g.dim() == 3 and (g.size(1) == 1 or g.stride(1) == 0):
            gb = self.mlp(g[:, 0, :])  # (B, 2H)
            gamma_raw, beta = gb.chunk(2, dim=-1)
            gamma = self.gamma_range * (2 * torch.sigmoid(gamma_raw) - 1)  # (B, H)
            u = (1.0 + gamma).unsqueeze(1) * x_norm + beta.unsqueeze(1)
            return x + self.rezero * u

        if g.dim() == 2:
            gb = self.mlp(g)  # (B, 2H)
            gamma_raw, beta = gb.chunk(2, dim=-1)
            gamma = self.gamma_range * (2 * torch.sigmoid(gamma_raw) - 1)  # (B, H)
            u = (1.0 + gamma).unsqueeze(1) * x_norm + beta.unsqueeze(1)
            return x + self.rezero * u

        gb = self.mlp(g.reshape(-1, g_dim))  # (B*C, 2H) (may materialize if `g` is non-contiguous)
        gamma_raw, beta = gb.chunk(2, dim=-1)
        gamma = self.gamma_range * (2 * torch.sigmoid(gamma_raw) - 1)
        gamma = gamma.view(B, C, H)
        beta = beta.view(B, C, H)
        u = (1.0 + gamma) * x_norm + beta
        return x + self.rezero * u


class _FiLMPoolGate(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        H = int(hidden_dim)
        self.norm = nn.LayerNorm(H, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(inplace=False),
            nn.Linear(H, 2 * H),
        )
        _tiny_last_linear(self.mlp, std=1e-4)
        self.rezero = nn.Parameter(torch.tensor(0.0))
        self.gamma_range = nn.Parameter(torch.full((H,), 0.05))

    def forward(self, x: Tensor, mask_f: Tensor) -> Tensor:
        # Expect a per-candidate mask; coerce to (B, C, 1) for stable broadcasting.
        mask_f = mask_f.reshape(x.size(0), x.size(1), 1).to(dtype=x.dtype)  # (B, C, 1)
        counts = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = (x * mask_f).sum(dim=1) / counts  # (B, H)
        xn = self.norm(x)
        gb = self.mlp(pooled)
        gamma_raw, beta = gb.chunk(2, dim=-1)
        gamma = self.gamma_range * (2 * torch.sigmoid(gamma_raw) - 1)
        u = (1.0 + gamma).unsqueeze(1) * xn + beta.unsqueeze(1)
        return x + self.rezero * u


class _FiLMBlock(nn.Module):
    """Linear -> (LN) -> FiLM -> mask -> act."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        g_dim: int,
        layer_init,
        activation: nn.Module,
        layer_norm: bool,
    ):
        super().__init__()
        self.linear = layer_init(nn.Linear(in_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.film = _FiLMGenerator(g_dim, out_dim, layer_init) if g_dim > 0 else None
        self.act = activation

    def forward(self, x: Tensor, g: Tensor | None, mask_f: Tensor) -> Tensor:
        x = self.linear(x)
        x_norm = self.norm(x)
        if self.film is not None:
            B, C = x.size(0), x.size(1)
            x = self.film(x, x_norm, g, B, C)
        x = x * mask_f
        return self.act(x)


class OutputHead(nn.Module):
    def __init__(
        self,
        input_dim: int | None,
        hidden_channels: int,
        output_dim: int,
        activation: DictConfig | None = None,
        initialization: DictConfig | None = None,
        layer_norm: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.debug = bool(debug)

        layer1_init = kaiming_init
        layer2_init = kaiming_init

        if initialization is not None:
            if "layer1" in initialization:
                layer1_init = call(initialization["layer1"])
                layer2_init = call(initialization["layer2"])
            else:
                common_init = call(initialization)
                layer1_init = common_init
                layer2_init = common_init

        self._layer1_init = layer1_init
        self._lazy_layer1_initialized = False
        self._lazy_layer = None

        act = _make_act(activation)

        if input_dim is None:
            layer1: nn.Module = nn.LazyLinear(hidden_channels)
            self._lazy_layer = layer1
        else:
            layer1 = layer1_init(nn.Linear(input_dim, hidden_channels))
            self._lazy_layer1_initialized = True

        layers: list[nn.Module] = [layer1]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_channels))
        layers.append(act)
        layers.append(layer2_init(nn.Linear(hidden_channels, output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.network(x)

        if self._lazy_layer is not None and not self._lazy_layer1_initialized:
            if not hasattr(self._lazy_layer, "has_uninitialized_params") or not self._lazy_layer.has_uninitialized_params():
                self._layer1_init(self._lazy_layer)
                self._lazy_layer1_initialized = True

        return out


class MLPCriticHead(nn.Module):
    in_keys = [("observation",), ("embed",)]
    out_keys = [("state_value",)]

    def __init__(
        self,
        input_dim: int | None = None,
        input_proj: int = 32,
        hidden_channels: int = 64,
        factored: bool = False,
        output_dim: int = 1,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
        debug: bool = False,
        add_progress: bool = True,
        add_device_load: bool = True,
        n_devices: int = 5,
        **_ignored,
    ):

        super().__init__()

        self.add_progress = add_progress
        self.add_device_load = add_device_load
        self.factored = factored

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = input_proj
        self.hidden_channels = hidden_channels
        self.side_info_dim = get_aux_feature_dim(add_device_load=add_device_load, add_progress=add_progress, n_devices=n_devices)

        self.stem = nn.Sequential(
            nn.LazyLinear(input_proj),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.side_info_mlp = nn.Sequential(
            nn.LazyLinear(self.input_proj),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        self.value_mlp = OutputHead(
            input_dim=3*self.input_proj,
            hidden_channels=hidden_channels,
            output_dim=output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
            debug=debug,
        )

    def forward(self, obs: TensorDict, emb: Tensor) -> Tensor:
        emb_flat, batch_shape, B, C, _ = _flatten_candidates(emb)
        mask = obs["aux", "candidate_mask"].reshape(-1, C)

        aux, _, _ = build_aux_features(
            obs,
            add_device_load=self.add_device_load,
            add_progress=self.add_progress,
        ) # (..., D)
        if aux is not None:
            aux = aux.reshape(-1, aux.shape[-1])
            aux = self.side_info_mlp(aux) # (B*N, H)
        
        emb_flat = self.stem(emb_flat) # (B*N, C, H)
        pooled = masked_mean_pool(emb_flat, mask)
        pooled = pooled.view(-1, self.input_proj) # (B*N, H)

        if aux is not None:
            pooled = torch.cat([pooled, aux], dim=-1) # (B*N, 2H)
        
        # Expand to (B*N, C, H), concat with candidates, and pass through MLP.
        pooled = pooled.reshape(-1, 1, pooled.shape[-1]).expand(-1, C, -1) # (B*N, C, H)
        pooled = torch.cat([pooled, emb_flat], dim=-1) # (B*N, C, 2H)
        # value_mlp maps 2H -> A
        v = self.value_mlp(pooled) # (B*N, C, A)

        if not self.factored:
            v = v.mean(dim=1).unsqueeze(-1) # (B*N, A)
            v =  v.view(*batch_shape, self.output_dim)
        else:
            v = v.view(*batch_shape, C, self.output_dim)
            if self.output_dim == 1:
                v = v.squeeze(-1)

        return v


class MLPQValueHead(MLPCriticHead):
    in_keys = [("observation",), ("embed",)]
    out_keys = [("action_value",)]
    
    def __init__(self, *args, **kwargs):
        kwargs["factored"] = True
        super().__init__(*args, **kwargs)

    def forward(self, obs: TensorDict, emb: Tensor) -> Tensor:
        q = super().forward(obs, emb) # (B, C, A)
        return q


class MLPActorHead(OutputHead):
    in_keys = [("observation",), ("embed",)]
    out_keys = [("logits",)]

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_channels: int = 128,
        output_dim: int = 1,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            output_dim=output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
            debug=debug,
        )

    def forward(self, obs: TensorDict, emb: Tensor) -> Tensor:
        emb_flat, batch_shape, B, C, _ = _flatten_candidates(emb)
        _ = obs["aux", "candidate_mask"].reshape(B, C)
        out_flat = super().forward(emb_flat)
        out = out_flat.view(*batch_shape, C, -1)
        return out
    
class MLPEncoder(nn.Module):
    in_keys = [("observation",)]
    out_keys = [("embed",)]

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: list[int] | int,
        add_progress: bool = False,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
        add_device_load: bool = True,
        n_devices: int = 5,
        **_ignored,
    ):
        super().__init__()
        self.feature_config = feature_config
        self.hidden_channels = _as_list(hidden_channels)
        self.k = len(self.hidden_channels)
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)

        input_dim = feature_config.task_feature_dim
        if add_device_load:
            input_dim += 3 * n_devices
        if add_progress:
            input_dim += 2

        self.layers, self.output_dim = _build_mlp(
            input_dim,
            self.hidden_channels,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
        )

    def forward(self, tensordict: TensorDict) -> Tensor:
        x_flat, batch_shape, _, C, aux_cand = _prepare_candidate_inputs(
            tensordict,
            add_device_load=self.add_device_load,
            add_progress=self.add_progress,
        )
        if aux_cand is not None:
            x_flat = torch.cat([x_flat, aux_cand], dim=-1)

        return self.layers(x_flat).view(*batch_shape, C, self.output_dim)


class MLPFiLMEncoder(nn.Module):

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: list[int] | int,
        add_progress: bool = False,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
        add_device_load: bool = True,
        n_devices: int = 5,
        pool_film: bool = True,
        **_ignored,
    ):
        super().__init__()

        self.feature_config = feature_config
        self.hidden_channels = _as_list(hidden_channels)
        self.k = len(self.hidden_channels)
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.pool_film = bool(pool_film)

        layer_init = _get_layer_init(initialization)
        input_dim = feature_config.task_feature_dim

        g_dim = 0

        if add_progress:
            g_dim += 2

        if add_device_load:
            g_dim += 3 * n_devices

        self.g_dim = g_dim
        self.act = instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.blocks = nn.ModuleList()

        if self.k == 0:
            self.output_dim = input_dim
            self.pool_gate = None
        else:
            in_ch = input_dim
            for out_ch in self.hidden_channels:
                self.blocks.append(
                    _FiLMBlock(
                        in_ch,
                        out_ch,
                        g_dim=g_dim,
                        layer_init=layer_init,
                        activation=self.act,
                        layer_norm=layer_norm,
                    )
                )
                in_ch = out_ch
            self.output_dim = in_ch
            self.output_keys = ["embed"]

            self.pool_gate = _FiLMPoolGate(in_ch) if self.pool_film else None
        self.in_keys = [("observation",)]
        self.out_keys = [("embed",)]

    def forward(self, tensordict: TensorDict) -> Tensor:
        x_flat, batch_shape, B, C, g = _prepare_candidate_inputs(
            tensordict,
            add_device_load=self.add_device_load,
            add_progress=self.add_progress,
        )

        cand_mask_f = tensordict["aux", "candidate_mask"].reshape(-1, C).unsqueeze(-1)

        if self.k == 0:
            return x_flat.view(*batch_shape, C, self.output_dim)

        for blk in self.blocks:
            x_flat = blk(x_flat, g, cand_mask_f)

        if self.pool_gate is not None:
            x_flat = self.pool_gate(x_flat, cand_mask_f)
            x_flat = x_flat * cand_mask_f

        return x_flat.view(*batch_shape, C, self.output_dim)
