from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from task4feedback.ml.models.nn_utils import FeatureDimConfig, _choose_gn_groups, _expand_to_batch
from task4feedback.ml.models.common import build_aux_features, flatten_task_grid


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=pad)
        self.act1 = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=pad)
        self.act2 = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return out + residual


class CNNEncoder(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: int,
        add_progress: bool = False,
        activation=None,
        initialization=None,
        width: int = 4,
        length: int = 4,
    ):
        super().__init__()
        self.in_channels = feature_config.task_feature_dim
        self.add_progress = add_progress
        kernel_size = 3
        hidden_ch = hidden_channels
        n_layers = width - 1
        self.width = width
        self.length = length

        blocks = []
        ch = self.in_channels

        pad = kernel_size // 2
        blocks += [
            nn.Conv2d(ch, hidden_ch, kernel_size, padding=pad),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
        ]
        ch = hidden_ch

        for _ in range((n_layers - 2) // 2):
            blocks.append(ResidualBlock(ch, hidden_ch, kernel_size))
            ch = hidden_ch

        if n_layers % 2 == 1:
            blocks += [
                nn.Conv2d(ch, hidden_ch, kernel_size, padding=pad),
                nn.LeakyReLU(inplace=True, negative_slope=0.01),
            ]
            ch = hidden_ch
        blocks.append(nn.Conv2d(ch, 1, kernel_size, padding=pad))
        blocks.append(nn.LeakyReLU(inplace=True, negative_slope=0.01))
        ch = 1

        self.net = nn.Sequential(*blocks)
        self.output_dim = ((self.width * self.length) * ch + 1) if self.add_progress else ((self.width * self.length) * ch)
        self.output_keys = ["embed"]
        self.in_keys = [("observation",)]
        self.out_keys = [("embed",)]
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        width = self.width
        length = self.length

        x_tasks = x["nodes", "tasks", "attr"]
        h, batch_shape, single = flatten_task_grid(x_tasks, length=length, width=width, in_channels=self.in_channels)
        B = h.size(0)

        h = self.net(h).flatten(1)

        if self.add_progress:
            progress_feature = x["aux", "progress"]
            if progress_feature.dim() == 0:
                progress_feature = progress_feature.view(1, 1)
            elif progress_feature.dim() == 1:
                progress_feature = progress_feature.unsqueeze(-1)
            else:
                progress_feature = progress_feature.reshape(-1, progress_feature.shape[-1])
            progress_feature = _expand_to_batch(progress_feature, B)
            h = torch.cat([h, progress_feature], dim=-1)

        return h.squeeze(0) if single else h.view(*batch_shape, -1)


class ConvNormAct(nn.Module):
    def __init__(self, C_in, C_out, k=3, dilation=1, groups=1, act="silu"):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=k, padding=pad, dilation=dilation, bias=False, groups=groups)
        self.norm = nn.GroupNorm(_choose_gn_groups(C_out), C_out)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DilatedResBlock(nn.Module):
    def __init__(self, C: int, dilation: int = 1, act="silu"):
        super().__init__()
        self.conv1 = ConvNormAct(C, C, k=3, dilation=dilation, act=act)
        self.conv2 = ConvNormAct(C, C, k=3, dilation=1, act=act)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ECA(nn.Module):
    def __init__(self, C: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y


class TinyASPP(nn.Module):
    def __init__(self, C: int, rates=(1, 2, 3), act="silu"):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(C, C, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.GroupNorm(_choose_gn_groups(C), C),
                    nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True),
                )
                for r in rates
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(len(rates) * C, C, kernel_size=1, bias=False),
            nn.GroupNorm(_choose_gn_groups(C), C),
            nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class SpatialModulator(nn.Module):
    def __init__(self, C: int, H: int, W: int, z_spa_dim: int, ch_hidden: int = 128, seed_hw: Optional[tuple[int, int]] = None, init_scale_gamma_xy: float = 0.5, init_scale_beta_xy: float = 0.5):
        super().__init__()
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.h0 = max(4, H // 4) if not seed_hw else seed_hw[0]
        self.w0 = max(4, W // 4) if not seed_hw else seed_hw[1]

        self.to_seed = nn.Sequential(nn.Linear(z_spa_dim, ch_hidden), nn.SiLU(inplace=True), nn.Linear(ch_hidden, 2 * C * self.h0 * self.w0))
        nn.init.normal_(self.to_seed[-1].weight, std=1e-4)
        nn.init.zeros_(self.to_seed[-1].bias)

        self.scale_gamma_xy = nn.Parameter(torch.tensor(float(init_scale_gamma_xy)), requires_grad=False)
        self.scale_beta_xy = nn.Parameter(torch.tensor(float(init_scale_beta_xy)), requires_grad=False)

    @torch.no_grad()
    def set_strength(self, gamma_xy: Optional[float] = None, beta_xy: Optional[float] = None):
        if gamma_xy is not None:
            self.scale_gamma_xy.fill_(float(gamma_xy))
        if beta_xy is not None:
            self.scale_beta_xy.fill_(float(beta_xy))

    def forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        *lead, z_dim = z.shape
        B = math.prod(lead) if lead else z.shape[0]
        zf = z.reshape(-1, z_dim)
        seed = self.to_seed(zf).view(B, 2 * self.C, self.h0, self.w0)
        maps = F.interpolate(seed, size=(self.H, self.W), mode="bilinear", align_corners=False)
        g_raw, b_raw = maps.chunk(2, dim=1)
        g_xy = self.scale_gamma_xy * torch.tanh(g_raw)
        b_xy = self.scale_beta_xy * b_raw
        return g_xy, b_xy


class AdaSPADE_GN(nn.Module):
    def __init__(
        self,
        C: int,
        groups: int,
        spatial: SpatialModulator,
        z_ch_dim: int,
        ch_hidden: int = 128,
        init_scale_gamma_c: float = 0.5,
        init_scale_beta_c: float = 0.5,
        enable_spatial: bool = True,
        enable_channel: bool = True,
    ):
        super().__init__()
        self.gn = nn.GroupNorm(groups, C, affine=False)
        self.to_gb_c = nn.Sequential(nn.Linear(z_ch_dim, ch_hidden), nn.SiLU(inplace=True), nn.Linear(ch_hidden, 2 * C))
        nn.init.normal_(self.to_gb_c[-1].weight, std=1e-4)
        nn.init.zeros_(self.to_gb_c[-1].bias)

        self.scale_gamma_c = nn.Parameter(torch.tensor(init_scale_gamma_c), requires_grad=False)
        self.scale_beta_c = nn.Parameter(torch.tensor(init_scale_beta_c), requires_grad=False)

        self.spatial = spatial
        self.enable_spatial = bool(enable_spatial)
        self.enable_channel = bool(enable_channel)

    @torch.no_grad()
    def set_strength(self, gamma_c: Optional[float] = None, beta_c: Optional[float] = None):
        if gamma_c is not None:
            self.scale_gamma_c.fill_(float(gamma_c))
        if beta_c is not None:
            self.scale_beta_c.fill_(float(beta_c))

    def forward(self, x: torch.Tensor, z_ch: torch.Tensor, z_spa: torch.Tensor) -> torch.Tensor:
        B, Cx, H, W = x.shape
        z_ch = z_ch.reshape(B, -1)
        z_spa = z_spa.reshape(B, -1)

        x = self.gn(x)

        if not self.enable_channel:
            return x

        g_c_raw, b_c_raw = self.to_gb_c(z_ch).chunk(2, dim=-1)
        g_c = 1.0 + self.scale_gamma_c * torch.tanh(g_c_raw)
        b_c = self.scale_beta_c * b_c_raw

        if self.enable_spatial:
            g_xy, b_xy = self.spatial(z_spa)
            gamma = g_c.unsqueeze(-1).unsqueeze(-1) * (1.0 + g_xy)
            beta = b_c.unsqueeze(-1).unsqueeze(-1) + b_xy
        else:
            gamma = g_c.unsqueeze(-1).unsqueeze(-1)
            beta = b_c.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta


class DilatedResBlock_SPADE(nn.Module):
    def __init__(self, C: int, dilation: int, norm1: AdaSPADE_GN, norm2: AdaSPADE_GN):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = norm1
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.norm2 = norm2
        self.act2 = nn.SiLU(inplace=True)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, z_ch, z_spa):
        h = self.act1(self.norm1(self.conv1(x), z_ch, z_spa))
        h = self.act2(self.norm2(self.conv2(h), z_ch, z_spa))
        return x + h


class DilationEncoder(nn.Module):
    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        width: int,
        length: int,
        z_ch_dim: int = 8,
        z_spa_dim: int = 8,
        num_blocks: int = 3,
        dilation_schedule: Optional[list[int]] = None,
        use_eca: bool = True,
        add_z: bool = False,
        add_device_load: bool = False,
        n_devices: int = 5,
        spatial_in_all_blocks: bool = False,
        film_in_all_blocks: bool = False,
        spatial_last_k: int = 0,
        film_last_k: int = 2,
        init_gamma_c: float = 0.05,
        init_beta_c: float = 0.05,
        init_gamma_xy: float = 0.05,
        init_beta_xy: float = 0.05,
        debug: bool = False,
        add_progress: bool = False,
        **_ignored,
    ):
        super().__init__()
        if not hasattr(feature_config, "task_feature_dim"):
            raise AttributeError("feature_config must have attribute 'task_feature_dim'")

        self.width = int(width)
        self.length = int(length)
        self.in_channels = int(feature_config.task_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.debug = bool(debug)
        self.output_dim = self.hidden_channels
        self.output_keys = ["embed"]
        self.in_keys = [("observation",)]
        self.out_keys = [("embed",)]
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.add_z = bool(add_z)
        self.register_buffer("_z0", torch.zeros(1, 1), persistent=False)

        C_in = self.in_channels
        C = self.hidden_channels

        self.stem = ConvNormAct(C_in, C, k=3, dilation=1, act="silu")
        if not dilation_schedule:
            dilation_schedule = [1, 2, 3, 1]

        if self.add_z:
            zc_eff = int(z_ch_dim)
            zs_eff = int(z_spa_dim)
        else:
            zc_eff = 1
            zs_eff = 1

        if self.add_progress:
            zc_eff = zc_eff + 2
            zs_eff = zs_eff + 2

        if self.add_device_load:
            zc_eff = zc_eff + 3 * n_devices
            zs_eff = zs_eff + 3 * n_devices

        self.spatial = SpatialModulator(
            C=C,
            H=self.length,
            W=self.width,
            z_spa_dim=zs_eff,
            ch_hidden=16,
            seed_hw=(max(4, self.length // 4), max(4, self.width // 4)),
            init_scale_gamma_xy=init_gamma_xy,
            init_scale_beta_xy=init_beta_xy,
        )

        groups = _choose_gn_groups(C)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_spa = spatial_in_all_blocks or (i >= num_blocks - spatial_last_k)
            use_film = film_in_all_blocks or (i >= num_blocks - film_last_k)

            norm1 = AdaSPADE_GN(
                C=C, groups=groups, spatial=self.spatial, z_ch_dim=zc_eff, ch_hidden=16, init_scale_gamma_c=init_gamma_c, init_scale_beta_c=init_beta_c, enable_spatial=use_spa, enable_channel=use_film
            )
            norm2 = AdaSPADE_GN(
                C=C, groups=groups, spatial=self.spatial, z_ch_dim=zc_eff, ch_hidden=16, init_scale_gamma_c=init_gamma_c, init_scale_beta_c=init_beta_c, enable_spatial=use_spa, enable_channel=use_film
            )

            self.blocks.append(DilatedResBlock_SPADE(C, dilation=dilation_schedule[i % len(dilation_schedule)], norm1=norm1, norm2=norm2))

        self.eca = ECA(C, k_size=3) if use_eca else nn.Identity()

    @torch.no_grad()
    def set_noise_strength(self, gamma_c=None, beta_c=None, gamma_xy=None, beta_xy=None):
        if gamma_xy is not None or beta_xy is not None:
            self.spatial.set_strength(gamma_xy, beta_xy)
        for blk in self.blocks:
            blk.norm1.set_strength(gamma_c, beta_c)
            blk.norm2.set_strength(gamma_c, beta_c)

    def forward(self, observation):
        xt = observation[("nodes", "tasks", "attr")]
        h, batch_shape, single = flatten_task_grid(xt, length=self.length, width=self.width, in_channels=self.in_channels)
        H, W = self.length, self.width
        B = h.size(0)

        if self.add_z:
            z_ch = observation[("aux", "z_ch")]
            z_spa = observation[("aux", "z_spa")]
        else:
            z = self._z0
            if z.device != xt.device:
                z = z.to(device=xt.device)
            if z.dtype != xt.dtype:
                z = z.to(dtype=xt.dtype)
            z = z.expand(B, 1)
            z_ch = z
            z_spa = z

        if self.add_device_load or self.add_progress:
            aux_flat, _, _ = build_aux_features(
                observation,
                add_device_load=self.add_device_load,
                add_progress=self.add_progress,
            )
            if aux_flat is not None:
                aux_flat = _expand_to_batch(aux_flat, B)
                if self.add_z:
                    z_ch = torch.cat([z_ch, aux_flat], dim=-1)
                    z_spa = torch.cat([z_spa, aux_flat], dim=-1)
                else:
                    z = torch.cat([z_ch, aux_flat], dim=-1)
                    z_ch = z
                    z_spa = z

        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h, z_ch, z_spa)
        h = self.eca(h)

        if single:
            h = h.squeeze(0)
        else:
            C = h.size(1)
            h = h.view(*batch_shape, C, H, W)

        return (h,)


class DilationActorHead(nn.Module):
    in_keys = [("observation",), ("embed",)]
    out_keys = [("logits",)]

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int = 1,
        width: int = 4,
        length: int = 4,
        init_mode: str = "tiny",
        tiny_std: float = 1e-3,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__()
        self.width = int(width)
        self.length = int(length)
        self.Cin = int(input_dim) if input_dim is not None else None
        self.A = int(output_dim)
        self.debug = bool(debug)

        self.input_keys = ["embed"]
        self.output_dim = self.A

        self._init_mode = init_mode.lower()
        self._tiny_std = float(tiny_std)
        self._proj_initialized = input_dim is not None

        if input_dim is None:
            self.proj = nn.LazyConv2d(self.A, kernel_size=1, bias=True)
        else:
            self.proj = nn.Conv2d(self.Cin, self.A, kernel_size=1, bias=True)  # type: ignore[arg-type]
            self._init_proj()

    def _init_proj(self) -> None:
        mode = self._init_mode
        if mode == "zero":
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif mode == "tiny":
            nn.init.normal_(self.proj.weight, std=self._tiny_std)
            nn.init.zeros_(self.proj.bias)
        elif mode == "kaiming":
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
            nn.init.zeros_(self.proj.bias)
        else:
            raise ValueError(
                f"init_mode must be 'zero' | 'tiny' | 'kaiming', got {mode!r}"
            )

    def forward(self, obs, embed):
        if embed.dim() == 3:
            h = embed.unsqueeze(0)
            single = True
        else:
            h = embed
            single = False

        *B, C, H, W = h.shape
        h = embed.view(-1, C, H, W)
        logits_hw = self.proj(h)
        if not self._proj_initialized:
            if not hasattr(self.proj, "has_uninitialized_params") or not self.proj.has_uninitialized_params():  # type: ignore[attr-defined]
                self._init_proj()
                self._proj_initialized = True
        logits = logits_hw.permute(0, 2, 3, 1).reshape(h.size(0), H * W, self.A)
        logits = logits.view(*B, H * W, self.A)
        return logits[0] if single else logits


class DilationCriticHead(nn.Module):
    in_keys = [("observation",), ("embed",)]
    out_keys = [("state_value",)]

    def __init__(
        self,
        input_dim: int | None = None,
        z_dim: int = 8,
        proj_dim: int = 8,
        hidden_channels: int = 128,
        tiny_std: float = 1e-3,
        add_gap: bool = True,
        add_z: bool = False,
        add_progress: bool = True,
        add_device_load: bool = False,
        n_devices: int = 5,
        **_ignored,
    ):

        super().__init__()
        P = int(proj_dim)
        Dz = int(z_dim) * 2

        self._mix_initialized = input_dim is not None
        if input_dim is None:
            self.mix = nn.LazyConv2d(P, kernel_size=1, bias=False)
        else:
            C = int(input_dim)
            self.mix = nn.Conv2d(C, P, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.mix.weight, nonlinearity="relu")
            self._mix_initialized = True

        self.add_z = bool(add_z)
        self.add_device_load = bool(add_device_load)
        self.add_progress = bool(add_progress)

        self.attn = nn.Conv2d(P, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.attn.weight, std=tiny_std)
        nn.init.zeros_(self.attn.bias)

        self.add_gap = bool(add_gap)
        mlp_in = (
            (2 * P if self.add_gap else P)
            + (Dz if add_z else 0)
            + (3 * n_devices if add_device_load else 0)
            + (2 if add_progress else 0)
        )
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=tiny_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs, embed):
        z_f = None
        if self.add_z:
            z_ch = obs[("aux", "z_ch")]
            z_spa = obs[("aux", "z_spa")]
            z_f = torch.cat([z_ch, z_spa], dim=-1).reshape(-1, z_ch.size(-1) + z_spa.size(-1))

        *batch, C, H, W = embed.shape
        B = math.prod(batch) if batch else 1
        embed_f = embed.reshape(-1, C, H, W)

        Fm = F.silu(self.mix(embed_f), inplace=True)
        if not self._mix_initialized:
            if not hasattr(self.mix, "has_uninitialized_params") or not self.mix.has_uninitialized_params():  # type: ignore[attr-defined]
                nn.init.kaiming_normal_(self.mix.weight, nonlinearity="relu")
                self._mix_initialized = True
        scores = self.attn(Fm)
        attn = scores.flatten(2).softmax(dim=-1).view(B, 1, H, W)
        pooled_attn = (Fm * attn).sum(dim=(2, 3))

        if self.add_gap:
            pooled_gap = Fm.mean(dim=(2, 3))
            pooled = torch.cat([pooled_attn, pooled_gap], dim=1)
        else:
            pooled = pooled_attn

        if self.add_z:
            pooled = torch.cat([pooled, z_f], dim=-1)

        aux_flat, _, _ = build_aux_features(obs, add_device_load=self.add_device_load, add_progress=self.add_progress)
        if aux_flat is not None:
            aux_flat = _expand_to_batch(aux_flat, pooled.size(0))
            pooled = torch.cat([pooled, aux_flat], dim=-1)

        v = self.mlp(pooled).squeeze(-1)
        v = v.view(*batch, 1)
        return v
