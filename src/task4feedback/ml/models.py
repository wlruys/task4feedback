from torch_geometric.utils import coalesce, remove_self_loops, to_dense_adj

from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Dict, Optional, Self, Any
from typing import Optional, Callable, Union
from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import observation_to_heterodata, observation_to_heterodata_truncate
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Sequence
from torch_geometric.nn.norm import GraphNorm, LayerNorm, MessageNorm
from torch_geometric.nn import MessagePassing 

# from task4feedback.interface.wrappers import (
#     observation_to_heterodata_truncate as observation_to_heterodata,
# )
from torch import Tensor, dtype
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch.profiler import record_function

from tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
    GATConv,
    GraphConv,
    GCNConv,
    SimpleConv,
    EdgeConv,
    global_mean_pool,
    global_add_pool,
    SAGPooling,
    HeteroConv,
    SAGEConv,
)
import numpy as np
import time
import torch.nn.functional as F
import math
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

def _tiny_last_linear(seq: nn.Sequential, std: float = 1e-4):
    last = None
    for m in reversed(seq):
        if isinstance(m, nn.Linear):
            last = m
            break
    assert last is not None
    nn.init.normal_(last.weight, std=std)
    if last.bias is not None:
        nn.init.zeros_(last.bias)

def kaiming_init(layer, a=0.01, mode="fan_in", nonlinearity="leaky_relu"):
    """
    Initializes a layer with Kaiming He initialization.
    """
    print(f"Initializing layer {layer} with Kaiming He initialization")
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def xavier_init(layer, gain=1.0):
    """
    Initializes a layer with Xavier initialization.
    """
    print(f"Initializing layer {layer} with Xavier initialization")
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def orthogonal_init(layer, gain=1.0):
    """
    Initializes a layer with orthogonal initialization.
    """
    print(f"Initializing layer {layer} with Orthogonal initialization")
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None, dim: int = -1):
    if mask is None:
        return F.softmax(scores, dim=dim)
    mask = mask.to(dtype=torch.bool)
    mask_f = mask.to(dtype=scores.dtype)
    all_masked = ~mask.any(dim=dim, keepdim=True)

    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    # Avoid NaNs from softmax(all -inf): use a harmless value for fully-masked rows.
    masked_scores = masked_scores.masked_fill(all_masked, 0.0)

    weights = torch.softmax(masked_scores, dim=dim) * mask_f
    Z = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    out = weights / Z
    return torch.where(all_masked, torch.zeros_like(out), out)

def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int = 1, keepdim: bool = False):
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    num = (x * m).sum(dim=dim, keepdim=keepdim)
    den = m.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return num / den

def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)

def select_active_candidates(tasks: torch.Tensor, active_per_batch: torch.Tensor, max_candidates: int) -> torch.Tensor:
    N, C = tasks.shape 
    print("select_active_candidates: tasks shape", tasks.shape)
    #TODO: Reimplement 



class BatchWrapper(nn.Module):
    def __init__(self, network: nn.Module, device: Optional[str] = "cpu"):
        super(BatchWrapper, self).__init__()
        self.network = network

        self.register_parameter("dummy_param_0", nn.Parameter(torch.randn(1)))

    def _is_batch(self, obs: TensorDict) -> bool:
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(self, obs: TensorDict, is_batch: bool = False) -> HeteroData | Batch:
        if not is_batch:
            return obs["hetero_data"]

        # Otherwise form batch

        hetero_data_list = obs["hetero_data"]

        print(obs.shape, len(obs))

        batches = []

        for hlist in hetero_data_list:
            batches.append(Batch.from_data_list(hlist))

        # hetero_batch = Batch.from_data_list(hetero_data_list)

        print(batches)

        return batches

    def forward(self, obs: TensorDict):
        is_batch = self._is_batch(obs)

        with torch.no_grad():
            data = self._convert_to_heterodata(obs, is_batch)

        out = self.network(data)
        return out


class HeteroDataWrapper(nn.Module):
    def __init__(self, device: Optional[str] = "cpu"):
        super(HeteroDataWrapper, self).__init__()

        self.register_parameter("dummy_param_0", nn.Parameter(torch.randn(1)))

    def _is_batch(self, obs: TensorDict) -> bool:
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(
        self,
        obs: TensorDict,
        is_batch: bool = False,
        actions: Optional[TensorDict] = None,
    ) -> HeteroData:
        is_cuda = any(p.is_cuda for p in self.parameters())

        if not is_batch:
            if actions is not None:
                _obs = observation_to_heterodata_truncate(obs, actions=actions)
            else:
                _obs = observation_to_heterodata_truncate(obs)

            if is_cuda:
                _obs = _obs.to("cuda", non_blocking=True)

            return _obs

        # Otherwise we're batching, possibly over multiple batch dimensions

        # flatten and save the batch size
        self.batch_size = obs.batch_size
        # print("1 BATCH SHAPE obs", obs.shape, obs.batch_size, self.batch_size)
        obs = obs.reshape(-1)
        if actions is not None and hasattr(actions, "reshape"):
            actions = actions.reshape(-1)

        _h_data = []

        # print("2 BATCH SHAPE obs", obs.shape, obs.batch_size, self.batch_size)

        for i in range(obs.batch_size[0]):
            if actions is not None:
                _obs = observation_to_heterodata_truncate(obs[i], actions=actions[i])
            else:
                _obs = observation_to_heterodata_truncate(obs[i])
            _h_data.append(_obs)

        batch_obs = Batch.from_data_list(_h_data)

        if isinstance(batch_obs, tuple):
            batch_obs = batch_obs[0]

        if is_cuda:
            batch_obs = batch_obs.to("cuda", non_blocking=True)

        return batch_obs

    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)

        with torch.no_grad():
            data = self._convert_to_heterodata(obs, is_batch, actions=actions)

        return data


@dataclass
class FeatureDimConfig:
    task_feature_dim: int = 12
    data_feature_dim: int = 5
    device_feature_dim: int = 12
    task_data_edge_dim: int = 3
    task_device_edge_dim: int = 2
    task_task_edge_dim: int = 1

    @staticmethod
    def from_observer(observer: ExternalObserver):
        # print(f"task_feature_dim: {observer.task_feature_dim}")
        # print(f"data_feature_dim: {observer.data_feature_dim}")
        # print(f"device_feature_dim: {observer.device_feature_dim}")
        # print(f"task_data_edge_dim: {observer.task_data_edge_dim}")
        # print(f"task_device_edge_dim: {observer.task_device_edge_dim}")
        # print(f"task_task_edge_dim: {observer.task_task_edge_dim}")

        return FeatureDimConfig(
            task_feature_dim=observer.task_feature_dim,
            data_feature_dim=observer.data_feature_dim,
            device_feature_dim=observer.device_feature_dim,
            task_data_edge_dim=observer.task_data_edge_dim,
            task_device_edge_dim=observer.task_device_edge_dim,
            task_task_edge_dim=observer.task_task_edge_dim,
        )

    @staticmethod
    def from_config(other: Self, **overrides):
        return FeatureDimConfig(
            task_feature_dim=overrides.get("task_feature_dim", other.task_feature_dim),
            data_feature_dim=overrides.get("data_feature_dim", other.data_feature_dim),
            device_feature_dim=overrides.get("device_feature_dim", other.device_feature_dim),
            task_data_edge_dim=overrides.get("task_data_edge_dim", other.task_data_edge_dim),
            task_device_edge_dim=overrides.get("task_device_edge_dim", other.task_device_edge_dim),
            task_task_edge_dim=overrides.get("task_task_edge_dim", other.task_task_edge_dim),
        )


@dataclass
class LayerConfig:
    hidden_channels: int = 16
    n_heads: int = 1
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None


class OutputHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_channels: int, output_dim: int, activation: DictConfig = None, initialization: DictConfig = None, layer_norm: bool = True, debug: bool = False, **_ignored
    ):
        super(OutputHead, self).__init__()
        self.debug = debug
        if initialization is None:
            layer1_init = kaiming_init
            layer2_init = kaiming_init
        else:
            layer1_init = call(initialization["layer1"])
            layer2_init = call(initialization["layer2"])

        layers = []
        layers.append(layer1_init(nn.Linear(input_dim, hidden_channels)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_channels))
        layers.append(instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01))
        layers.append(layer2_init(nn.Linear(hidden_channels, output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.debug:
            print("[OutputHead] input {x.shape}")
        return self.network(x)


class LogitStabilizer(nn.Module):
    def __init__(
        self,
        init_tau: float = 2.0,
        learnable: bool = True,
        min_tau: float = 1.0,
        max_tau: Optional[float] = None,
    ):
        super().__init__()
        if min_tau <= 0:
            raise ValueError("min_tau must be > 0")
        if max_tau is not None and (max_tau <= min_tau):
            raise ValueError("max_tau must be None or > min_tau")
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau) if max_tau is not None else None
        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log(), requires_grad=learnable)

    @property
    def tau(self) -> torch.Tensor:
        tau = self.log_tau.exp()
        if self.max_tau is not None:
            tau = tau.clamp(self.min_tau, self.max_tau)
        else:
            tau = tau.clamp_min(self.min_tau)
        return tau

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 0:
            return logits  # nothing to stabilize
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits / self.tau


class LogitsOutputHead(OutputHead):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
        logit_stabilizer: Optional[LogitStabilizer] = None,
        debug: bool = False,
    ):
        super(LogitsOutputHead, self).__init__(input_dim, hidden_channels, output_dim, activation=activation, initialization=initialization, layer_norm=layer_norm, debug=debug)
        if logit_stabilizer is None:
            self.logit_stabilizer = LogitStabilizer()
        else:
            self.logit_stabilizer = logit_stabilizer

    def forward(self, x):
        if self.debug:
            print("[LogitsOutputHead] input {x.shape}")
        logits = super().forward(x)
        logits = self.logit_stabilizer(logits)
        return logits

    

class VectorValueHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_channels: int = 64,
            output_dim: int = 1,
            proj_dim: int = 32,
            activation: DictConfig = None,
            initialiation: DictConfig = None,
            layer_norm: bool = True,
            debug: bool = False,
            add_progress: bool = True,
            add_device_load: bool = True,
            n_devices: int = 5,
            **_ignored
        ):

        super(VectorValueHead, self).__init__()

        self.add_progress = add_progress
        self.add_device_load = add_device_load

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.side_info_dim = 0

        if self.add_progress:
            self.side_info_dim += 2

        if self.add_device_load:
            self.side_info_dim += 3 * n_devices

        if self.side_info_dim > 0:
            self.side_info_mlp = nn.Sequential(
                nn.Linear(self.side_info_dim, self.side_info_dim),
                nn.LeakyReLU(negative_slope=0.01),
            )
        else:
            self.side_info_mlp = nn.Identity()

        self.attn_in_dim = input_dim + self.side_info_dim 
        # self.ln = nn.LayerNorm(self.attn_in_dim)
        # self.gate_W = nn.Linear(self.attn_in_dim, proj_dim, bias=True)
        # self.gate_v = nn.Linear(proj_dim, 1, bias=False)

        pooled_dim = self.attn_in_dim


        self.value_mlp = OutputHead(
            input_dim=pooled_dim,
            hidden_channels=hidden_channels,
            output_dim=output_dim,
            activation=activation,
            initialization=initialiation,
            layer_norm=layer_norm,
            debug=debug,
        )

    def forward(self, obs, emb):
        *batch, C, k = emb.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        emb = emb.reshape(B, C, k)

        mask = obs["aux", "candidate_mask"]
        mask = mask.reshape(B, C)

        side_info = None 
        if self.add_device_load:
            device_load = obs["aux", "device_load"]
            device_memory = obs["aux", "device_memory"]
            device_feat = torch.cat([device_load, device_memory], dim=-1)  # [B, 3*n_devices]
            device_feat = device_feat.reshape(-1, device_feat.size(-1))  # [B, 3*n_devices]
            side_info = device_feat

        if self.add_progress:
            time_feature = obs["aux", "time"] / obs["aux", "baseline"]
            progress_feature = obs["aux", "progress"]
            time_feature = time_feature.reshape(-1, 1)  # [B, 1]
            progress_feature = progress_feature.reshape(-1, 1)  # [B, 1]
            prog_feats = torch.cat([time_feature, progress_feature], dim=-1)  # [B, 2]
            if side_info is None:
                side_info = prog_feats
            else:
                side_info = torch.cat([side_info, prog_feats], dim=-1)  # [B, side_info_dim]

        if side_info is not None:
            side_info_emb = self.side_info_mlp(side_info)  # [B, side_info_dim]
            side_info_emb = side_info_emb.unsqueeze(1).expand(-1, C, -1)  # [B, C, side_info_dim]
            emb = torch.cat([emb, side_info_emb], dim=-1)  # [B, C, k + side_info_dim]

        # m_pooled = masked_mean(emb, mask=mask, dim=1)  # [B, k + side_info_dim]
        # m_pooled = m_pooled.view(*batch, -1)

        # h = torch.tanh(self.gate_W(self.ln(emb)))  # [B, C, proj_dim]
        # scores = self.gate_v(h).squeeze(-1)  # [B, C]
        # attn_weights = masked_softmax(scores, mask=mask, dim=-1)  # [B, C]
        # attn_pooled = torch.einsum("bc, bcd -> bd", attn_weights, emb)  # [B, k + side_info_dim]
        # attn_pooled = attn_pooled.view(*batch, -1)

        # pooled = torch.cat([m_pooled, attn_pooled], dim=-1)  # [B, 2*(k + side_info_dim)]

        weights = mask.to(dtype=emb.dtype).unsqueeze(-1)  # [B, C, 1]
        pooled = (emb * weights).sum(dim=1)
        pooled = pooled.view(*batch, -1)
        out = self.value_mlp(pooled)
        return out

class VectorPolicyHead(OutputHead):
    def __init__(self, *args, **kwargs):
        super(VectorPolicyHead, self).__init__(*args, **kwargs)

    def forward(self, obs, emb):
        mask = obs["aux", "candidate_mask"]
        *batch, C, k = emb.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        emb = emb.reshape(B, C, k)
        mask = mask.reshape(B, C)

        out = super().forward(emb)
        out = out.view(*batch, C, -1)
        return out
    
class GNNValueHead(OutputHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, obs, emb):
        mask = obs["aux", "candidate_mask"]
        print(f"GNNValueHead forward: emb shape {emb.shape}, mask shape {mask.shape}")
        out = super().forward(emb)
        return out
    
class GNNPolicyHead(OutputHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs, emb):
        mask = obs["aux", "candidate_mask"]
        print(f"GNNPolicyHead forward: emb shape {emb.shape}, mask shape {mask.shape}")
        out = super().forward(emb)
        return out 


class VectorStateNet(nn.Module):

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
        super(VectorStateNet, self).__init__()
        self.feature_config = feature_config
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.hidden_channels = hidden_channels
        self.k = len(self.hidden_channels)
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)

        def make_activation(activation_config):
            return instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01)

        layer_init = call(initialization if initialization else kaiming_init)
        input_dim = feature_config.task_feature_dim

        if add_progress:
            input_dim += 2

        if add_device_load:
            input_dim += 3 * n_devices

        if self.k == 0:
            self.layers = nn.Identity()
            self.output_dim = input_dim
        else:
            # Build k MLP layers
            layers = []
            layer_channels = input_dim
            for i in range(self.k):
                layer_channels = hidden_channels[i]
                layers.append(layer_init(nn.Linear(input_dim, layer_channels)))
                if layer_norm:
                    layers.append(nn.LayerNorm(layer_channels))
                layers.append(make_activation(activation))
                input_dim = layer_channels

            self.layers = nn.Sequential(*layers)
            self.output_dim = layer_channels

            self.output_keys = ["embed"]

    def forward(self, tensordict: TensorDict):
        task_features = tensordict["nodes", "tasks", "attr"]
        candidate_counts = tensordict["aux", "candidates", "count"]
        *batch, C, k = task_features.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        task_features = task_features.reshape(B, C, k)

        if self.add_device_load:
            device_load = tensordict["aux", "device_load"]
            device_memory = tensordict["aux", "device_memory"]
            device_feat = torch.cat([device_load, device_memory], dim=-1)  # [B, 3*n_devices]
            device_feat = device_feat.reshape(-1, device_feat.size(-1))  # [B, 3*n_devices]
            # expand to match task features
            device_feat = device_feat.unsqueeze(1).expand(-1, C, -1)  # [B, C, 3*n_devices]
            task_features = torch.cat([task_features, device_feat], dim=-1)  # [B, C, k + 3*n_devices]

        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]
            time_feature = time_feature.reshape(-1, 1)  # [B, 1]
            progress_feature = progress_feature.reshape(-1, 1)  # [B, 1]
            prog_feats = torch.cat([time_feature, progress_feature], dim=-1)  # [B, 2]
            prog_feats = prog_feats.unsqueeze(1).expand(-1, C, -1)  # [B, C, 2]
            task_features = torch.cat([task_features, prog_feats], dim=-1)  # [B, C, k + 2]

        task_activations = self.layers(task_features)
        task_activations = task_activations.view(*batch, C, self.hidden_channels[-1])

        return task_activations


class VectorFiLMStateNet(nn.Module):

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
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.hidden_channels = hidden_channels
        self.k = len(self.hidden_channels)
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.pool_film = bool(pool_film)

        layer_init = call(initialization if initialization else kaiming_init)
        input_dim = feature_config.task_feature_dim

        g_dim = 0

        if add_progress:
            g_dim += 2

        if add_device_load:
            g_dim += 3 * n_devices

        # Build per-layer modules so we can do: Linear -> (LN) -> FiLM -> Act
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01)
        self.film_generators = nn.ModuleList() if g_dim > 0 else None

        self.rezeros = nn.ParameterList()
        self.gamma_ranges = nn.ParameterList()

        if self.k == 0:
            self.layers = nn.Identity()
            self.output_dim = input_dim
        else:
            in_ch = input_dim
            for i in range(self.k):
                out_ch = hidden_channels[i]
                self.linears.append(layer_init(nn.Linear(in_ch, out_ch)))
                self.norms.append(nn.LayerNorm(out_ch))

                if self.film_generators is not None:
                    mlp = nn.Sequential(layer_init(nn.Linear(g_dim, out_ch)),
                        nn.SiLU(),
                        layer_init(nn.Linear(out_ch, 2 * out_ch))
                    )
                    _tiny_last_linear(mlp, std=1e-4)
                    self.film_generators.append(mlp)
                    self.rezeros.append(nn.Parameter(torch.tensor(0.0)))
                    self.gamma_ranges.append(nn.Parameter(torch.full((out_ch,), 0.05)))
                in_ch = out_ch
            self.output_dim = in_ch
            self.output_keys = ["embed"]

            self.gate_mlp = nn.Sequential(
                nn.Linear(out_ch, out_ch),
                nn.SiLU(),
                nn.Linear(out_ch, 2 * out_ch),
            )
            self.gate_rezero = nn.Parameter(torch.tensor(0.0))
            self.gate_gamma_range = nn.Parameter(torch.full((out_ch,), 0.05))
            self.gate_norm = nn.LayerNorm(out_ch, elementwise_affine=False)
            _tiny_last_linear(self.gate_mlp, std=1e-4)

    def forward(self, tensordict: TensorDict):
        task_features = tensordict["nodes", "tasks", "attr"]
        candidate_counts = tensordict["aux", "candidates", "count"]
        *batch, C, k = task_features.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        x = task_features.reshape(B, C, k)

        cand_mask = tensordict["aux", "candidate_mask"].reshape(B, C).to(torch.bool)

        # Build conditioning tensor g per candidate (B, C, g_dim) if enabled
        g = None
        if self.add_device_load:
            device_load = tensordict["aux", "device_load"]
            device_memory = tensordict["aux", "device_memory"]
            device_feat = torch.cat([device_load, device_memory], dim=-1)  # [B, 3*n_devices]
            device_feat = device_feat.reshape(-1, device_feat.size(-1)).unsqueeze(1).expand(-1, C, -1)  # [B, C, 3*n_devices]
            g = device_feat
        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]
            prog_feats = torch.stack([time_feature.reshape(-1), progress_feature.reshape(-1)], dim=-1)  # [B, 2]
            prog_feats = prog_feats.unsqueeze(1).expand(-1, C, -1)  # [B, C, 2]
            g = prog_feats if g is None else torch.cat([g, prog_feats], dim=-1)  # [B, C, g_dim]

        if self.k == 0:
            return x.view(*batch, C, self.output_dim)

        for i in range(self.k):
            x = self.linears[i](x) # (B, C, H_i)
            nx = self.norms[i](x)  # (B, C, H_i)

            if g is not None:
                H_i = x.size(-1)
                gb = self.film_generators[i](g.reshape(-1, g.size(-1)))  # (B*C, 2*H_i)
                gamma_raw, beta = gb.chunk(2, dim=-1)
                gamma = self.gamma_ranges[i] * (2*torch.sigmoid(gamma_raw) - 1)
                gamma = gamma.view(B, C, H_i)
                beta = beta.view(B, C, H_i)
                u = (1.0 + gamma) * nx + beta
                x = x + self.rezeros[i] * u

            x = cand_mask.unsqueeze(-1)*x
            x = self.act(x)

        if self.pool_film:
            pooled = (x * cand_mask.unsqueeze(-1)).sum(dim=1) / cand_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

            H = x.size(-1)
            xn = self.gate_norm(x)
            gb = self.gate_mlp(pooled)  # (B, 2H)
            gamma_raw, beta = gb.chunk(2, dim=-1)  # (B,H), (B,H)
            gamma = self.gate_gamma_range * (2*torch.sigmoid(gamma_raw) - 1)
            gamma = gamma.view(B, H)
            beta = beta.view(B, H)
            u = (1.0 + gamma).unsqueeze(1) * xn + beta.unsqueeze(1)
            x = x + self.gate_rezero * u
            x = cand_mask.unsqueeze(-1)*x

        task_activations = x.view(*batch, C, self.output_dim)
        return task_activations

class VectorAttnStateNet(nn.Module):

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
        coord_layers: int = 1,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        ffn_multiplier: float = 2.0,
        **_ignored,
    ):
        super(VectorAttnStateNet, self).__init__()
        self.feature_config = feature_config
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.hidden_channels = hidden_channels
        self.k = len(self.hidden_channels)
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)

        self.coord_layers = int(coord_layers)
        self.num_heads = int(num_heads)
        self.attn_dropout = float(attn_dropout)
        self.ffn_multiplier = float(ffn_multiplier)

        def make_activation(activation_config):
            return instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01)

        layer_init = call(initialization if initialization else kaiming_init)
        input_dim = feature_config.task_feature_dim

        if add_progress:
            input_dim += 2

        if add_device_load:
            input_dim += 3 * n_devices

        if self.k == 0:
            self.layers = nn.Identity()
            self.output_dim = input_dim
        else:
            layers = []
            layer_channels = input_dim
            for i in range(self.k):
                layer_channels = hidden_channels[i]
                layers.append(layer_init(nn.Linear(input_dim, layer_channels)))
                if layer_norm:
                    layers.append(nn.LayerNorm(layer_channels))
                layers.append(make_activation(activation))
                input_dim = layer_channels

            self.layers = nn.Sequential(*layers)
            self.output_dim = layer_channels

        if self.coord_layers > 0:
            d = self.output_dim
            self._sab_blocks = nn.ModuleList()
            for _ in range(self.coord_layers):
                block = nn.ModuleDict({
                    "prenorm": nn.LayerNorm(d),
                    "attn": nn.MultiheadAttention(
                        embed_dim=d, num_heads=self.num_heads,
                        dropout=self.attn_dropout, batch_first=True
                    ),
                    "norm1": nn.LayerNorm(d),
                    "ffn": nn.Sequential(
                        layer_init(nn.Linear(d, int(self.ffn_multiplier * d))),
                        make_activation(activation),
                        layer_init(nn.Linear(int(self.ffn_multiplier * d), d)),
                    ),
                    "norm2": nn.LayerNorm(d),
                })
                self._sab_blocks.append(block)

        self.output_keys = ["embed"]

    def forward(self, tensordict: TensorDict):
        task_features = tensordict["nodes", "tasks", "attr"]
        candidate_counts = tensordict["aux", "candidates", "count"]
        *batch, C, k = task_features.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        task_features = task_features.reshape(B, C, k)

        if self.add_device_load:
            device_load = tensordict["aux", "device_load"]
            device_memory = tensordict["aux", "device_memory"]
            device_feat = torch.cat([device_load, device_memory], dim=-1)  # [B, 3*n_devices]
            device_feat = device_feat.reshape(-1, device_feat.size(-1)).unsqueeze(1).expand(-1, C, -1)
            task_features = torch.cat([task_features, device_feat], dim=-1)

        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]
            prog_feats = torch.stack([time_feature.reshape(-1), progress_feature.reshape(-1)], dim=-1)
            prog_feats = prog_feats.unsqueeze(1).expand(-1, C, -1)
            task_features = torch.cat([task_features, prog_feats], dim=-1)

        x = self.layers(task_features)  # (B, C, D=self.output_dim)

        cand_mask = tensordict["aux", "candidate_mask"].reshape(B, C).to(torch.bool)  # [B, C]

        x = x * cand_mask.unsqueeze(-1)  # Mask out non-candidates

        if self.coord_layers > 0:
            for blk in self._sab_blocks:
                # Self-attention with residual + norm
                ax = blk["prenorm"](x)
                attn_out, _ = blk["attn"](
                    ax, ax, x,
                    need_weights=False,
                )  # (B, C, D)
                x = blk["norm1"](x + attn_out)
                ffn_out = blk["ffn"](x)
                x = blk["norm2"](x + ffn_out)

        task_activations = x
        task_activations = task_activations.view(*batch, C, self.output_dim)
        return task_activations

class VectorDCGStateNet(nn.Module):

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
        dcg_layers: int = 3,      # message-passing rounds
        dcg_rank: int = 8,       # rank for pair weights w_ij
        **_ignored,
    ):
        super(VectorDCGStateNet, self).__init__()
        self.feature_config = feature_config
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.hidden_channels = hidden_channels
        self.k = len(self.hidden_channels)
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)

        def make_activation(activation_config):
            return nn.SiLU()

        layer_init = call(initialization if initialization else kaiming_init)
        input_dim = feature_config.task_feature_dim
        if add_progress:
            input_dim += 2
        if add_device_load:
            input_dim += 3 * n_devices

        if self.k == 0:
            self.layers = nn.Identity()
            self.output_dim = input_dim
        else:
            layers = []
            layer_channels = input_dim
            for i in range(self.k):
                layer_channels = hidden_channels[i]
                layers.append(layer_init(nn.Linear(input_dim, layer_channels)))
                if layer_norm:
                    layers.append(nn.LayerNorm(layer_channels))
                layers.append(make_activation(activation))
                input_dim = layer_channels
            self.layers = nn.Sequential(*layers)
            self.output_dim = layer_channels

        self.dcg_layers = int(dcg_layers)
        self.dcg_rank = int(dcg_rank)
        d = self.output_dim
        R = self.dcg_rank
        self._dcg_blocks = nn.ModuleList()
        for _ in range(self.dcg_layers):
            self._dcg_blocks.append(nn.ModuleDict({
                "prenorm": nn.LayerNorm(d),                   
                "proj_i":  layer_init(nn.Linear(d, R, bias=False)),
                "proj_j":  layer_init(nn.Linear(d, R, bias=False)),
                "msg":     nn.Sequential(
                                layer_init(nn.Linear(d, 2*d)),
                                nn.SiLU(),
                                layer_init(nn.Linear(2*d, d))),
                "local":   nn.Sequential(
                                layer_init(nn.Linear(d, 2*d)),
                                nn.SiLU(),
                                layer_init(nn.Linear(2*d, d))),
                "norm":    nn.LayerNorm(d),
            }))
        self._dcg_scale = (self.dcg_rank ** -0.5)

        self.output_keys = ["embed"]

    def forward(self, tensordict: TensorDict):
        task_features = tensordict["nodes", "tasks", "attr"]
        candidate_counts = tensordict["aux", "candidates", "count"]
        *batch, C, k = task_features.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        task_features = task_features.reshape(B, C, k)

        if self.add_device_load:
            device_load = tensordict["aux", "device_load"]
            device_memory = tensordict["aux", "device_memory"]
            device_feat = torch.cat([device_load, device_memory], dim=-1)  # [B, 3*n_devices]
            device_feat = device_feat.reshape(-1, device_feat.size(-1)).unsqueeze(1).expand(-1, C, -1)
            task_features = torch.cat([task_features, device_feat], dim=-1)

        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]
            prog_feats = torch.stack([time_feature.reshape(-1), progress_feature.reshape(-1)], dim=-1)
            prog_feats = prog_feats.unsqueeze(1).expand(-1, C, -1)
            task_features = torch.cat([task_features, prog_feats], dim=-1)

        x = self.layers(task_features)  # (B, C, D)
        cand_mask = tensordict["aux", "candidate_mask"].reshape(B, C).to(torch.bool)  # (B, C)

        adj = tensordict.get(("aux", "candidate_adj"), None)
        if adj is None:
            eye = torch.eye(C, dtype=torch.bool, device=x.device).unsqueeze(0)  # (1, C, C)
            adj = cand_mask.unsqueeze(1) & cand_mask.unsqueeze(2) & (~eye)     # (B, C, C)

        if self.dcg_layers > 0:
            for blk in self._dcg_blocks:
                x_n = blk["prenorm"](x)
                Pi = blk["proj_i"](x_n)                     # (B,C,R)
                Pj = blk["proj_j"](x_n)                     # (B,C,R)
                scores = torch.einsum("bir,bjr->bij", Pi, Pj) * self._dcg_scale
                w = F.softplus(scores) * adj.to(scores.dtype)

                deg = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                w = w / deg

                m_j = blk["msg"](x_n)                       # (B,C,D)
                agg = torch.einsum("bij,bjd->bid", w, m_j)  # (B,C,D)

                x = blk["norm"](x + blk["local"](x_n) + agg)

        task_activations = x.view(*batch, C, self.output_dim)
        return task_activations


class _FiLM(nn.Module):
    def __init__(self, node_types: List[str], num_layers: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.mod = nn.ModuleDict(
            {
                nt: nn.ModuleList([nn.Sequential(nn.Linear(cond_dim, max(64, hidden_dim // 2)), nn.SiLU(), nn.Linear(max(64, hidden_dim // 2), 2 * hidden_dim)) for _ in range(num_layers)])
                for nt in node_types
            }
        )

        for nt in node_types:
            for l in range(num_layers):
                _tiny_last_linear(self.mod[nt][l], std=1e-4)

    def forward(self, x_dict, batch_dict, g: Tensor, layer_idx: int):
        if g.dim() == 1:
            g = g.unsqueeze(0)  # [1, G]
        B = g.size(0)
        out = {}
        for nt, x in x_dict.items():
            if x is None:
                out[nt] = None
                continue
            gb = self.mod[nt][layer_idx](g)  # [B, 2C]
            gamma, beta = gb.chunk(2, dim=-1)  # [B,C], [B,C]
            b = batch_dict.get(nt, None)
            gamma = 1.0 + 0.5 * torch.tanh(gamma)
            beta = 0.5 * beta

            if b is None:
                g_nodes = gamma[0].expand_as(x)
                b_nodes = beta[0].expand_as(x)
            else:
                if b.max().item() >= B:
                    raise ValueError("Global vector batch size mismatches node batch indices.")
                g_nodes = gamma.index_select(0, b)
                b_nodes = beta.index_select(0, b)
            out[nt] = (g_nodes) * x + b_nodes
        return out


class GATStateNet(nn.Module):

    def _mask_edges(self, edge_index, edge_mask, edge_attr=None):
        mask = edge_mask.to(torch.bool)
        edge_index_masked = edge_index[:, mask]
        edge_attr_masked = edge_attr[mask] if edge_attr is not None else None
        return edge_index_masked, edge_attr_masked

    def __init__(self, feature_config: FeatureDimConfig, hidden_channels: int = 16, num_layers: int = 2, add_device_load: bool = False, add_progress: bool = False, n_devices: int = 5, **_ignored):
        print("Initializing GATStateNet")
        super(GATStateNet, self).__init__()
        self.feature_config = feature_config
        self.hidden_channels = hidden_channels

        self.convert_data = HeteroDataWrapper()

        self.act = nn.SiLU()

        g_dim = 0

        if add_progress:
            g_dim += 2

        if add_device_load:
            g_dim += 3 * n_devices

        self.num_layers = int(num_layers)
        self.g_dim = g_dim
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.n_devices = int(n_devices)

        self.stem_proj = nn.ModuleDict(
            {
                "tasks": Linear(int(feature_config.task_feature_dim), self.hidden_channels, bias=True),
                "data": Linear(int(feature_config.data_feature_dim), self.hidden_channels, bias=True),
            }
        )
        self.stem_norm = nn.ModuleDict(
            {
                "tasks": nn.LayerNorm(self.hidden_channels),
                "data": nn.LayerNorm(self.hidden_channels),
            }
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ("tasks", "to", "tasks"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="add", root_weight=False),
                ("tasks", "from", "tasks"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="add", root_weight=False),
                ("tasks", "read", "data"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="add", root_weight=False),
                ("data", "read", "tasks"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="add", root_weight=False),
            }
            # conv_dict = {
            #     ("tasks", "to", "tasks"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            #     # ("tasks", "from", "tasks"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            #     ("tasks", "read", "data"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            #     ("data", "read", "tasks"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            # }
            hetero_conv = HeteroConv(conv_dict, aggr="mean")
            self.convs.append(hetero_conv)

        self.norms = nn.ModuleDict(
            {
                "tasks": nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(num_layers + 1)]),
                "data": nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(num_layers + 1)]),
            }
        )

        self.beta = nn.ModuleDict(
            {
                "tasks": nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_layers)]),
                "data": nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_layers)]),
            }
        )
        for nt in self.beta.keys():
            for b in self.beta[nt]:
                # init to 0.5
                nn.init.constant_(b, 0.5)

        # self.post_norms = nn.ModuleDict(
        #     {
        #         "tasks": nn.ModuleList([MessageNorm(learn_scale=True) for _ in range(num_layers)]),
        #         "data": nn.ModuleList([MessageNorm(learn_scale=True) for _ in range(num_layers)]),
        #     }
        # )

        if self.add_device_load or self.add_progress:
            self.film = _FiLM(node_types=["tasks", "data"], num_layers=self.num_layers, cond_dim=int(self.g_dim), hidden_dim=self.hidden_channels)
        else:
            self.film = None  # No FiLM conditioning

        self.mlp_global_pool = nn.ModuleDict(
            {
                "tasks": nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels)),
                "data": nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels)),
            }
        )

        if self.add_device_load or self.add_progress:
            self.mlp_side_info = nn.Sequential(nn.Linear(self.g_dim, self.hidden_channels))
        else:
            self.mlp_side_info = None

        self.mlp_norm = nn.LayerNorm(8)

        _tiny_last_linear(self.mlp_global_pool["tasks"])
        _tiny_last_linear(self.mlp_global_pool["data"])
        if self.mlp_side_info is not None:
            _tiny_last_linear(self.mlp_side_info)

        nn.init.zeros_(self.stem_proj["tasks"].bias)
        nn.init.zeros_(self.stem_proj["data"].bias)

        self.output_dim = hidden_channels*4 if self.g_dim > 0 else hidden_channels*3

        self.output_keys = ["embed", "task_embed", "data_embed", "ptr"]

    def forward(self, tensordict: TensorDict):
        print("GATStateNet forward called")
        batch_size = tensordict.batch_size
        data = self.convert_data(tensordict)

        b_tasks = data["tasks"].batch if isinstance(data, Batch) else None
        b_data = data["data"].batch if isinstance(data, Batch) else None

        x_tasks = self.stem_proj["tasks"](data["tasks"].x)
        # x_tasks = self.stem_norm["tasks"](x_tasks)
        x_tasks = self.act(x_tasks)

        x_data = self.stem_proj["data"](data["data"].x)
        # x_data = self.stem_norm["data"](x_data)
        x_data = self.act(x_data)

        x_dict = {"tasks": x_tasks, "data": x_data}
        batch_dict = {"tasks": b_tasks, "data": b_data}

        tasks_read_data = data["tasks", "read", "data"].edge_index
        mask = data["tasks", "read", "data"].edge_attr
        masked_task_data, _ = self._mask_edges(edge_index=tasks_read_data, edge_mask=mask[:, 0])

        edge_index_dict = {
            ("tasks", "to", "tasks"): data["tasks", "to", "tasks"].edge_index,
            ("tasks", "from", "tasks"): data["tasks", "from", "tasks"].edge_index,
            ("tasks", "read", "data"): masked_task_data,
            ("data", "read", "tasks"): masked_task_data.flip(0),
        }

        g = None
        if self.film is not None:
            if self.add_progress:
                time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
                time_feature = time_feature.reshape(-1, 1)
                progress_feature = tensordict["aux", "progress"]
                progress_feature = progress_feature.reshape(-1, 1)
                g = torch.cat([time_feature, progress_feature], dim=-1)

            if self.add_device_load:
                device_load = tensordict["aux", "device_load"]
                device_memory = tensordict["aux", "device_memory"]
                device_load = device_load.reshape(-1, 2 * self.n_devices)
                device_memory = device_memory.reshape(-1, 1 * self.n_devices)

                if g is None:
                    g = torch.cat([device_load, device_memory], dim=-1)
                else:
                    g = torch.cat([g, device_load, device_memory], dim=-1)

        for l, conv in enumerate(self.convs):

            # pre-norm
            # x_pre = {nt: self.norms[nt][l](x_dict[nt]) for nt in x_dict.keys()}

            # conv
            x_new = conv(x_dict, edge_index_dict=edge_index_dict)

            # #post-norm
            # x_new  = {nt: self.post_norms[nt][l](x_dict[nt], x_new[nt]) for nt in x_new.keys()}

            x_new = {nt: self.norms[nt][l](x_new[nt]) for nt in x_new.keys()}

            # film
            if self.film is not None: 
                x_new = self.film(x_new, batch_dict, g=g, layer_idx=l)

            # activation
            x_new = {nt: self.act(x_new[nt]) for nt in x_new.keys()}

            # residual
            for nt in x_dict.keys():
                beta = self.beta[nt][l]
                beta = torch.sigmoid(beta)
                x_new[nt] = (1 - beta) * x_dict[nt] + beta * x_new[nt]

            # update for next layer
            x_dict = {nt: x_new[nt] for nt in x_new.keys()}

        # final norm
        # x_dict = {nt: self.norms[nt][-1](x_dict[nt]) for nt in x_dict.keys()}

        task_counts = tensordict["aux", "candidates", "count"]
        max_candidates = tensordict["aux", "candidates", "idx"].size(-1)
        print(f"max_candidates: {max_candidates}, task_counts: {task_counts}")

        if b_tasks is not None:
            ptr = data["tasks"].ptr
            idx = data["tasks"].ptr[:-1]
            x = x_dict["tasks"][idx]
        else:
            ptr = None 
            x = x_dict["tasks"][0]

        pooled_tasks = global_mean_pool(x_dict["tasks"], b_tasks)
        pooled_data = global_mean_pool(x_dict["data"], b_data)

        pt_f = pooled_tasks
        pd_f = pooled_data

        if self.mlp_side_info is not None:
            side_f = self.mlp_side_info(g)
            side_f = self.act(side_f)
            y = torch.cat([pt_f, pd_f, side_f], dim=-1)
        else:
            y = torch.cat([pt_f, pd_f], dim=-1)

        # y = self.mlp_norm(y)
        # y = self.act(y)

        if b_tasks is None:
            y = y.squeeze(0)

        # print(f"x shape before cat: {x.shape}, y shape: {y.shape}, batch_size: {batch_size}")

        x = torch.cat([x, y], dim=-1)
        x = x.reshape(*batch_size, -1, x.shape[-1])
        # print(f"x shape before return: {x.shape}")
        return x.select(dim=-2, index=0), 
    

class TaskIterationGNNStateNet(nn.Module):

    def _mask_edges(self, edge_index, edge_mask, edge_attr=None):
        mask = edge_mask.to(torch.bool)
        edge_index_masked = edge_index[:, mask]
        edge_attr_masked = edge_attr[mask] if edge_attr is not None else None
        return edge_index_masked, edge_attr_masked

    def __init__(
        self, 
        feature_config: FeatureDimConfig,
        hidden_channels: int = 16,
        n_heads: int = 2,
        add_device_load: bool = False,
        add_progress: bool = False,
        n_devices: int = 5,
        num_layers: int = 1,
        **_ignored,
    ):

        super(TaskIterationGNNStateNet, self).__init__()

        self.feature_config = feature_config
        self.n_heads = n_heads
        self.hidden_channels = hidden_channels
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.n_devices = int(n_devices)

        self.g_dim = 0
        if add_progress:
            self.g_dim += 2
        if add_device_load:
            self.g_dim += 3 * n_devices

        self.stem_prog = nn.ModuleDict(
            {
                "tasks": Linear(int(feature_config.task_feature_dim), hidden_channels, bias=True),
                "data": Linear(int(feature_config.data_feature_dim), hidden_channels, bias=True),
            }
        )

        self.stem_norm = nn.ModuleDict(
            {
                "tasks": nn.LayerNorm(hidden_channels),
                "data": nn.LayerNorm(hidden_channels),
            }
        )        

        self.convert_data = HeteroDataWrapper()

        self.gnn_tasks_data = GATv2Conv(
            (hidden_channels, hidden_channels),
            hidden_channels,
            heads=n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        )
        self.norm_tasks_data = nn.LayerNorm(hidden_channels)

        self.task_dependency_convs = nn.ModuleList()
        self.task_dependency_norms = nn.ModuleList()
        self.task_dependent_convs = nn.ModuleList()
        self.task_dependent_norms = nn.ModuleList()
        self.task_merge_mlps = nn.ModuleList()
        

        for _ in range(num_layers):
            self.task_dependency_convs.append(
                GATv2Conv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    add_self_loops=False,
                )
            )
            self.task_dependency_norms.append(nn.LayerNorm(hidden_channels))

            self.task_dependent_convs.append(
                GATv2Conv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    add_self_loops=False,
                )
            )
            self.task_dependent_norms.append(nn.LayerNorm(hidden_channels))

            self.task_merge_mlps.append(
                        nn.Sequential(
                            nn.Linear(hidden_channels *2, hidden_channels),
                            nn.LayerNorm(hidden_channels),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.Linear(hidden_channels, hidden_channels),
                        )
            )


        self.act = nn.LeakyReLU(negative_slope=0.01) 

        self.g_mlp = nn.Sequential(
            nn.Linear(self.g_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels, hidden_channels),
        ) if self.g_dim > 0 else None       

        self.output_dim = hidden_channels * 2 + (hidden_channels if self.g_dim > 0 else 0)
        self.output_keys = ["embed"]


    def forward(self, tensordict: TensorDict):
        batch_size = tensordict.batch_size
        data= self.convert_data(tensordict)

        b_tasks = data["tasks"].batch if isinstance(data, Batch) else None

        x_tasks = self.stem_prog["tasks"](data["tasks"].x)
        #x_tasks = self.stem_norm["tasks"](x_tasks)
        x_tasks = self.act(x_tasks)

        x_data = self.stem_prog["data"](data["data"].x)
        #x_data = self.stem_norm["data"](x_data)
        x_data = self.act(x_data)

        data_read_tasks = data["data", "read", "tasks"].edge_index
        mask = data["data", "read", "tasks"].edge_attr
        read_edges_masked, _ = self._mask_edges(edge_index=data_read_tasks, edge_mask=mask[:, 0])

        tasks_w_data = self.gnn_tasks_data(
            (x_data, x_tasks),
            read_edges_masked,
        )
        tasks_w_data = self.norm_tasks_data(tasks_w_data)
        tasks_w_data = self.act(tasks_w_data)

        x_tasks= tasks_w_data

        for l in range(len(self.task_dependency_convs)):

            x_tasks_in = self.task_dependency_convs[l](
                (x_tasks, x_tasks),
                data["tasks", "to", "tasks"].edge_index,
            )
            x_tasks_in = self.task_dependency_norms[l](x_tasks_in)
            x_tasks_in = self.act(x_tasks_in)

            x_tasks_out = self.task_dependent_convs[l](
                (x_tasks, x_tasks),
                data["tasks", "from", "tasks"].edge_index,
            )
            x_tasks_out = self.task_dependent_norms[l](x_tasks_out)
            x_tasks_out = self.act(x_tasks_out)

            x_new = torch.cat([x_tasks_in, x_tasks_out], dim=-1)
            x_new = self.task_merge_mlps[l](x_new)
            x_new = self.act(x_new)
            x_tasks = x_tasks + self.act(x_tasks)

        tasks_global = global_mean_pool(x_tasks, b_tasks)

        g = None 
        if self.g_dim > 0:
            if self.add_progress:
                time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
                time_feature = time_feature.reshape(-1, 1)
                progress_feature = tensordict["aux", "progress"]
                progress_feature = progress_feature.reshape(-1, 1)
                g = torch.cat([time_feature, progress_feature], dim=-1)

            if self.add_device_load:
                device_load = tensordict["aux", "device_load"]
                device_memory = tensordict["aux", "device_memory"]
                device_load = device_load.reshape(-1, 2 * self.n_devices)
                device_memory = device_memory.reshape(-1, 1 * self.n_devices)

                if g is None:
                    g = torch.cat([device_load, device_memory], dim=-1)
                else:
                    g = torch.cat([g, device_load, device_memory], dim=-1)

            g = self.g_mlp(g)
            g = self.act(g)

        if b_tasks is not None:
            idx = data["tasks"].ptr[:-1]
            x = x_tasks[idx]
        else:
            x = x_tasks[0]

        if b_tasks is None:
            tasks_global = tasks_global.squeeze(0)
            if g is not None:
                g = g.squeeze(0)

        x = torch.cat([x, tasks_global], dim=-1)
        if g is not None:
            x = torch.cat([x, g], dim=-1)
        
        x = x.reshape(*batch_size, -1, x.shape[-1])
        return x.select(dim=-2, index=0)



class DataIterationGNNStateNet(nn.Module):

    def _mask_edges(self, edge_index, edge_mask, edge_attr=None):
        mask = edge_mask.to(torch.bool)
        edge_index_masked = edge_index[:, mask]
        edge_attr_masked = edge_attr[mask] if edge_attr is not None else None
        return edge_index_masked, edge_attr_masked

    def __init__(
        self, 
        feature_config: FeatureDimConfig,
        hidden_channels: int = 16,
        n_heads: int = 1,
        add_device_load: bool = False,
        add_progress: bool = False,
        n_devices: int = 5,
        num_layers: int = 1,
        conv_type: str = "GATv2",
        use_norm: bool = False,
        **_ignored,
    ):

        super(DataIterationGNNStateNet, self).__init__()
        print("INITIALIZING DataIterationGNNStateNet")

        self.use_norm = bool(use_norm)
        self.feature_config = feature_config
        self.n_heads = n_heads
        self.hidden_channels = hidden_channels
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.n_devices = int(n_devices)
        self.num_layers = int(num_layers)

        self.g_dim = 0
        if add_progress:
            self.g_dim += 2
        if add_device_load:
            self.g_dim += 3 * n_devices

        self.stem_prog = nn.ModuleDict(
            {
                "tasks": Linear(int(feature_config.task_feature_dim), hidden_channels, bias=True),
                "data": Linear(int(feature_config.data_feature_dim), hidden_channels, bias=True),
            }
        )

        self.stem_norm = nn.ModuleDict(
            {
                "tasks": nn.LayerNorm(hidden_channels),
                "data": nn.LayerNorm(hidden_channels),
            }
        )        

        self.convert_data = HeteroDataWrapper()


        self.gnn_tasks_read_data = GATv2Conv(
            (hidden_channels, hidden_channels),
            hidden_channels,
            heads=n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        ) if conv_type == "GATv2" else SAGEConv(hidden_channels, hidden_channels, project=False, aggr="add", root_weight=True)

        self.gnn_tasks_from_tasks = GATv2Conv(
            (hidden_channels, hidden_channels),
            hidden_channels,
            heads=n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        ) if conv_type == "GATv2" else SAGEConv(hidden_channels, hidden_channels, project=False, aggr="add", root_weight=True)

        self.gnn_tasks_to_tasks = GATv2Conv(
            (hidden_channels, hidden_channels),
            hidden_channels,
            heads=n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        ) if conv_type == "GATv2" else SAGEConv(hidden_channels, hidden_channels, project=False, aggr="add", root_weight=True)

        self.norm_tasks_to_tasks = nn.LayerNorm(hidden_channels)
        self.norm_tasks_from_tasks = nn.LayerNorm(hidden_channels)
        self.norm_task_read_data = nn.LayerNorm(hidden_channels)

        self.task_merge_mlp = nn.Sequential(
            nn.Linear(hidden_channels *2, hidden_channels),
        )

        self.global_merge_mlp = nn.Sequential(
            nn.Linear(hidden_channels *2, hidden_channels),
        )
        self.global_merge_norm = nn.LayerNorm(hidden_channels)

        self.task_data_convs = nn.ModuleList()
        self.data_task_convs = nn.ModuleList()
        self.data_task_norms = nn.ModuleList()
        self.task_data_norms = nn.ModuleList()

        for l in range(self.num_layers):
            
            self.task_data_convs.append(
                GATv2Conv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    add_self_loops=False,
                )
            if conv_type == "GATv2" else SAGEConv(hidden_channels, hidden_channels, project=True, aggr="mean", root_weight=False))


            self.data_task_convs.append(
                GATv2Conv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    add_self_loops=False,
                )
            if conv_type == "GATv2" else SAGEConv(hidden_channels, hidden_channels, project=True, aggr="mean", root_weight=False))

            self.task_data_norms.append(nn.LayerNorm(hidden_channels))
            self.data_task_norms.append(nn.LayerNorm(hidden_channels))

        self.act = nn.LeakyReLU(negative_slope=0.01) 

        self.g_mlp = nn.Sequential(
            nn.Linear(self.g_dim, hidden_channels),
        ) if self.g_dim > 0 else None      

        self.g_norm = nn.LayerNorm(hidden_channels) if self.g_dim > 0 else None 

        self.output_dim = hidden_channels * 2 + (hidden_channels if self.g_dim > 0 else 0)
        self.output_keys = ["embed"]


    def forward(self, tensordict: TensorDict):
        batch_size = tensordict.batch_size
        data= self.convert_data(tensordict)

        b_tasks = data["tasks"].batch if isinstance(data, Batch) else None
        b_data = data["data"].batch if isinstance(data, Batch) else None
        x_tasks = self.stem_prog["tasks"](data["tasks"].x)
        #x_tasks = self.stem_norm["tasks"](x_tasks)
        x_tasks = self.act(x_tasks)

        x_data = self.stem_prog["data"](data["data"].x)
        #x_data = self.stem_norm["data"](x_data)
        x_data = self.act(x_data)

        data_read_tasks = data["data", "read", "tasks"].edge_index
        mask = data["data", "read", "tasks"].edge_attr
        read_edges_masked, _ = self._mask_edges(edge_index=data_read_tasks, edge_mask=mask[:, 0])

        tasks_read_data = self.gnn_tasks_read_data(
            (x_data, x_tasks),
            read_edges_masked,
        )
        if self.use_norm:
            tasks_read_data = self.norm_task_read_data(tasks_read_data)
        tasks_read_data = self.act(tasks_read_data)

        x_tasks= tasks_read_data

        tasks_from_tasks = self.gnn_tasks_from_tasks(
            (x_tasks, x_tasks),
            data["tasks", "from", "tasks"].edge_index,
        )
        tasks_from_tasks = self.norm_tasks_from_tasks(tasks_from_tasks)
        tasks_from_tasks = self.act(tasks_from_tasks)

        tasks_to_tasks = self.gnn_tasks_to_tasks(
            (x_tasks, x_tasks),
            data["tasks", "to", "tasks"].edge_index,
        )
        if self.use_norm:
            tasks_to_tasks = self.norm_tasks_to_tasks(tasks_to_tasks)
        tasks_to_tasks = self.act(tasks_to_tasks)

        x_tasks = self.task_merge_mlp(torch.cat([tasks_from_tasks, tasks_to_tasks], dim=-1))
        x_tasks = self.act(x_tasks)

        for l in range(self.num_layers):

            x_data_new = self.data_task_convs[l](
                (x_tasks, x_data),
                read_edges_masked.flip(0),
            )
            if self.use_norm:
                x_data_new = self.data_task_norms[l](x_data_new)
            x_data_new = self.act(x_data_new)
            #x_data = x_data + x_data_new

            x_tasks_new = self.task_data_convs[l](
                (x_data_new, x_tasks),
                read_edges_masked,
            )
            if self.use_norm:
                x_tasks_new = self.task_data_norms[l](x_tasks_new)
            x_tasks_new = self.act(x_tasks_new)

            #x_tasks = x_tasks + x_tasks_new

        tasks_global = global_mean_pool(x_tasks, b_tasks)
        data_global = global_mean_pool(x_data, b_data)

        global_state = self.global_merge_mlp(torch.cat([tasks_global, data_global], dim=-1))
        if self.use_norm:
            global_state = self.global_merge_norm(global_state)
        global_state = self.act(global_state)

        g = None 
        if self.g_dim > 0:
            if self.add_progress:
                time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
                time_feature = time_feature.reshape(-1, 1)
                progress_feature = tensordict["aux", "progress"]
                progress_feature = progress_feature.reshape(-1, 1)
                g = torch.cat([time_feature, progress_feature], dim=-1)

            if self.add_device_load:
                device_load = tensordict["aux", "device_load"]
                device_memory = tensordict["aux", "device_memory"]
                device_load = device_load.reshape(-1, 2 * self.n_devices)
                device_memory = device_memory.reshape(-1, 1 * self.n_devices)

                if g is None:
                    g = torch.cat([device_load, device_memory], dim=-1)
                else:
                    g = torch.cat([g, device_load, device_memory], dim=-1)

            g = self.g_mlp(g)
            if self.use_norm:
                g = self.g_norm(g)
            g = self.act(g)

        if b_tasks is not None:
            idx = data["tasks"].ptr[:-1]
            x = x_tasks[idx]
        else:
            x = x_tasks[0]

        if b_tasks is None:
            global_state = global_state.squeeze(0)
            if g is not None:
                g = g.squeeze(0)

        x = torch.cat([x, global_state], dim=-1)
        if g is not None:
            x = torch.cat([x, g], dim=-1)
        
        x = x.reshape(*batch_size, -1, x.shape[-1])
        return x.select(dim=-2, index=0)






        



class OriginalGNNStateNet(nn.Module):

    def _mask_edges(self, edge_index, edge_mask, edge_attr=None):
        mask = edge_mask.to(torch.bool)
        edge_index_masked = edge_index[:, mask]
        edge_attr_masked = edge_attr[mask] if edge_attr is not None else None
        return edge_index_masked, edge_attr_masked

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: int = 16,
        n_heads: int = 2,
        add_device_load: bool = False,
        add_progress: bool = False,
        n_devices: int = 5,
        **_ignored,
    ):
        super(OriginalGNNStateNet, self).__init__()

        self.feature_config = feature_config
        self.n_heads = n_heads
        self.hidden_channels = hidden_channels
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.n_devices = int(n_devices)

        self.g_dim = 0
        if add_progress:
            self.g_dim += 2
        if add_device_load:
            self.g_dim += 3 * n_devices

        if self.g_dim > 0:
            self.g_proj = nn.Linear(self.g_dim, hidden_channels)
            self.g_norm = nn.LayerNorm(hidden_channels)
            self.g_act = nn.LeakyReLU(negative_slope=0.01)

        if self.g_dim == 0:
            self.g_proj = None
            self.g_norm = None
            self.g_act = None

        self.stem_prog = nn.ModuleDict(
            {
                "tasks": Linear(int(feature_config.task_feature_dim), hidden_channels, bias=True),
                "data": Linear(int(feature_config.data_feature_dim), hidden_channels, bias=True),
            }
        )

        self.stem_norm = nn.ModuleDict(
            {
                "tasks": nn.LayerNorm(hidden_channels),
                "data": nn.LayerNorm(hidden_channels),
            }
        )

        self.convert_data = HeteroDataWrapper()

        self.gnn_tasks_data = GATv2Conv(
            (hidden_channels, hidden_channels),
            hidden_channels,
            heads=n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATv2Conv(
            (hidden_channels, hidden_channels),
            hidden_channels,
            heads=n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        )

        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = hidden_channels * 6 + (hidden_channels if self.g_dim > 0 else 0)
        self.output_keys = ["embed"]

    def forward(self, tensordict: TensorDict):
        batch_size = tensordict.batch_size
        data = self.convert_data(tensordict)

        b_tasks = data["tasks"].batch if isinstance(data, Batch) else None

        x_tasks = self.stem_prog["tasks"](data["tasks"].x)
        x_tasks = self.stem_norm["tasks"](x_tasks)
        x_tasks = self.act(x_tasks)

        x_data = self.stem_prog["data"](data["data"].x)
        x_data = self.stem_norm["data"](x_data)
        x_data = self.act(x_data)

        data_read_tasks = data["data", "read", "tasks"].edge_index
        mask = data["data", "read", "tasks"].edge_attr
        read_edges_masked, _ = self._mask_edges(edge_index=data_read_tasks, edge_mask=mask[:, 0])

        data_fused_tasks = self.gnn_tasks_data(
            (x_data, x_tasks),
            read_edges_masked,
        )

        tasks_fused_tasks = self.gnn_tasks_tasks(
            (x_tasks, x_tasks),
            data["tasks", "to", "tasks"].edge_index,
        )

        x_data_updated = self.layer_norm1(data_fused_tasks)
        x_data_updated = self.act(x_data_updated)

        x_tasks_updated = self.layer_norm2(tasks_fused_tasks)
        x_tasks_updated = self.act(x_tasks_updated)

        x_fused = torch.cat([x_tasks, x_tasks_updated, x_data_updated], dim=-1)

        global_fused = global_mean_pool(x_fused, b_tasks)

        g = None
        if self.g_dim > 0:

            if self.add_progress:
                time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
                time_feature = time_feature.reshape(-1, 1)
                progress_feature = tensordict["aux", "progress"]
                progress_feature = progress_feature.reshape(-1, 1)
                g = torch.cat([time_feature, progress_feature], dim=-1)

            if self.add_device_load:
                device_load = tensordict["aux", "device_load"]
                device_memory = tensordict["aux", "device_memory"]
                device_load = device_load.reshape(-1, 2 * self.n_devices)
                device_memory = device_memory.reshape(-1, 1 * self.n_devices)

                if g is None:
                    g = torch.cat([device_load, device_memory], dim=-1)
                else:
                    g = torch.cat([g, device_load, device_memory], dim=-1)

            g = self.g_proj(g)
            g = self.g_norm(g)
            g = self.g_act(g)

        if b_tasks is None:
            global_fused = global_fused.squeeze(0)
            g = g.squeeze(0) if self.g_dim > 0 else None

        if self.g_dim > 0:
            global_fused = torch.cat([global_fused, g], dim=-1)

        if b_tasks is not None:
            idx = data["tasks"].ptr[:-1]
            x = x_fused[idx]
        else:
            x = x_fused[0]

        x = torch.cat([x, global_fused], dim=-1)

        x = x.reshape(*batch_size, -1, x.shape[-1])
        return x.select(dim=-2, index=0)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size, padding=pad)
        self.act1 = nn.LeakyReLU(inplace=False, negative_slope=0.01)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=pad)
        self.act2 = nn.LeakyReLU(inplace=False, negative_slope=0.01)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return out + residual


class CNNSingleStateNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: int,
        add_progress: bool = False,
        activation: DictConfig = None,
        initialization: DictConfig = None,
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
            nn.LeakyReLU(inplace=False, negative_slope=0.01),
        ]
        ch = hidden_ch

        # build floor(n_layers/2) residual blocks
        for _ in range((n_layers - 2) // 2):
            blocks.append(ResidualBlock(ch, hidden_ch, kernel_size))
            ch = hidden_ch

        # if odd number of layers, tack on a final conv+ReLU
        if n_layers % 2 == 1:
            blocks += [
                nn.Conv2d(ch, hidden_ch, kernel_size, padding=pad),
                nn.LeakyReLU(inplace=False, negative_slope=0.01),
            ]
            ch = hidden_ch
        # final conv layer
        blocks.append(nn.Conv2d(ch, 1, kernel_size, padding=pad))
        blocks.append(nn.LeakyReLU(inplace=False, negative_slope=0.01))
        ch = 1

        self.net = nn.Sequential(*blocks)
        self.output_dim = ((self.width * self.length) * ch + 1) if self.add_progress else ((self.width * self.length) * ch)
        # Initialize CNN weights
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x is a TensorDict; x.batch_size might be [], [N], [N,M], etc.
        width = self.width
        length = self.length
        batch_size = x.batch_size

        # Pull out the tasks tensor: shape = (*batch_size, tasks, in_channels)
        x_tasks = x["nodes", "tasks", "attr"]
        # Split off the leading batch dims vs. the last two dims (tasks, channels)
        *batch_shape, tasks, in_channels = x_tasks.shape

        # Flatten all leading batch dims into one:
        flat_bs = 1
        for d in batch_shape:
            flat_bs *= d

        # Now we have a 3-D tensor (flat_bs, tasks, in_channels)
        x_flat = x_tasks.reshape(flat_bs, tasks, in_channels)

        # Convert the 'tasks' dim back into (width, length) spatial dims
        x_flat = x_flat.view(flat_bs, width, length, in_channels).permute(0, 3, 1, 2)  # (flat_bs, W, L, C_in)  # (flat_bs, C_in, W, L)

        # Run through your convolutional net
        x_flat = self.net(x_flat)

        # Collapse spatial/channel dims into a single feature vector
        x_flat = x_flat.contiguous().view(flat_bs, -1)

        # Finally, reshape back to the original batch dimensions:
        if batch_shape:
            # e.g. for batch_shape=[N,M], gives (N, M, features)
            x_out = x_flat.view(*batch_shape, -1)
        else:
            # single sample: drop the artificial batch axis → (features,)
            x_out = x_flat.squeeze(0)

        if self.add_progress:
            # Add time and progress features
            progress_feature = x["aux", "progress"]
            x_out = torch.cat([x_out, progress_feature], dim=-1)
        return x_out


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compute_num_downsampling_layers(length: int, width: int, minimum_resolution: int) -> int:
    h, w = int(length), int(width)
    layers = 0
    while min(h, w) >= 2 * minimum_resolution:
        h //= 2
        w //= 2
        layers += 1
    return layers


def _init_deconv_bilinear_(deconv: nn.ConvTranspose2d) -> None:
    if deconv.groups != 1:
        return
    k_h, k_w = deconv.kernel_size
    s_h, s_w = deconv.stride
    if not (k_h == k_w == s_h == s_w):
        return
    if deconv.in_channels != deconv.out_channels:
        return

    k = k_h
    factor = (k + 1) // 2
    center = factor - 1 if (k % 2 == 1) else factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    filt1d = 1 - torch.abs(og - center) / factor
    filt2d = torch.outer(filt1d, filt1d)

    with torch.no_grad():
        w = deconv.weight
        w.zero_()
        # Weight shape for ConvTranspose2d is (in_ch, out_ch, k, k) when groups==1
        for c in range(deconv.in_channels):
            w[c, c, :, :] = filt2d
        if deconv.bias is not None:
            deconv.bias.zero_()


def _align_and_concat(up_feat: torch.Tensor, enc_feat: torch.Tensor) -> torch.Tensor:
    uh, uw = up_feat.shape[-2:]
    eh, ew = enc_feat.shape[-2:]
    dh, dw = eh - uh, ew - uw
    if dh > 0 or dw > 0:
        # pad order: (left, right, top, bottom)
        pad = [max(dw // 2, 0), max(dw - dw // 2, 0), max(dh // 2, 0), max(dh - dh // 2, 0)]
        up_feat = F.pad(up_feat, pad)
    elif dh < 0 or dw < 0:
        top = (-dh) // 2
        left = (-dw) // 2
        up_feat = up_feat[..., top : top + eh, left : left + ew]
    return torch.cat([up_feat, enc_feat], dim=1)


def _flatten_to_BCHW(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Accept (C,H,W) or (*batch, C, H, W). Return (B, C, H, W),
    """
    if x.dim() == 3:
        return x.unsqueeze(0), (), 1
    elif x.dim() >= 4:
        *batch, C, H, W = x.shape
        B = 1
        for d in batch:
            B *= int(d)
        return x.reshape(B, C, H, W), tuple(batch), B
    else:
        raise ValueError(f"Expected (C,H,W) or (*batch,C,H,W), got {tuple(x.shape)}")


def _unflatten_from_B(xB: torch.Tensor, batch_shape: Tuple[int, ...]) -> torch.Tensor:
    """Inverse of _flatten_to_BCHW for the *batch* part; keeps (C,H,W) intact."""
    return xB.squeeze(0) if not batch_shape else xB.view(*batch_shape, *xB.shape[1:])


def _flatten_last_dim(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Accept (P) or (*batch, P). Return (B, P), batch_shape, B.
    """
    if x.dim() == 1:
        return x.unsqueeze(0), (), 1
    elif x.dim() >= 2:
        *batch, P = x.shape
        B = 1
        for d in batch:
            B *= int(d)
        return x.reshape(B, P), tuple(batch), B
    else:
        raise ValueError(f"Expected (..., P), got {tuple(x.shape)}")


class ConvNormAct(nn.Module):
    def __init__(self, C_in, C_out, k=3, dilation=1, groups=1, act="silu"):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=k, padding=pad, dilation=dilation, bias=False, groups=groups)
        self.norm = nn.GroupNorm(_choose_gn_groups(C_out), C_out)
        self.act = nn.SiLU(inplace=False) if act == "silu" else nn.ReLU(inplace=False)

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
    """
    Efficient Channel Attention: global avg pool -> 1D conv (k odd) -> sigmoid gate.
    """

    def __init__(self, C: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)  # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)  # (B,1,C)
        y = self.conv(y)  # (B,1,C)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return x * y


def _choose_gn_groups(C: int) -> int:
    for g in (8, 4, 2):
        if C % g == 0:
            return g
    return 1


class TinyASPP(nn.Module):
    """
    Concats parallel feats (different dilation) and reduces back to C with a 1x1.
    """

    def __init__(self, C: int, rates=(1, 2, 3), act="silu"):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(C, C, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.GroupNorm(_choose_gn_groups(C), C),
                    nn.SiLU(inplace=False) if act == "silu" else nn.ReLU(inplace=False),
                )
                for r in rates
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(len(rates) * C, C, kernel_size=1, bias=False),
            nn.GroupNorm(_choose_gn_groups(C), C),
            nn.SiLU(inplace=False) if act == "silu" else nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class SpatialModulator(nn.Module):
    """
    SPADE-style: z_spa -> (gamma_xy, beta_xy) in R^{B×C×H×W}.
    """

    def __init__(self, C: int, H: int, W: int, z_spa_dim: int, ch_hidden: int = 128, seed_hw: Optional[Tuple[int, int]] = None, init_scale_gamma_xy: float = 0.5, init_scale_beta_xy: float = 0.5):
        super().__init__()
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.h0 = max(4, H // 4) if not seed_hw else seed_hw[0]
        self.w0 = max(4, W // 4) if not seed_hw else seed_hw[1]

        self.to_seed = nn.Sequential(nn.Linear(z_spa_dim, ch_hidden), nn.SiLU(), nn.Linear(ch_hidden, 2 * C * self.h0 * self.w0))
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

    def forward(self, z) -> Tuple[torch.Tensor, torch.Tensor]:  # z: (B, z_spa_dim)

        *lead, z_dim = z.shape
        B = int(torch.prod(torch.tensor(lead))) if lead else z.shape[0]
        zf = z.reshape(-1, z_dim)
        seed = self.to_seed(zf).view(B, 2 * self.C, self.h0, self.w0)
        maps = F.interpolate(seed, size=(self.H, self.W), mode="bilinear", align_corners=False)
        g_raw, b_raw = maps.chunk(2, dim=1)  # (B,C,H,W)
        g_xy = self.scale_gamma_xy * torch.tanh(g_raw)  # bounded, ≈0
        b_xy = self.scale_beta_xy * b_raw  # small bias
        return g_xy, b_xy


class AdaSPADE_GN(nn.Module):
    """
    GroupNorm (affine=False) + channel FiLM(z_ch) + (optional) spatial FiLM(z_spa).
    Dimensions explicit via z_ch_dim.
    """

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
        self.to_gb_c = nn.Sequential(nn.Linear(z_ch_dim, ch_hidden), nn.SiLU(), nn.Linear(ch_hidden, 2 * C))
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
        # z_ch: (B, z_ch_dim); z_spa: (B, z_spa_dim)
        B, Cx, H, W = x.shape
        z_ch = z_ch.reshape(B, -1)
        z_spa = z_spa.reshape(B, -1)

        x = self.gn(x)

        if not self.enable_channel:
            return x

        g_c_raw, b_c_raw = self.to_gb_c(z_ch).chunk(2, dim=-1)  # (B,C)
        g_c = 1.0 + self.scale_gamma_c * torch.tanh(g_c_raw)
        b_c = self.scale_beta_c * b_c_raw

        if self.enable_spatial:
            g_xy, b_xy = self.spatial(z_spa)  # (B,C,H,W)
        else:
            B, C, H, W = x.shape
            g_xy = x.new_zeros((B, Cx, H, W))
            b_xy = x.new_zeros((B, Cx, H, W))

        gamma = g_c.unsqueeze(-1).unsqueeze(-1) * (1.0 + g_xy)
        beta = b_c.unsqueeze(-1).unsqueeze(-1) + b_xy
        return x * gamma + beta


class DilatedResBlock_SPADE(nn.Module):
    def __init__(self, C: int, dilation: int, norm1: AdaSPADE_GN, norm2: AdaSPADE_GN):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = norm1
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.norm2 = norm2
        self.act2 = nn.SiLU()
        nn.init.zeros_(self.conv2.weight)  # identity-at-init

    def forward(self, x, z_ch, z_spa):
        h = self.act1(self.norm1(self.conv1(x), z_ch, z_spa))
        h = self.act2(self.norm2(self.conv2(h), z_ch, z_spa))
        return x + h


class DilationState(nn.Module):
    """
    Fixed-resolution CNN backbone, to be used as `layers.state`.
    """

    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        width: int,
        length: int,
        z_ch_dim: int = 8,
        z_spa_dim: int = 8,
        num_blocks: int = 3,
        dilation_schedule: Optional[List[int]] = None,
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
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.add_z = bool(add_z)

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
        _z_ch = observation[("aux", "z_ch")]
        _z_spa = observation[("aux", "z_spa")]
        _device_load = observation["aux", "device_load"]
        _device_memory = observation["aux", "device_memory"]
        _progress = observation["aux", "progress"]
        _baseline = observation["aux", "baseline"]
        _time = observation["aux", "time"]
        _perc = _time / _baseline

        single = xt.dim() == 2
        if single:
            xt = xt.unsqueeze(0)
        *batch_shape, T, Cin = xt.shape
        H, W = self.length, self.width
        assert T == H * W, f"tasks={T} differs from H*W={H*W}"
        assert Cin == self.in_channels

        B = 1
        for d in batch_shape:
            B *= int(d)
        h = xt.reshape(B, H, W, Cin).permute(0, 3, 1, 2)  # (B,Cin,H,W)

        if self.add_z:
            z_ch = _z_ch
            z_spa = _z_spa
        else:
            if B == 1:
                z_ch = torch.zeros(1, device=xt.device, dtype=xt.dtype)
                z_spa = torch.zeros(1, device=xt.device, dtype=xt.dtype)
            else:
                z_ch = torch.zeros(B, 1, device=xt.device, dtype=xt.dtype)
                z_spa = torch.zeros(B, 1, device=xt.device, dtype=xt.dtype)

        if self.add_device_load:
            if B > 1:
                _device_memory = _device_memory.reshape(B, -1)
                _device_load = _device_load.reshape(B, -1)
            z_ch = torch.cat([z_ch, _device_load, _device_memory], dim=-1)
            z_spa = torch.cat([z_spa, _device_load, _device_memory], dim=-1)

        if self.add_progress:
            if B > 1:
                _progress = _progress.reshape(B, -1)
                _perc = _perc.reshape(B, -1)
            z_ch = torch.cat([z_ch, _progress, _perc], dim=-1)
            z_spa = torch.cat([z_spa, _progress, _perc], dim=-1)

        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h, z_ch, z_spa)
        h = self.eca(h)

        if single:
            h = h.squeeze(0)  # (C,H,W)
        else:
            C = h.size(1)
            h = h.view(*batch_shape, C, H, W)

        return (h,)


class DilationPolicyHead(nn.Module):
    """
    Minimal actor head: ('embed')=(..., C, H, W) -> logits (…, H*W, A)
    """

    def __init__(self, input_dim: int, output_dim: int, width: int, length: int, init_mode: str = "tiny", tiny_std: float = 1e-3, debug: bool = False, **_ignored):  # 'zero' | 'tiny' | 'kaiming'
        super().__init__()
        self.width = int(width)
        self.length = int(length)
        self.Cin = int(input_dim)
        self.A = int(output_dim)
        self.debug = bool(debug)

        self.input_keys = ["embed"]
        self.output_dim = self.A

        self.proj = nn.Conv2d(self.Cin, self.A, kernel_size=1, bias=True)

        init_mode = init_mode.lower()
        if init_mode == "zero":
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif init_mode == "tiny":
            nn.init.normal_(self.proj.weight, std=float(tiny_std))
            nn.init.zeros_(self.proj.bias)
        elif init_mode == "kaiming":
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
            nn.init.zeros_(self.proj.bias)
        else:
            raise ValueError(f"init_mode must be 'zero' | 'tiny' | 'kaiming', got {init_mode!r}")

    def forward(self, obs, embed):
        if embed.dim() == 3:
            h = embed.unsqueeze(0)
            single = True
        else:
            h = embed
            single = False

        *B, C, H, W = h.shape
        h = embed.view(-1, C, H, W)
        logits_hw = self.proj(h)  # (B, A, H, W)
        logits = logits_hw.permute(0, 2, 3, 1).reshape(h.size(0), H * W, self.A)
        logits = logits.view(*B, H * W, self.A)
        return logits[0] if single else logits


class DilationValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        z_dim: int = 8,
        proj_dim: int = 8,
        hidden_channels: int = 128,
        tiny_std: float = 1e-3,
        add_gap: bool = True,  # global avg pool
        add_z: bool = False,
        add_progress: bool = True,
        add_device_load: bool = False,
        n_devices: int = 5,
        **_ignored,
    ):

        super().__init__()
        C = int(input_dim)
        P = int(proj_dim)
        Dz = int(z_dim) * 2

        self.mix = nn.Conv2d(C, P, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.mix.weight, nonlinearity="relu")
        self.add_z = bool(add_z)
        self.add_device_load = bool(add_device_load)
        self.add_progress = bool(add_progress)

        # attention scorer -> (B,1,H,W);
        self.attn = nn.Conv2d(P, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.attn.weight, std=tiny_std)
        nn.init.zeros_(self.attn.bias)

        self.add_gap = bool(add_gap)
        mlp_in = (2 * P if self.add_gap else P) + (Dz if add_z else 0) + (3 * n_devices if add_device_load else 0) + (2 if add_progress else 0)
        self.mlp = nn.Sequential(nn.Linear(mlp_in, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, 1))
        nn.init.normal_(self.mlp[-1].weight, std=tiny_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs, embed):
        z_ch = obs[("aux", "z_ch")]
        z_spa = obs[("aux", "z_spa")]

        z_f = torch.cat([z_ch, z_spa], dim=-1)
        z_f = z_f.reshape(-1, z_f.size(-1))  # (B, Dz)

        *batch, C, H, W = embed.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        embed_f = embed.reshape(-1, C, H, W)  # (B, C, H, W)

        Fm = F.silu(self.mix(embed_f))  # (B, P, H, W)
        scores = self.attn(Fm)  # (B, 1, H, W)
        attn = scores.flatten(2).softmax(dim=-1).view(B, 1, H, W)
        pooled_attn = (Fm * attn).sum(dim=(2, 3))  # (B, P)

        if self.add_gap:
            pooled_gap = Fm.mean(dim=(2, 3))  # (B, P)
            pooled = torch.cat([pooled_attn, pooled_gap], dim=1)  # (B, 2P)
        else:
            pooled = pooled_attn

        if self.add_z:
            pooled = torch.cat([pooled, z_f], dim=-1)  # (B, 2P + Dz)
        else:
            pooled = pooled

        if self.add_device_load:
            device_load = obs["aux", "device_load"]
            device_memory = obs["aux", "device_memory"]
            device_feat = torch.cat([device_load, device_memory], dim=-1)
            device_feat = device_feat.reshape(-1, device_feat.size(-1))  # (B, 3*n_devices)
            pooled = torch.cat([pooled, device_feat], dim=-1)

        if self.add_progress:
            progress = obs["aux", "progress"].reshape(B, -1)  # (B, 1)
            baseline = obs["aux", "baseline"].reshape(B, -1)  # (B, 1)
            time = obs["aux", "time"].reshape(B, -1)  # (B, 1)
            perc = time / baseline
            prog_feat = torch.cat([progress, perc], dim=-1)  # (B, 2)
            pooled = torch.cat([pooled, prog_feat], dim=-1)

        v = self.mlp(pooled).squeeze(-1)  # (B,)
        v = v.view(*batch, 1)  # (*batch,)
        return v


class UnconditionedDilationState(nn.Module):

    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        width: int,
        length: int,
        add_progress: bool = False,
        add_device_load: bool = True,
        n_devices: int = 5,
        debug: bool = True,
        num_blocks: int = 2,
        dilation_schedule: Optional[List[int]] = None,
        use_eca: bool = True,
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

        self.num_layers = 0
        C_in = self.in_channels
        C = self.hidden_channels
        self.stem = ConvNormAct(C_in, C, k=3, dilation=1, act="silu")

        # Dilated residual stack
        if not dilation_schedule:
            dilation_schedule = [1, 2, 3]

        self.blocks = nn.ModuleList([DilatedResBlock(C, dilation=dilation_schedule[i % len(dilation_schedule)], act="silu") for i in range(num_blocks)])

        self.eca = ECA(C, k_size=3) if use_eca else nn.Identity()

        film_dim = 2 if add_progress else 0
        film_dim += 3 * n_devices if add_device_load else 0
        self.add_progress = bool(add_progress)
        self.add_device_load = bool(add_device_load)
        self.n_devices = int(n_devices)

        self.film_dim = film_dim
        if film_dim > 0:
            self.film = nn.Linear(film_dim, 2 * self.hidden_channels, bias=True)
        else:
            self.film = None

        self.in_channels_per_scale: List[int] = [C]
        self.output_dim = C
        self.output_keys: List[str] = ["embed"]

    def forward(self, x):
        xt = x["nodes", "tasks", "attr"]

        single = xt.dim() == 2
        if single:
            xt = xt.unsqueeze(0)  # (1, tasks, C)

        *batch_shape, T, Cin = xt.shape
        H, W = self.length, self.width
        assert T == H * W, f"tasks={T} differs from length*width={H*W}"
        assert Cin == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {Cin}"

        # Flatten and reshape to BCHW
        B = 1
        for d in batch_shape:
            B *= int(d)
        h = xt.reshape(B, H, W, Cin).permute(0, 3, 1, 2)  # (B,Cin,H,W)

        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)

        if self.film is not None:
            c = None

            if self.add_device_load:
                device_load = x["aux", "device_load"]
                device_memory = x["aux", "device_memory"]
                device_feat = torch.cat([device_load, device_memory], dim=-1)
                if c is None:
                    c = device_feat
                else:
                    c = torch.cat([c, device_feat], dim=-1)

            if self.add_progress:
                progress = x["aux", "progress"]
                baseline = x["aux", "baseline"]
                time = x["aux", "time"]
                perc = time / baseline
                prog_feat = torch.cat([progress, perc], dim=-1)
                if c is None:
                    c = prog_feat
                else:
                    c = torch.cat([c, prog_feat], dim=-1)

            cB, _, _ = _flatten_last_dim(c)
            gamma_beta = self.film(cB)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)

        h = self.eca(h)

        if single:
            h = h.squeeze(0)  # (C,H,W)
            if self.debug:
                print(f"[Encoder] embed {h.shape}")
            return (h,)
        else:
            h = h.view(*batch_shape, *h.shape[1:])  # (*batch, C, H, W)
            if self.debug:
                print(f"[Encoder] embed {h.shape}")

        return (h,)


class UnconditionedDilationPolicyHead(nn.Module):

    def __init__(
        self, input_dim: int, hidden_channels: int, width: int, length: int, output_dim: int, debug: bool = True, num_blocks: int = 2, dilation_schedule: Optional[List[int]] = None, **_ignored
    ):
        super().__init__()
        self.width = int(width)
        self.length = int(length)
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.debug = bool(debug)
        self.num_layers = 0

        if not dilation_schedule:
            dilation_schedule = [1, 2]
        self.pre = ConvNormAct(self.input_dim, self.hidden_channels, k=3, dilation=1, act="silu")
        self.blocks = nn.ModuleList([DilatedResBlock(self.hidden_channels, dilation=dilation_schedule[i % len(dilation_schedule)], act="silu") for i in range(num_blocks)])
        self.out_conv = nn.Conv2d(self.hidden_channels, self.output_dim, kernel_size=1)

        self.in_channels_per_scale: List[int] = [self.input_dim]
        self.input_keys: List[str] = ["embed"]

    def forward(self, obs, *features):
        if len(features) == 0:
            raise ValueError("Decoder expects encoder features: (*enc_feats, bottleneck_map)")
        b_map = features[-1]  # (C,H,W) or (*batch,C,H,W)

        # Normalize to BCHW
        hB, batch_shape, B = _flatten_to_BCHW(b_map)  # (B,C,H,W)
        _, C, H, W = hB.shape
        assert C == self.input_dim, f"Decoder input_dim={self.input_dim}, got bottleneck C={C}"

        hB = self.pre(hB)
        for blk in self.blocks:
            hB = blk(hB)
        logits_map = self.out_conv(hB)  # (B, A, H, W)

        if len(batch_shape) == 0:
            logits = logits_map.permute(0, 2, 3, 1).reshape(H * W, self.output_dim).squeeze(0)
        else:
            logits = logits_map.permute(0, 2, 3, 1).reshape(B, H * W, self.output_dim).view(*batch_shape, H * W, self.output_dim)
        return logits


class UNetState(nn.Module):

    def __init__(self, feature_config, hidden_channels: int, width: int, length: int, add_progress: bool = False, minimum_resolution: int = 2, debug: bool = True, pool_mode: str = "avg", **_ignored):
        super().__init__()
        if not hasattr(feature_config, "task_feature_dim"):
            raise AttributeError("feature_config must have attribute 'task_feature_dim'")
        self.width = int(width)
        self.length = int(length)
        self.in_channels = int(feature_config.task_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.minimum_resolution = int(minimum_resolution)
        self.debug = debug
        self.add_progress = bool(add_progress)
        self.progress_dim = 1 if self.add_progress else 0

        self.num_layers = _compute_num_downsampling_layers(self.length, self.width, self.minimum_resolution)

        self.enc_blocks = nn.ModuleList()
        if self.num_layers == 0:
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=False),
            )
            channels = self.hidden_channels
            self.in_channels_per_scale = [channels]
        else:
            in_ch = self.in_channels
            skip_channels = []
            for i in range(self.num_layers):
                out_ch = self.hidden_channels * (2**i)
                self.enc_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.01, inplace=False),
                    )
                )
                in_ch = out_ch
                skip_channels.append(out_ch)

            channels = in_ch  # = hidden * 2**(num_layers-1)
            self.in_channels_per_scale = [*skip_channels, channels]

        if pool_mode == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, count_include_pad=False)
        elif pool_mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("pool_mode must be 'max' or 'avg'")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

        if self.add_progress:
            self.film = nn.Linear(self.progress_dim, 2 * channels, bias=True)

        self.output_dim = channels
        self.output_keys: List[str] = [f"enc_{i}" for i in range(self.num_layers)] + ["embed"]

    def forward(self, x):
        xt = x["nodes", "tasks", "attr"]  # shape: (*batch, tasks, C) or (tasks, C)

        single = xt.dim() == 2  # (tasks, C)
        if single:
            xt = xt.unsqueeze(0)  # -> (1, tasks, C)

        *batch_shape, tasks, in_ch = xt.shape
        assert in_ch == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {in_ch}"
        assert tasks == self.length * self.width, f"got tasks={tasks}, expected length*width={self.length*self.width}"

        B = 1
        for d in batch_shape:
            B *= int(d)
        h = xt.reshape(B, self.length, self.width, self.in_channels).permute(0, 3, 1, 2)

        enc_feats: List[torch.Tensor] = []
        if self.num_layers == 0:
            h = self.stem(h)
        else:
            for block in self.enc_blocks:
                if self.debug:
                    print(f"[Encoder] pre-mix {h.shape}")
                h = block(h)
                if self.debug:
                    print(f"[Encoder] post-mix {h.shape}")
                enc_feats.append(h)  # pre-pool skip
                h = self.pool(h)
                if self.debug:
                    print(f"[Encoder] post-pool {h.shape}")

        b_map = self.bottleneck(h)

        if self.add_progress:
            z = x["aux", "progress"]
            zB, _, _ = _flatten_last_dim(z)
            gamma_beta = self.film(zB)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            b_map = gamma.unsqueeze(-1).unsqueeze(-1) * b_map + beta.unsqueeze(-1).unsqueeze(-1)

        if self.debug:
            print(f"[Encoder] bottleneck {h.shape}")

        def unflatten(t):
            return t.squeeze(0) if single else t.view(*batch_shape, *t.shape[1:])

        enc_feats = [unflatten(f) for f in enc_feats]
        b_map = unflatten(b_map)

        output = (*enc_feats, b_map)
        return output


class UNetPolicyHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        width: int,
        length: int,
        output_dim: int,
        minimum_resolution: int = 2,
        debug: bool = True,
        upsample_type: str = "nearest",
        deconv_bilinear_init: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.width = int(width)
        self.length = int(length)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)
        self.minimum_resolution = int(minimum_resolution)
        self.upsample_type = str(upsample_type)
        self.deconv_bilinear_init = bool(deconv_bilinear_init)
        self.debug = debug

        self.num_layers = _compute_num_downsampling_layers(self.length, self.width, self.minimum_resolution)

        if self.num_layers == 0:
            self.in_channels_per_scale = [self.input_dim]  # just bottleneck
        else:
            skip_channels = [self.hidden_channels * (2**i) for i in range(self.num_layers)]
            self.in_channels_per_scale = [*skip_channels, self.input_dim]

        expected_bottleneck_ch = self.hidden_channels * (2 ** max(self.num_layers - 1, 0))
        assert self.input_dim == expected_bottleneck_ch, f"Decoder input_dim={self.input_dim} must equal encoder bottleneck channels {expected_bottleneck_ch}"

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        prev_ch = self.input_dim
        for i in reversed(range(self.num_layers)):
            out_ch = self.hidden_channels * (2**i)

            if self.upsample_type == "deconv":
                up = nn.ConvTranspose2d(prev_ch, out_ch, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
                if self.deconv_bilinear_init:
                    _init_deconv_bilinear_(up)
            elif self.upsample_type == "nearest":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1, bias=True),
                )
            else:
                raise ValueError("upsample_type must be 'deconv' or 'nearest'")

            self.up_blocks.append(up)
            # After concat with skip (C=out_ch): fuse back to out_ch
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=False),
                )
            )
            prev_ch = out_ch

        self.input_keys: List[str] = [f"enc_{i}" for i in range(self.num_layers)] + ["embed"]

        # Final projection to logits at full resolution
        final_in = self.hidden_channels if self.num_layers >= 1 else self.input_dim
        self.out_conv = nn.Conv2d(final_in, 2 * self.output_dim, kernel_size=1)

        self.logit_layer = LogitsOutputHead(
            input_dim=2 * self.output_dim,
            hidden_channels=self.hidden_channels,
            output_dim=self.output_dim,
        )

    def forward(self, obs, *features):
        if len(features) == 0:
            raise ValueError("Decoder expects encoder features: (*enc_feats, bottleneck_map)")
        enc_feats = features[:-1]
        b_map = features[-1]  # shape: (C,H,W) or (*batch,C,H,W)

        single = b_map.dim() == 3

        # Normalize shapes
        b_mapB, batch_shape, B = _flatten_to_BCHW(b_map)
        encB = [_flatten_to_BCHW(e)[0] for e in enc_feats]

        # Decode
        h = b_mapB
        if self.num_layers > 0:
            for up, dec, enc in zip(self.up_blocks, self.dec_blocks, reversed(encB)):
                h = up(h)
                if self.debug:
                    print(f"[Decoder] up: {h.shape} + {enc.shape}")
                h = _align_and_concat(h, enc)
                if self.debug:
                    print(f"[Decoder] concat: {h.shape} + {enc.shape}")
                h = dec(h)
                if self.debug:
                    print(f"[Decoder] dec: {h.shape}")

        logits_map = self.out_conv(h)  # (B, output_dim, H, W)
        if self.debug:
            print(f"[Decoder] logits: {logits_map.shape}")

        if single:
            logits = logits_map.permute(0, 2, 3, 1).reshape(-1, 2 * self.output_dim).squeeze(0)
            logits = self.logit_layer(logits)  # (H*W, output_dim)
        else:
            _, _, H, W = logits_map.shape
            logits = logits_map.permute(0, 2, 3, 1).reshape(B, H * W, 2 * self.output_dim).view(*batch_shape, H * W, 2 * self.output_dim)
            logits = self.logit_layer(logits)  # (*batch, H*W, output_dim)
        return logits


class PooledOutputHead(nn.Module):
    def __init__(
        self,
        input_dim: int,  # shared dim before final MLP
        hidden_channels: int,  # hidden size in OutputHead
        output_dim: int,  # final dimension (e.g., 1 for V(s))
        activation: Optional[nn.Module] = None,
        initialization: Optional[dict] = None,
        layer_norm: bool = True,
        in_channels_per_scale: Optional[Sequence[int]] = None,
        add_device_load: bool = False,
        add_progress: bool = True,
        n_devices: int = 5,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__()
        self.proj_dim = int(input_dim)
        self.output_hidden = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.debug = debug

        self._built: bool = False
        self._in_dims: Optional[List[int]] = None
        self._proj = nn.ModuleList()
        self._head: Optional[OutputHead] = None
        self.add_device_load = bool(add_device_load)
        self.n_devices = int(n_devices)
        self.add_progress = bool(add_progress)

        if in_channels_per_scale is not None:
            self._build(list(int(c) for c in in_channels_per_scale))

        self.in_channels_per_scale: Optional[List[int]] = list(in_channels_per_scale) if in_channels_per_scale is not None else None

        oh_input_dim = self.proj_dim
        if self.add_device_load:
            oh_input_dim += 3 * self.n_devices
        if self.add_progress:
            oh_input_dim += 2

        self._oh_kwargs = dict(
            input_dim=oh_input_dim,
            hidden_channels=self.output_hidden,
            output_dim=self.output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
        )

    def _build(self, in_dims: List[int]) -> None:
        if len(in_dims) == 0:
            raise ValueError("PooledOutputHead: at least one scale is required.")
        self._in_dims = in_dims
        self.in_channels_per_scale = list(in_dims)

        # Per-scale LN + Linear(C_i -> D)
        self._proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(Ci),
                    nn.Linear(Ci, self.proj_dim, bias=False),
                )
                for Ci in in_dims
            ]
        )

        for m in self._proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self._head = OutputHead(**self._oh_kwargs)

        self._built = True

    def forward(self, obs, *encoder_outputs: torch.Tensor) -> torch.Tensor:
        if len(encoder_outputs) == 0:
            raise ValueError("PooledOutputHead expects at least one encoder feature.")

        featsB: List[torch.Tensor] = []
        batch_shape_ref: Optional[Tuple[int, ...]] = None
        seen_dims: List[int] = []

        for f in encoder_outputs:
            fB, batch_shape, _ = _flatten_to_BCHW(f)
            seen_dims.append(int(fB.shape[1]))
            if self.debug:
                print(f"[PooledOutputHead] features {fB.shape} {batch_shape}")
            if batch_shape_ref is None:
                batch_shape_ref = batch_shape
            elif batch_shape != batch_shape_ref:
                raise ValueError(f"Mismatched batch shapes among inputs: {batch_shape} vs {batch_shape_ref}")
            featsB.append(fB)

        if not self._built:
            self._build(seen_dims)
            self.to(featsB[0].device)
        else:
            assert self._in_dims is not None and self._head is not None
            if len(seen_dims) != len(self._in_dims):
                raise ValueError(f"Expected {len(self._in_dims)} feature maps, got {len(seen_dims)}.")
            for k, (got, exp) in enumerate(zip(seen_dims, self._in_dims)):
                if got != exp:
                    raise ValueError(f"Channel mismatch at scale {k}: got C={got}, expected C={exp}.")

        # Per-scale: GAP -> (B,C_i) -> LN+Linear -> (B,D)
        zs: List[torch.Tensor] = []
        for fB, proj in zip(featsB, self._proj):
            z = F.adaptive_avg_pool2d(fB, 1).flatten(1)  # (B, C_i)
            z = proj(z)  # (B, D)
            zs.append(z)
            if self.debug:
                print(f"[PooledOutputHead] features {fB.shape}")

        v = torch.stack(zs, dim=1).sum(dim=1)  # (B, D)

        if self.add_device_load:
            obs_device_load = obs["aux", "device_load"]
            obs_device_memory = obs["aux", "device_memory"]
            device_feat = torch.cat([obs_device_load, obs_device_memory], dim=-1)
            deviceB, _, _ = _flatten_last_dim(device_feat)
            v = torch.cat([v, deviceB], dim=-1)  # (B, D + 3*n_devices)

        if self.add_progress:
            progress = obs["aux", "progress"]
            baseline = obs["aux", "baseline"]
            time = obs["aux", "time"]
            perc = time / baseline
            prog_feat = torch.cat([progress, perc], dim=-1)
            progB, _, _ = _flatten_last_dim(prog_feat)
            v = torch.cat([v, progB], dim=-1)  # (B, D + 2)

        yB = self._head(v)  # (B, output_dim)
        output = _unflatten_from_B(yB, batch_shape_ref or ())

        return output


class UNetValueHead(nn.Module):
    """
    Wrapper for PooledOutputHead to match UNet interface
    """

    def __init__(
        self,
        input_dim: int,  # shared dim before final MLP
        hidden_channels: int,  # hidden size in OutputHead
        output_dim: int,  # final dimension (e.g., 1 for V(s))
        activation: Optional[nn.Module] = None,
        initialization: Optional[dict] = None,
        layer_norm: bool = True,
        in_channels_per_scale: Optional[Sequence[int]] = None,
        debug: bool = False,
        add_device_load: bool = True,
        add_progress: bool = False,
        n_devices: int = 5,
        **_ignored,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.debug = bool(debug)

        self.in_channels_per_scale: Optional[List[int]] = list(int(c) for c in in_channels_per_scale) if in_channels_per_scale is not None else None
        self.head = PooledOutputHead(
            input_dim=self.input_dim,
            hidden_channels=self.hidden_channels,
            output_dim=self.output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
            in_channels_per_scale=self.in_channels_per_scale,
            debug=self.debug,
            add_device_load=add_device_load,
            n_devices=n_devices,
            add_progress=add_progress,
        )
        self.output_dim = output_dim

    def forward(self, obs, *features):
        if len(features) == 0:
            raise ValueError("ValueHead expects at least one encoder feature.")
        return self.head(obs, *features)


class UnconditionedDilationValueHead(nn.Module):
    """
    Wrapper for PooledOutputHead to match DilationNet interface
    """

    def __init__(
        self,
        input_dim: int,  # shared dim before final MLP
        hidden_channels: int,  # hidden size in OutputHead
        output_dim: int,  # final dimension (e.g., 1 for V(s))
        activation: Optional[nn.Module] = None,
        initialization: Optional[dict] = None,
        layer_norm: bool = True,
        in_channels_per_scale: Optional[Sequence[int]] = None,
        debug: bool = False,
        add_device_load: bool = True,
        add_progress: bool = True,
        n_devices: int = 5,
        **_ignored,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.debug = bool(debug)

        self.in_channels_per_scale: Optional[List[int]] = list(int(c) for c in in_channels_per_scale) if in_channels_per_scale is not None else None

        self.head = PooledOutputHead(
            input_dim=self.input_dim,
            hidden_channels=self.hidden_channels,
            output_dim=self.output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
            in_channels_per_scale=self.in_channels_per_scale,
            debug=self.debug,
            add_device_load=add_device_load,
            n_devices=n_devices,
            add_progress=add_progress,
        )
        self.output_dim = output_dim

    def forward(self, obs, *features):
        if len(features) == 0:
            raise ValueError("ValueHead expects at least one encoder feature.")
        return self.head(obs, *features)

#-------GNN prototypes-------
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Any
from collections import deque
from pathlib import Path
import json
import os

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import coalesce, remove_self_loops, to_dense_adj


# ============================================================
# Exact k-hop edges for a PyG Batch (variable topology + N)
# ============================================================

@torch.no_grad()
def exact_k_hop_edge_index_batched_dense(
    edge_index: torch.Tensor,   # (2, E) over the batched disjoint union
    batch: torch.Tensor,        # (N,) graph id per node
    k: int,
    *,
    undirected: bool = True,
    remove_self: bool = True,
    max_num_nodes: Optional[int] = None,  # if None, PyG pads to max nodes in batch
) -> torch.Tensor:
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert batch.dim() == 1
    assert k >= 1

    device = edge_index.device
    N = int(batch.numel())
    if N == 0:
        return edge_index.new_zeros((2, 0))

    B = int(batch.max().item()) + 1

    ei = edge_index
    if undirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)

    ei = coalesce(ei, num_nodes=N)

    A = to_dense_adj(ei, batch=batch, max_num_nodes=max_num_nodes).to(torch.bool)
    B2, Nmax, Nmax2 = A.shape
    assert B2 == B and Nmax == Nmax2

    counts = torch.bincount(batch, minlength=B)  # (B,)
    idx = torch.arange(Nmax, device=device)
    valid = idx.unsqueeze(0) < counts.unsqueeze(1)         
    valid_ij = valid.unsqueeze(2) & valid.unsqueeze(1)      

    A = A & valid_ij

    if remove_self:
        diag = torch.arange(Nmax, device=device)
        A[:, diag, diag] = False

    if k == 1:
        exact = A
    else:
        A_f = A.to(torch.float32)
        reach = A.clone()           
        reach_lt = reach.clone()    

        for i in range(2, k + 1):
            reach = (reach.to(torch.float32).bmm(A_f) > 0)
            reach = reach & valid_ij
            if remove_self:
                diag = torch.arange(Nmax, device=device)
                reach[:, diag, diag] = False
            if i < k:
                reach_lt |= reach

        exact = reach & (~reach_lt)

    offsets = torch.zeros(B, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)[:-1]

    rows_all = []
    cols_all = []
    for b in range(B):
        nb = int(counts[b].item())
        if nb <= 0:
            continue
        row, col = exact[b, :nb, :nb].nonzero(as_tuple=True)
        if row.numel() == 0:
            continue
        off = offsets[b]
        rows_all.append(row + off)
        cols_all.append(col + off)

    if len(rows_all) == 0:
        out = edge_index.new_zeros((2, 0))
    else:
        out = torch.stack([torch.cat(rows_all), torch.cat(cols_all)], dim=0)

    if remove_self:
        out, _ = remove_self_loops(out)

    out = coalesce(out, num_nodes=N)
    return out


def _edge_index_to_pair_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    if edge_index.numel() == 0:
        return set()
    return {
        (int(src), int(dst))
        for src, dst in edge_index.t().tolist()
    }


@torch.no_grad()
def exact_k_hop_edge_index_batched_bfs(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    k: int,
    *,
    undirected: bool = True,
    remove_self: bool = True,
) -> torch.Tensor:
    """
    Independent exact-k-hop builder using shortest-path BFS per source node.
    Intended for debug/validation against the dense boolean-matmul implementation.
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    assert batch.dim() == 1
    assert k >= 1

    N = int(batch.numel())
    if N == 0:
        return edge_index.new_zeros((2, 0))

    ei = edge_index
    if undirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    ei = coalesce(ei, num_nodes=N)
    if remove_self:
        ei, _ = remove_self_loops(ei)

    batch_ids = [int(v) for v in batch.tolist()]
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for src, dst in ei.t().tolist():
        src_i = int(src)
        dst_i = int(dst)
        if batch_ids[src_i] != batch_ids[dst_i]:
            continue
        neighbors[src_i].add(dst_i)

    B = int(batch.max().item()) + 1
    nodes_per_graph: list[list[int]] = [[] for _ in range(B)]
    for node_id, graph_id in enumerate(batch_ids):
        nodes_per_graph[graph_id].append(node_id)

    rows: list[int] = []
    cols: list[int] = []
    for graph_nodes in nodes_per_graph:
        for src in graph_nodes:
            dist: dict[int, int] = {src: 0}
            queue: deque[int] = deque([src])
            while queue:
                current = queue.popleft()
                current_dist = dist[current]
                if current_dist >= k:
                    continue
                for nxt in neighbors[current]:
                    next_dist = current_dist + 1
                    prev_dist = dist.get(nxt)
                    if prev_dist is None or next_dist < prev_dist:
                        dist[nxt] = next_dist
                        if next_dist < k:
                            queue.append(nxt)
            for dst, hop in dist.items():
                if hop != k:
                    continue
                if remove_self and src == dst:
                    continue
                rows.append(int(src))
                cols.append(int(dst))

    if len(rows) == 0:
        out = edge_index.new_zeros((2, 0))
    else:
        out = edge_index.new_tensor([rows, cols], dtype=edge_index.dtype)
    if remove_self:
        out, _ = remove_self_loops(out)
    out = coalesce(out, num_nodes=N)
    return out


@torch.no_grad()
def validate_exact_k_hop_edge_index_batched(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    k_values: Sequence[int],
    *,
    undirected: bool = True,
    remove_self: bool = True,
    max_num_nodes: Optional[int] = None,
) -> dict[int, dict[str, Any]]:
    """
    Validate exact-k-hop edges by comparing the dense implementation against an
    independent BFS shortest-path implementation.
    """
    report: dict[int, dict[str, Any]] = {}
    unique_k = sorted({int(v) for v in k_values})
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    for k in unique_k:
        dense_edges = exact_k_hop_edge_index_batched_dense(
            edge_index=edge_index,
            batch=batch,
            k=k,
            undirected=undirected,
            remove_self=remove_self,
            max_num_nodes=max_num_nodes,
        )
        bfs_edges = exact_k_hop_edge_index_batched_bfs(
            edge_index=edge_index,
            batch=batch,
            k=k,
            undirected=undirected,
            remove_self=remove_self,
        )

        dense_set = _edge_index_to_pair_set(dense_edges)
        bfs_set = _edge_index_to_pair_set(bfs_edges)
        missing = sorted(bfs_set - dense_set)
        extra = sorted(dense_set - bfs_set)

        dense_counts = [0 for _ in range(B)]
        bfs_counts = [0 for _ in range(B)]
        if dense_edges.numel() > 0:
            dense_counts = torch.bincount(batch[dense_edges[0]], minlength=B).tolist()
            dense_counts = [int(v) for v in dense_counts]
        if bfs_edges.numel() > 0:
            bfs_counts = torch.bincount(batch[bfs_edges[0]], minlength=B).tolist()
            bfs_counts = [int(v) for v in bfs_counts]

        report[k] = {
            "is_exact_match": len(missing) == 0 and len(extra) == 0,
            "dense_edge_count": int(dense_edges.size(1)),
            "bfs_edge_count": int(bfs_edges.size(1)),
            "dense_counts_by_graph": dense_counts,
            "bfs_counts_by_graph": bfs_counts,
            "missing_edges": [[int(src), int(dst)] for src, dst in missing],
            "extra_edges": [[int(src), int(dst)] for src, dst in extra],
        }

    return report

# ============================================================
# Channel-only FiLM (per-graph) + GraphNorm
# ============================================================

class GraphFiLM(nn.Module):
    """
    GraphNorm + per-graph channel FiLM:
      x -> GraphNorm -> gamma(z_ch), beta(z_ch) -> broadcast to nodes using batch
    """
    def __init__(
        self,
        channels: int,
        z_ch_dim: int,
        *,
        hidden: int = 16,
        init_scale_gamma_c: float = 0.05,
        init_scale_beta_c: float = 0.05,
        enable_channel: bool = True,
    ):
        super().__init__()
        self.norm = GraphNorm(channels)
        self.enable_channel = bool(enable_channel)

        self.to_gb = nn.Sequential(
            nn.Linear(z_ch_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.normal_(self.to_gb[-1].weight, std=1e-4)
        nn.init.zeros_(self.to_gb[-1].bias)

        self.scale_gamma_c = nn.Parameter(torch.tensor(float(init_scale_gamma_c)), requires_grad=False)
        self.scale_beta_c = nn.Parameter(torch.tensor(float(init_scale_beta_c)), requires_grad=False)

    @torch.no_grad()
    def set_strength(self, gamma_c: Optional[float] = None, beta_c: Optional[float] = None):
        if gamma_c is not None:
            self.scale_gamma_c.fill_(float(gamma_c))
        if beta_c is not None:
            self.scale_beta_c.fill_(float(beta_c))

    def forward(self, x: torch.Tensor, batch: torch.Tensor, z_ch: torch.Tensor) -> torch.Tensor:
        x = self.norm(x, batch=batch)
        if not self.enable_channel:
            return x

        g_raw, b_raw = self.to_gb(z_ch).chunk(2, dim=-1)  # (B,C) each

        gamma = 1.0 + self.scale_gamma_c * torch.tanh(g_raw)
        beta = self.scale_beta_c * b_raw

        return x * gamma[batch] + beta[batch]



class LinearMessagePassing(MessagePassing):
    """
    Minimal "conv-like" message passing:
      message: linear(x_j)
      aggregate: aggr (add/mean/max)
    """
    def __init__(self, channels: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(channels, channels, bias=False)
        nn.init.kaiming_normal_(self.lin.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.lin(x_j)


# ============================================================
# DilatedResBlockGNN that mirrors DilatedResBlock_SPADE
#   conv1: exact hop r
#   conv2: 1-hop
#   norm/act at both sites
#   conv2 zero-init for identity-at-init
# ============================================================

class DilatedResBlockGNN(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        norm1: GraphFiLM,
        norm2: GraphFiLM,
        aggr: str = "add",
        zero_init_conv2: bool = True,
    ):
        super().__init__()
        self.dilation = int(dilation)

        self.conv1 = LinearMessagePassing(channels, aggr=aggr)  # will be used on exact-r edges
        self.norm1 = norm1
        self.act1 = nn.SiLU()

        self.conv2 = LinearMessagePassing(channels, aggr=aggr)  # used on 1-hop edges
        if zero_init_conv2:
            nn.init.zeros_(self.conv2.lin.weight)
        self.norm2 = norm2
        self.act2 = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,            # (N,C)
        batch: torch.Tensor,        # (N,)
        *,
        edge_index_r: torch.Tensor, # exact hop r
        edge_index_1: torch.Tensor, # 1-hop adjacency
        z_ch: torch.Tensor,         # (B,z_ch_dim)
    ) -> torch.Tensor:
        h = self.conv1(x, edge_index_r)
        h = self.act1(self.norm1(h, batch=batch, z_ch=z_ch))

        h = self.conv2(h, edge_index_1)
        h = self.act2(self.norm2(h, batch=batch, z_ch=z_ch))

        return x + h

class GraphECA(nn.Module):
    """
    Per-graph global mean pool -> 1D conv over channels -> sigmoid -> gate node features.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return x
        B = int(batch.max().item()) + 1
        C = x.size(-1)

        pooled = x.new_zeros((B, C))
        pooled.index_add_(0, batch, x)
        counts = torch.bincount(batch, minlength=B).clamp_min(1).to(x.dtype).unsqueeze(-1)
        pooled = pooled / counts

        y = pooled.unsqueeze(1)         # (B,1,C)
        y = self.sigmoid(self.conv(y))  # (B,1,C)
        gate = y.squeeze(1)[batch]      # (N,C)
        return x * gate



class DilationStateGNN(nn.Module):
    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        *,
        num_blocks: int = 3,
        dilation_schedule: Optional[Sequence[int]] = None,
        use_eca: bool = True,
        add_z: bool = False,
        z_ch_dim: int = 8,
        add_device_load: bool = True,
        add_progress: bool = False,
        n_devices: int = 5,
        film_in_all_blocks: bool = False,
        film_last_k: int = 2,
        init_gamma_c: float = 0.05,
        init_beta_c: float = 0.05,
        undirected: bool = True,
        aggr: str = "add",
        max_num_nodes: Optional[int] = None,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__()
        if not hasattr(feature_config, "task_feature_dim"):
            raise AttributeError("feature_config must have attribute 'task_feature_dim'")

        self.in_channels = int(feature_config.task_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.debug = bool(debug)

        self.convert_data = HeteroDataWrapper()

        if dilation_schedule is None or len(dilation_schedule) == 0:
            dilation_schedule = [1, 2, 3, 1]
        self.dilation_schedule = list(map(int, dilation_schedule))

        self.add_z = bool(add_z)
        self.add_device_load = bool(add_device_load)
        self.add_progress = bool(add_progress)
        self.n_devices = int(n_devices)
        self.output_dim = self.hidden_channels
        self.output_keys = ["embed"]

        self.undirected = bool(undirected)
        self.aggr = str(aggr)
        self.max_num_nodes = max_num_nodes

        g_dim = 0
        if self.add_z:
            g_dim += int(z_ch_dim)
        if self.add_device_load:
            g_dim += 3 * self.n_devices
        if self.add_progress:
            g_dim += 2

        C_in = self.in_channels
        C = self.hidden_channels

        self.proj = nn.Linear(C_in, C, bias=False)
        self.stem_mp = LinearMessagePassing(C, aggr=self.aggr)
        self.stem_norm = GraphNorm(C)
        self.stem_act = nn.SiLU()

        self.blocks = nn.ModuleList()
        for i in range(int(num_blocks)):
            use_film = bool(film_in_all_blocks or (i >= num_blocks - int(film_last_k)))

            norm1 = GraphFiLM(channels=C, z_ch_dim=g_dim, hidden=16, init_scale_gamma_c=init_gamma_c, init_scale_beta_c=init_beta_c, enable_channel=use_film)
            norm2 = GraphFiLM(channels=C, z_ch_dim=g_dim, hidden=16, init_scale_gamma_c=init_gamma_c, init_scale_beta_c=init_beta_c, enable_channel=use_film)

            dil = self.dilation_schedule[i % len(self.dilation_schedule)]
            self.blocks.append(DilatedResBlockGNN(channels=C, dilation=dil, norm1=norm1, norm2=norm2, aggr=self.aggr, zero_init_conv2=True))

        self.eca = GraphECA(C, k_size=3) if use_eca else nn.Identity()

    @torch.no_grad()
    def set_noise_strength(self, gamma_c: Optional[float] = None, beta_c: Optional[float] = None):
        for blk in self.blocks:
            blk.norm1.set_strength(gamma_c, beta_c)
            blk.norm2.set_strength(gamma_c, beta_c)

    def _build_g(self, observation: TensorDict, B: int) -> torch.Tensor:
        g_list = []
        if self.add_z:
            z_ch = observation[("aux", "z_ch")].reshape(B, -1)
            g_list.append(z_ch)
            
        if self.add_device_load:
            device_load = observation[("aux", "device_load")].reshape(-1, 2 * self.n_devices)
            device_memory = observation[("aux", "device_memory")].reshape(-1, self.n_devices)
            g_list.extend([device_load, device_memory])

        if self.add_progress:
            progress = observation[("aux", "progress")].reshape(-1, 1)
            baseline = observation[("aux", "baseline")].reshape(-1, 1)
            time = observation[("aux", "time")].reshape(-1, 1)
            perc = time / baseline.clamp_min(1e-12)
            g_list.extend([progress, perc])

        if len(g_list) > 0:
            return torch.cat(g_list, dim=-1)
        
        return torch.zeros((B, 0), device=observation["nodes", "tasks", "attr"].device)

    def _flatten_task_ids_from_observation(self, observation: TensorDict, B_flat: int) -> list[int]:
        try:
            task_glb = observation["nodes", "tasks", "glb"].reshape(B_flat, -1)
            task_counts = observation["nodes", "tasks", "count"].reshape(B_flat, -1)[:, 0]
        except Exception:
            return []

        flat_task_ids: list[int] = []
        for graph_id in range(B_flat):
            count = int(task_counts[graph_id].item())
            if count <= 0:
                continue
            flat_task_ids.extend([int(v) for v in task_glb[graph_id, :count].tolist()])
        return flat_task_ids

    @staticmethod
    def _jacobi_row_col_from_task_id(task_id: int, rows: int, cols: int) -> tuple[int, int]:
        total_cells = int(rows * cols)
        if total_cells <= 0:
            return -1, -1
        cell_id = int(task_id) % total_cells
        row = int(cell_id % rows)
        col = int(cell_id // rows)
        return row, col

    @torch.no_grad()
    def debug_dilation_edges(
        self,
        observation: TensorDict,
        dump_dir: Optional[str] = None,
        k_values: Optional[Sequence[int]] = None,
        grid_shape: Optional[Sequence[int]] = None,
    ) -> dict[str, Any]:
        """
        Validate k-hop edge construction against an independent BFS baseline and
        optionally dump CSV/DOT artifacts for visualization.
        """
        batch_shape = observation.batch_size
        B_flat = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1
        data = self.convert_data(observation)

        x = data["tasks"].x
        batch = data["tasks"].batch if isinstance(data, Batch) else None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        N = int(x.size(0))
        edge_index = data["tasks", "to", "tasks"].edge_index

        ei1 = edge_index
        if self.undirected:
            ei1 = torch.cat([ei1, ei1.flip(0)], dim=1)
        ei1 = coalesce(ei1, num_nodes=N)
        ei1, _ = remove_self_loops(ei1)

        ks = sorted({int(v) for v in (k_values if k_values is not None else self.dilation_schedule)})
        k_report = validate_exact_k_hop_edge_index_batched(
            edge_index=ei1,
            batch=batch,
            k_values=ks,
            undirected=False,
            remove_self=True,
            max_num_nodes=self.max_num_nodes,
        )

        node_task_ids = self._flatten_task_ids_from_observation(observation, B_flat)
        if len(node_task_ids) != N:
            node_task_ids = []

        grid_rows = -1
        grid_cols = -1
        if grid_shape is not None and len(grid_shape) == 2:
            grid_rows = int(grid_shape[0])
            grid_cols = int(grid_shape[1])

        report: dict[str, Any] = {
            "num_nodes": N,
            "num_1hop_edges": int(ei1.size(1)),
            "k_hop": {str(k): v for k, v in k_report.items()},
            "grid_shape": [grid_rows, grid_cols] if grid_rows > 0 and grid_cols > 0 else None,
        }

        if dump_dir is not None:
            root = Path(dump_dir)
            root.mkdir(parents=True, exist_ok=True)
            out_dir = root / "dilation_debug"
            suffix = 1
            while out_dir.exists():
                out_dir = root / f"dilation_debug_{suffix}"
                suffix += 1
            out_dir.mkdir(parents=True, exist_ok=False)

            node_lines = ["node_id,batch_id,task_id,row,col"]
            batch_list = [int(v) for v in batch.tolist()]
            for node_id in range(N):
                task_id = int(node_task_ids[node_id]) if len(node_task_ids) == N else -1
                row = -1
                col = -1
                if task_id >= 0 and grid_rows > 0 and grid_cols > 0:
                    row, col = self._jacobi_row_col_from_task_id(task_id, grid_rows, grid_cols)
                node_lines.append(f"{node_id},{batch_list[node_id]},{task_id},{row},{col}")
            (out_dir / "nodes.csv").write_text("\n".join(node_lines) + "\n", encoding="ascii")

            for k in ks:
                exact_edges = exact_k_hop_edge_index_batched_dense(
                    edge_index=ei1,
                    batch=batch,
                    k=int(k),
                    undirected=False,
                    remove_self=True,
                    max_num_nodes=self.max_num_nodes,
                )
                exact_set = _edge_index_to_pair_set(exact_edges)
                missing_set = {tuple(edge) for edge in k_report[k]["missing_edges"]}
                extra_set = {tuple(edge) for edge in k_report[k]["extra_edges"]}

                edge_lines = ["src,dst,src_task,dst_task,src_row,src_col,dst_row,dst_col,status"]
                for src, dst in sorted(exact_set):
                    src_task = int(node_task_ids[src]) if len(node_task_ids) == N else -1
                    dst_task = int(node_task_ids[dst]) if len(node_task_ids) == N else -1
                    src_row = -1
                    src_col = -1
                    dst_row = -1
                    dst_col = -1
                    if grid_rows > 0 and grid_cols > 0 and src_task >= 0 and dst_task >= 0:
                        src_row, src_col = self._jacobi_row_col_from_task_id(src_task, grid_rows, grid_cols)
                        dst_row, dst_col = self._jacobi_row_col_from_task_id(dst_task, grid_rows, grid_cols)
                    status = "exact"
                    if (src, dst) in extra_set:
                        status = "extra_vs_bfs"
                    edge_lines.append(
                        f"{src},{dst},{src_task},{dst_task},{src_row},{src_col},{dst_row},{dst_col},{status}"
                    )
                for src, dst in sorted(missing_set):
                    src_task = int(node_task_ids[src]) if len(node_task_ids) == N else -1
                    dst_task = int(node_task_ids[dst]) if len(node_task_ids) == N else -1
                    src_row = -1
                    src_col = -1
                    dst_row = -1
                    dst_col = -1
                    if grid_rows > 0 and grid_cols > 0 and src_task >= 0 and dst_task >= 0:
                        src_row, src_col = self._jacobi_row_col_from_task_id(src_task, grid_rows, grid_cols)
                        dst_row, dst_col = self._jacobi_row_col_from_task_id(dst_task, grid_rows, grid_cols)
                    edge_lines.append(
                        f"{src},{dst},{src_task},{dst_task},{src_row},{src_col},{dst_row},{dst_col},missing_vs_bfs"
                    )
                (out_dir / f"k{k}_edges.csv").write_text("\n".join(edge_lines) + "\n", encoding="ascii")

                dot_lines = ["digraph KHop {", "  rankdir=LR;"]
                for node_id in range(N):
                    batch_id = batch_list[node_id]
                    task_id = int(node_task_ids[node_id]) if len(node_task_ids) == N else -1
                    label = f"{node_id}|b{batch_id}|t{task_id}" if task_id >= 0 else f"{node_id}|b{batch_id}"
                    if grid_rows > 0 and grid_cols > 0 and task_id >= 0:
                        row, col = self._jacobi_row_col_from_task_id(task_id, grid_rows, grid_cols)
                        dot_lines.append(
                            f'  n{node_id} [label="{label}", pos="{col},{-row}!"];'
                        )
                    else:
                        dot_lines.append(f'  n{node_id} [label="{label}"];')
                for src, dst in sorted(exact_set - extra_set):
                    dot_lines.append(f"  n{src} -> n{dst} [color=black];")
                for src, dst in sorted(extra_set):
                    dot_lines.append(f"  n{src} -> n{dst} [color=blue, style=dotted];")
                for src, dst in sorted(missing_set):
                    dot_lines.append(f"  n{src} -> n{dst} [color=red, style=dashed];")
                dot_lines.append("}")
                (out_dir / f"k{k}_edges.dot").write_text("\n".join(dot_lines) + "\n", encoding="ascii")

            (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="ascii")
            report["dump_dir"] = str(out_dir)

        return report

    def _run_dilation_debug_hook(self, observation: TensorDict, needed: Sequence[int]):
        if os.getenv("TASK4FEEDBACK_DEBUG_DILATION", "0") != "1":
            return

        grid_shape = None
        grid_shape_raw = os.getenv("TASK4FEEDBACK_DEBUG_GRID_SHAPE")
        if grid_shape_raw:
            parts = [p.strip() for p in grid_shape_raw.split(",")]
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                grid_shape = (int(parts[0]), int(parts[1]))

        dump_root = os.getenv("TASK4FEEDBACK_DEBUG_DILATION_DIR")
        report = self.debug_dilation_edges(
            observation=observation,
            dump_dir=dump_root,
            k_values=needed,
            grid_shape=grid_shape,
        )

        mismatched = [k for k, info in report["k_hop"].items() if not info["is_exact_match"]]
        print(
            "[DilationStateGNN][k-hop] "
            f"nodes={report['num_nodes']} base_edges={report['num_1hop_edges']} "
            f"mismatched_hops={mismatched}"
        )

        strict = os.getenv("TASK4FEEDBACK_DEBUG_DILATION_STRICT", "0") == "1"
        if strict and len(mismatched) > 0:
            raise RuntimeError(f"DilationStateGNN k-hop mismatch for hops: {mismatched}")

    def forward(self, observation: TensorDict) -> torch.Tensor:
        batch_shape = observation.batch_size
        B_flat = int(math.prod(batch_shape)) if len(batch_shape) > 0 else 1
        data = self.convert_data(observation)

        x = data["tasks"].x
        batch = data["tasks"].batch if isinstance(data, Batch) else None
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        edge_index = data["tasks", "to", "tasks"].edge_index
        # `batch.max()+1` undercounts when some graphs have zero task nodes.
        # Prefer full graph count from observation batch shape when available.
        B = B_flat
        if len(batch_shape) == 0:
            # Unbatched path: derive graph count from PyG metadata.
            ptr = getattr(data["tasks"], "ptr", None)
            if ptr is not None and ptr.numel() >= 2:
                B = int(ptr.numel()) - 1
            elif batch.numel() > 0:
                B = int(batch.max().item()) + 1

        if batch.numel() > 0:
            max_gid = int(batch.max().item())
            if max_gid >= B:
                raise RuntimeError(
                    f"Invalid task batch ids in DilationStateGNN: max gid {max_gid} >= B={B}."
                )

        g = self._build_g(observation, B)

        N = int(x.size(0))
        M_max = int(observation["nodes", "tasks", "attr"].shape[-2])
        if N == 0:
            if len(batch_shape) == 0:
                return x.new_zeros((M_max, self.hidden_channels))
            return x.new_zeros((*batch_shape, M_max, self.hidden_channels))

        ei1 = edge_index
        if self.undirected:
            ei1 = torch.cat([ei1, ei1.flip(0)], dim=1)
        ei1 = coalesce(ei1, num_nodes=N)
        ei1, _ = remove_self_loops(ei1)

        needed = sorted(set(self.dilation_schedule))
        exact_edges: Dict[int, torch.Tensor] = {}
        for r in needed:
            exact_edges[r] = exact_k_hop_edge_index_batched_dense(
                edge_index=ei1, batch=batch, k=int(r), undirected=False, remove_self=True, max_num_nodes=self.max_num_nodes
            ).to(x.device)
        self._run_dilation_debug_hook(observation, needed)

        h = self.proj(x)
        h = self.stem_mp(h, ei1)
        h = self.stem_act(self.stem_norm(h, batch=batch))

        for blk in self.blocks:
            r = blk.dilation
            h = blk(h, batch=batch, edge_index_r=exact_edges[r], edge_index_1=ei1, z_ch=g)

        if not isinstance(self.eca, nn.Identity):
            h = self.eca(h, batch=batch)

        # Pack ragged node features to a fixed candidate axis.
        # `observation_to_heterodata_truncate` drops inactive nodes per graph, so
        # we cannot reshape by batch size directly.
        C = self.hidden_channels

        if B != B_flat:
            raise RuntimeError(
                f"Inconsistent graph batch size in DilationStateGNN: "
                f"PyG batch has {B} graphs, observation batch has {B_flat} ({batch_shape})."
            )

        counts = torch.bincount(batch, minlength=B_flat)
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        if max_count > M_max:
            raise RuntimeError(
                f"Task count ({max_count}) exceeds padded candidate axis ({M_max}) "
                "in DilationStateGNN."
            )

        h_dense = h.new_zeros((B_flat, M_max, C))
        if h.numel() > 0:
            starts = torch.cumsum(counts, dim=0) - counts
            local_idx = torch.arange(h.size(0), device=h.device) - starts[batch]
            valid = local_idx < M_max
            h_dense[batch[valid], local_idx[valid]] = h[valid]

        if len(batch_shape) == 0:
            return h_dense[0]
        return h_dense.view(*batch_shape, M_max, C)

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helpers: per-graph pooling on a PyG Batch
# ============================================================

def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, B: Optional[int] = None) -> torch.Tensor:
    """
    x: (N,C), batch: (N,)
    returns: (B,C)
    """
    if x.numel() == 0:
        B = 1 if B is None else int(B)
        return x.new_zeros((B, x.size(-1)))
    if B is None:
        B = int(batch.max().item()) + 1
    C = x.size(-1)
    out = x.new_zeros((B, C))
    out.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=B).clamp_min(1).to(x.dtype).unsqueeze(-1)
    return out / counts


def global_add_pool(x: torch.Tensor, batch: torch.Tensor, B: Optional[int] = None) -> torch.Tensor:
    """
    x: (N,C), batch: (N,)
    returns: (B,C)
    """
    if x.numel() == 0:
        B = 1 if B is None else int(B)
        return x.new_zeros((B, x.size(-1)))
    if B is None:
        B = int(batch.max().item()) + 1
    C = x.size(-1)
    out = x.new_zeros((B, C))
    out.index_add_(0, batch, x)
    return out


def global_softmax_attention_pool(
    x: torch.Tensor,
    batch: torch.Tensor,
    scores: torch.Tensor,
    B: Optional[int] = None,
) -> torch.Tensor:
    if x.numel() == 0:
        B = 1 if B is None else int(B)
        return x.new_zeros((B, x.size(-1)))

    if scores.dim() == 2 and scores.size(-1) == 1:
        scores = scores.squeeze(-1)  # (N,)

    if B is None:
        B = int(batch.max().item()) + 1

    # Native PyTorch scatter_reduce for fast segment max
    max_scores = torch.zeros(B, dtype=scores.dtype, device=scores.device)
    max_scores.scatter_reduce_(0, batch, scores, reduce="amax", include_self=False)
    
    scores_shifted = scores - max_scores[batch]
    exp_scores = scores_shifted.exp()
    
    sum_exp = torch.zeros(B, dtype=scores.dtype, device=scores.device)
    sum_exp.scatter_add_(0, batch, exp_scores)
    
    attn = exp_scores / sum_exp[batch].clamp_min(1e-12)

    weighted = x * attn.unsqueeze(-1)  # (N,P)
    pooled = global_add_pool(weighted, batch, B=B)  # (B,P)
    return pooled

# ============================================================
# Policy head: node embeddings -> per-node logits
#   CNN: (B,C,H,W) -> (B,H*W,A)
#   GNN: (N,C) with batch -> (N,A) OR ragged (B, n_i, A)
# ============================================================

class GNNDilationPolicyHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_mode: str = "tiny",
        tiny_std: float = 1e-3,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__()
        self.Cin = int(input_dim)
        self.A = int(output_dim)
        self.debug = bool(debug)

        self.input_keys = ["embed"]
        self.output_dim = self.A

        self.proj = nn.Linear(self.Cin, self.A, bias=True)

        init_mode = init_mode.lower()
        if init_mode == "zero":
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif init_mode == "tiny":
            nn.init.normal_(self.proj.weight, std=float(tiny_std))
            nn.init.zeros_(self.proj.bias)
        elif init_mode == "kaiming":
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
            nn.init.zeros_(self.proj.bias)

    def forward(self, obs, embed: torch.Tensor) -> torch.Tensor:
        logits = self.proj(embed)
        if self.debug:
            print(f"Shape of logits before reshape: {logits.shape}")
        return logits


class GNNDilationValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        z_dim: int = 8,
        proj_dim: int = 8,
        hidden_channels: int = 128,
        tiny_std: float = 1e-3,
        add_gap: bool = True,
        add_progress: bool = True,
        add_device_load: bool = False,
        n_devices: int = 5,
        add_z: bool = False,
        **_ignored,
    ):
        super().__init__()
        C = int(input_dim)
        P = int(proj_dim)
        Dz = int(z_dim) * 2

        self.mix = nn.Linear(C, P, bias=False)
        nn.init.kaiming_normal_(self.mix.weight, nonlinearity="relu")

        self.add_device_load = bool(add_device_load)
        self.add_progress = bool(add_progress)
        self.add_gap = bool(add_gap)
        self.add_z = bool(add_z)
        self.output_dim = 1

        self.attn = nn.Linear(P, 1, bias=True)
        nn.init.normal_(self.attn.weight, std=float(tiny_std))
        nn.init.zeros_(self.attn.bias)

        mlp_in = (2 * P if self.add_gap else P) \
                 + (Dz if self.add_z else 0) \
                 + (3 * int(n_devices) if self.add_device_load else 0) \
                 + (2 if self.add_progress else 0)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, int(hidden_channels)),
            nn.SiLU(),
            nn.Linear(int(hidden_channels), 1),
        )
        nn.init.normal_(self.mlp[-1].weight, std=float(tiny_std))
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, obs, embed: torch.Tensor) -> torch.Tensor:
        batch_shape = obs.batch_size

        # Fixed-shape masked path: embed is (M, C) or (..., M, C).
        if embed.dim() == 2:
            embed_bmc = embed.unsqueeze(0)
            B_flat = 1
        elif embed.dim() >= 3:
            B_flat = int(math.prod(batch_shape)) if len(batch_shape) > 0 else int(embed.shape[0])
            M = int(embed.shape[-2])
            C = int(embed.shape[-1])
            embed_bmc = embed.view(B_flat, M, C)
        else:
            raise RuntimeError(
                f"GNNDilationValueHead expected embed with rank >= 2, got shape {tuple(embed.shape)}"
            )

        if embed_bmc.numel() == 0:
            return embed.new_zeros((*batch_shape, 1))

        M = int(embed_bmc.shape[-2])
        cand_mask = obs["aux", "candidate_mask"].reshape(B_flat, M).to(torch.bool)

        # Avoid NaNs when a sample has no valid candidates.
        has_valid = cand_mask.any(dim=1)
        if not bool(has_valid.all()):
            cand_mask = cand_mask.clone()
            cand_mask[~has_valid, 0] = True

        Fm = F.silu(self.mix(embed_bmc))  # (B, M, P)
        scores = self.attn(Fm).squeeze(-1)  # (B, M)
        weights = masked_softmax(scores, cand_mask, dim=1)
        pooled = (Fm * weights.unsqueeze(-1)).sum(dim=1)  # (B, P)

        if self.add_gap:
            pooled_gap = masked_mean(Fm, cand_mask, dim=1)
            pooled = torch.cat([pooled, pooled_gap], dim=-1)  # (B, 2P)

        if self.add_z:
            z_f = torch.cat([obs[("aux", "z_ch")], obs[("aux", "z_spa")]], dim=-1).reshape(B_flat, -1)
            pooled = torch.cat([pooled, z_f], dim=-1)

        if self.add_device_load:
            device_feat = torch.cat([obs["aux", "device_load"], obs["aux", "device_memory"]], dim=-1).reshape(B_flat, -1)
            pooled = torch.cat([pooled, device_feat], dim=-1)

        if self.add_progress:
            progress = obs["aux", "progress"].reshape(B_flat, -1)
            baseline = obs["aux", "baseline"].reshape(B_flat, -1)
            time = obs["aux", "time"].reshape(B_flat, -1)
            perc = time / baseline.clamp_min(1e-12)
            prog_feat = torch.cat([progress, perc], dim=-1)  
            pooled = torch.cat([pooled, prog_feat], dim=-1)

        v = self.mlp(pooled).squeeze(-1)  
        return v.view(*batch_shape, 1)
