from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, Self

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import observation_to_heterodata, observation_to_heterodata_truncate
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Sequence
from torch_geometric.nn.norm import GraphNorm, MessageNorm

# from task4feedback.interface.wrappers import (
#     observation_to_heterodata_truncate as observation_to_heterodata,
# )
from torch import Tensor
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


def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


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


class ValueOutputHead(OutputHead):
    def __init__(self, *args, **kwargs):
        super(ValueOutputHead, self).__init__(*args, **kwargs)

    def forward(self, obs, emb):
        return super().forward(emb)


class PolicyOutputHead(OutputHead):
    def __init__(self, *args, **kwargs):
        super(PolicyOutputHead, self).__init__(*args, **kwargs)

    def forward(self, obs, emb):
        return super().forward(emb)


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
        if task_features.ndim == 2 and task_features.shape[0] == 1:
            # case [1, k] -> [k]
            task_features = task_features.squeeze(0)
        elif task_features.ndim == 3 and task_features.shape[1] == 1:
            # case [b, 1, k] -> [b, k]
            task_features = task_features.squeeze(1)
        elif task_features.ndim == 4 and task_features.shape[2] == 1:
            # case [b1, b2, 1, k] -> [b1, b2, k]
            task_features = task_features.squeeze(2)
        else:
            raise ValueError(f"Unexpected shape {task_features.shape}")

        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]
            task_features = torch.cat([task_features, time_feature, progress_feature], dim=-1)
        if self.add_device_load:
            device_load = tensordict["aux", "device_load"]
            device_memory = tensordict["aux", "device_memory"]
            task_features = torch.cat([task_features, device_load, device_memory], dim=-1)

        task_activations = self.layers(task_features)

        return task_activations


def _zero_last_linear(seq: nn.Sequential):
    last = None
    for m in reversed(seq):
        if isinstance(m, nn.Linear):
            last = m
            break
    assert last is not None
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


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
                ("tasks", "to", "tasks"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="mean", root_weight=False),
                ("tasks", "from", "tasks"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="mean", root_weight=False),
                ("tasks", "read", "data"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="mean", root_weight=False),
                ("data", "read", "tasks"): SAGEConv(hidden_channels, hidden_channels, project=True, aggr="mean", root_weight=False),
            }
            # conv_dict = {
            #     ("tasks", "to", "tasks"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            #     ("tasks", "from", "tasks"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            #     ("tasks", "read", "data"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            #     ("data", "read", "tasks"): GATv2Conv((self.hidden_channels, self.hidden_channels), self.hidden_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False),
            # }
            hetero_conv = HeteroConv(conv_dict, aggr="mean")
            self.convs.append(hetero_conv)

        self.norms = nn.ModuleDict({
            "tasks": nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(num_layers+1)]),
            "data": nn.ModuleList([nn.LayerNorm(self.hidden_channels) for _ in range(num_layers+1)]),
        })

        self.beta = nn.ModuleDict({
            "tasks": nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_layers)]),
            "data": nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_layers)]),
        })
        for nt in self.beta.keys():
            for b in self.beta[nt]:
                #init to 0.5
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
                "tasks": nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(), nn.Linear(self.hidden_channels, 8)),
                "data": nn.Sequential(nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(), nn.Linear(self.hidden_channels, 8)),
            }
        )

        if self.add_device_load or self.add_progress:
            self.mlp_side_info = nn.Sequential(nn.Linear(self.g_dim, self.hidden_channels), nn.SiLU(), nn.Linear(self.hidden_channels, 8))
        else:
            self.mlp_side_info = None

        self.mlp_norm = nn.LayerNorm(8)

        _tiny_last_linear(self.mlp_global_pool["tasks"])
        _tiny_last_linear(self.mlp_global_pool["data"])
        if self.mlp_side_info is not None:
            _tiny_last_linear(self.mlp_side_info)

        nn.init.zeros_(self.stem_proj["tasks"].bias)
        nn.init.zeros_(self.stem_proj["data"].bias)

        self.output_dim = hidden_channels + 8

        self.output_keys = ["embed"]

    def forward(self, tensordict: TensorDict):
        batch_size = tensordict.batch_size
        data = self.convert_data(tensordict)

        b_tasks = data["tasks"].batch if isinstance(data, Batch) else None
        b_data = data["data"].batch if isinstance(data, Batch) else None

        x_tasks = self.stem_proj["tasks"](data["tasks"].x)
        #x_tasks = self.stem_norm["tasks"](x_tasks)
        x_tasks = self.act(x_tasks)

        x_data = self.stem_proj["data"](data["data"].x)
        #x_data = self.stem_norm["data"](x_data)
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

            #pre-norm
            #x_pre = {nt: self.norms[nt][l](x_dict[nt]) for nt in x_dict.keys()}

            #conv
            x_new = conv(x_dict, edge_index_dict=edge_index_dict)

            # #post-norm
            # x_new  = {nt: self.post_norms[nt][l](x_dict[nt], x_new[nt]) for nt in x_new.keys()}

            x_new = {nt: self.norms[nt][l](x_new[nt]) for nt in x_new.keys()}

            #film
            if self.film is not None:
                x_new = self.film(x_new, batch_dict, g=g, layer_idx=l)

            #activation
            x_new = {nt: self.act(x_new[nt]) for nt in x_new.keys()}

            # residual 
            for nt in x_dict.keys():
                beta = self.beta[nt][l]
                beta = torch.sigmoid(beta)
                x_new[nt] = (1 - beta) * x_dict[nt] + beta * x_new[nt]

            #update for next layer
            x_dict = {nt: x_new[nt] for nt in x_new.keys()}

        # final norm
        # x_dict = {nt: self.norms[nt][-1](x_dict[nt]) for nt in x_dict.keys()}

        if b_tasks is not None:
            idx = data["tasks"].ptr[:-1]
            x = x_dict["tasks"][idx]
        else:
            x = x_dict["tasks"][0]

        pooled_tasks = global_mean_pool(x_dict["tasks"], b_tasks)
        pooled_data = global_mean_pool(x_dict["data"], b_data)

        pt_f = self.mlp_global_pool["tasks"](pooled_tasks)
        pd_f = self.mlp_global_pool["data"](pooled_data)

        if self.mlp_side_info is not None:
            g_f = self.mlp_side_info(g)
            y = pt_f + pd_f + g_f
        else:
            y = pt_f + pd_f

        y = self.mlp_norm(y)
        y = self.act(y)

        if b_tasks is None:
            y = y.squeeze(0)

        # print(f"x shape before cat: {x.shape}, y shape: {y.shape}, batch_size: {batch_size}")
        x = torch.cat([x, y], dim=-1)
        x = x.reshape(*batch_size, -1, x.shape[-1])
        # print(f"x shape before return: {x.shape}")
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
        data= self.convert_data(tensordict)

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


class OriginalUNetState(nn.Module):
    def __init__(self, feature_config: FeatureDimConfig, hidden_channels: int, width: int, add_progress: bool = False, **_ignored):
        super().__init__()
        self.output_keys = []
        self.width = width
        self.in_channels = feature_config.task_feature_dim
        self.hidden_channels = hidden_channels
        self.add_progress = add_progress
        # Determine number of downsampling layers based on width
        self.num_layers = int(math.floor(math.log2(width)))
        # Create encoder conv blocks dynamically
        self.enc_blocks = nn.ModuleList()

        channels = self.in_channels
        for i in range(self.num_layers):
            out_channels = hidden_channels * (2**i)
            block = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(
                    inplace=True,
                    negative_slope=0.01,
                ),
            )
            self.enc_blocks.append(block)
            channels = out_channels
            self.output_keys.append(f"enc_{i}")
        self.pool = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.LeakyReLU(
                inplace=True,
                negative_slope=0.01,
            ),
        )
        self.output_dim = channels + (1 if add_progress else 0)
        self.output_keys.append("embed")

    def forward(self, x):
        # x is a TensorDict; x.batch_size might be [], [N], [N,M], etc.
        single = x.batch_size == torch.Size([])
        width = self.width

        # 1) Pull out the tasks tensor: shape = (*batch_shape, tasks, in_channels)
        x_tasks = x["nodes", "tasks", "attr"]
        if single:
            # If single sample, add a batch dimension
            x_tasks = x_tasks.unsqueeze(0)

        *batch_shape, tasks, in_channels = x_tasks.shape

        # 2) Flatten all leading batch dims into one:
        flat_bs = 1
        for d in batch_shape:
            flat_bs *= d

        # 3) Reshape into (flat_bs, tasks, in_channels)
        x_flat = x_tasks.reshape(flat_bs, tasks, in_channels)

        # 4) Convert 'tasks' → spatial dims (width × width), then to (flat_bs, C_in, W, W)
        x_flat = x_flat.view(flat_bs, width, width, in_channels).permute(0, 3, 1, 2)  # (flat_bs, W, W, C_in)  # (flat_bs, C_in, W, W)

        # 5) Run through encoder blocks + pooling, collecting intermediate feats
        enc_feats_flat = []
        x_enc = x_flat
        for block in self.enc_blocks:
            x_enc = block(x_enc)
            enc_feats_flat.append(x_enc)
            x_enc = self.pool(x_enc)

        # 6) Bottleneck + flatten spatial → (flat_bs, feat_dim)
        b_flat = self.bottleneck(x_enc).flatten(start_dim=1)

        # 7) Un-flatten back to original batch_shape:
        #    a) intermediate feature maps
        enc_feats = []
        for feat in enc_feats_flat:
            # feat is (flat_bs, C, H, W) → reshape to (*batch_shape, C, H, W)
            enc_feats.append(feat.view(*batch_shape, *feat.shape[1:]))

        #    b) bottleneck vector
        if single:
            b = b_flat.squeeze(0)
        else:
            b = b_flat.view(*batch_shape, -1)

        # 8) Optionally concat progress feature
        if self.add_progress:
            prog = x["aux", "progress"]
            b = torch.cat([b, prog], dim=-1)

        return (*enc_feats, b)


class OriginalUNetPolicyHead(nn.Module):

    def __init__(self, input_dim: int, hidden_channels: int, width: int, output_dim: int, **_ignored):
        super().__init__()
        self.width = width
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_layers = int(math.floor(math.log2(width)))

        # Create upsampling and decoder blocks dynamically
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(self.num_layers)):
            in_ch = hidden_channels * (2 ** (i + 1)) if i < self.num_layers - 1 else hidden_channels * (2**i)
            out_ch = hidden_channels * (2**i)
            self.up_blocks.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        self.input_keys = []
        for i in range(self.num_layers):
            self.input_keys.append(f"enc_{i}")
        self.input_keys.append("embed")
        self.out_conv = nn.Conv2d(hidden_channels, output_dim, kernel_size=1)

    def _align_and_concat(self, up_feat, enc_feat):
        uh, uw = up_feat.shape[-2:]
        eh, ew = enc_feat.shape[-2:]
        dh, dw = eh - uh, ew - uw
        if dh > 0 or dw > 0:
            pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
            up_feat = F.pad(up_feat, pad)
        elif dh < 0 or dw < 0:
            top, left = (-dh) // 2, (-dw) // 2
            up_feat = up_feat[..., top : top + eh, left : left + ew]
        return torch.cat([up_feat, enc_feat], dim=1)

    def forward(self, obs, *features):
        single = obs.batch_size == torch.Size([])
        enc_feats = features[:-1]
        b = features[-1]

        if not single:
            *batch_shape, embed_dim = b.shape
            flat_bs = 1
            for d in batch_shape:
                flat_bs *= d

            b = b.reshape(flat_bs, embed_dim, 1, 1)
            b = b.view(
                flat_bs,
                self.hidden_channels * (2 ** (self.num_layers - 1)),
                self.width // (2**self.num_layers),
                self.width // (2**self.num_layers),
            )
        else:
            b = b.unsqueeze(0)
            b = b.view(
                1,
                self.hidden_channels * (2 ** (self.num_layers - 1)),
                self.width // (2**self.num_layers),
                self.width // (2**self.num_layers),
            )

        for up, dec, enc in zip(self.up_blocks, self.dec_blocks, reversed(enc_feats)):
            b = up(b)
            if not single:
                enc = enc.view(flat_bs, *enc.shape[len(batch_shape) :])
            b = self._align_and_concat(b, enc)
            b = dec(b)
        logits = self.out_conv(b)
        logits = logits.permute(0, 2, 3, 1).flatten(1, 2)
        if single:
            logits = logits.squeeze(0)
        else:
            logits = logits.view(*batch_shape, self.width * self.width, -1)
        return logits


class OriginalUNetValueHead(nn.Module):
    """
    Wrapper for OutputHead to match UNet interface
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        activation: Optional[nn.Module] = None,
        initialization: Optional[dict] = None,
        layer_norm: bool = True,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.debug = bool(debug)

        self.head = OutputHead(
            input_dim=self.input_dim,
            hidden_channels=self.hidden_channels,
            output_dim=self.output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
        )
        self.output_dim = output_dim

    def forward(self, obs, *features):
        return self.head(features[-1])  # use only bottleneck features
