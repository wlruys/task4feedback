from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, Self

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import observation_to_heterodata
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Sequence


# from task4feedback.interface.wrappers import (
#     observation_to_heterodata_truncate as observation_to_heterodata,
# )

from torch.profiler import record_function

from tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
    GATConv,
    GraphConv,
    SimpleConv,
    EdgeConv,
    global_mean_pool,
    global_add_pool,
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
        nn.init.kaiming_uniform_(
            layer.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(
            layer.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
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

    def _convert_to_heterodata(
        self, obs: TensorDict, is_batch: bool = False
    ) -> HeteroData | Batch:
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
    def __init__(self, network: nn.Module, device: Optional[str] = "cpu"):
        super(HeteroDataWrapper, self).__init__()
        self.network = network

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
                _obs = observation_to_heterodata(obs, actions=actions)
            else:
                _obs = observation_to_heterodata(obs)

            task_count = obs["nodes", "tasks", "count"]
            data_count = obs["nodes", "data", "count"]

            if is_cuda:
                _obs = _obs.to("cuda", non_blocking=True)
                task_count = task_count.to("cuda", non_blocking=True)
                data_count = data_count.to("cuda", non_blocking=True)

            return (
                _obs,
                obs["nodes", "tasks", "count"],
                obs["nodes", "data", "count"],
            )

        # Otherwise we're batching, possibly over multiple batch dimensions

        # flatten and save the batch size
        self.batch_size = obs.batch_size
        # obs = obs.reshape(-1)

        _h_data = []

        print("obs", obs.shape, obs.batch_size)

        if is_cuda:
            for i in range(obs.batch_size[0]):
                if actions is not None:
                    _obs = observation_to_heterodata(obs[i], actions=actions[i])
                else:
                    _obs = observation_to_heterodata(obs[i])

                # _obs = _obs.to("cuda", non_blocking=True)

                _h_data.append(_obs)
        else:
            for i in range(obs.batch_size[0]):
                if actions is not None:
                    _obs = observation_to_heterodata(obs[i], actions=actions[i])
                else:
                    print(i, len(obs))
                    print(obs[i])
                    _obs = observation_to_heterodata(obs[i])
                _h_data.append(_obs)

        batch_obs = Batch.from_data_list(_h_data)
        task_count = obs["nodes", "tasks", "count"]
        data_count = obs["nodes", "data", "count"]

        if isinstance(batch_obs, tuple):
            batch_obs = batch_obs[0]

        if is_cuda:
            task_count = task_count.to("cuda", non_blocking=True)
            data_count = data_count.to("cuda", non_blocking=True)
            batch_obs = batch_obs.to("cuda", non_blocking=True)

        return (batch_obs, task_count, data_count)

    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)

        with torch.no_grad():
            data, task_count, data_count = self._convert_to_heterodata(
                obs, is_batch, actions=actions
            )

        out = self.network(data, (task_count, data_count))
        return out


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
            device_feature_dim=overrides.get(
                "device_feature_dim", other.device_feature_dim
            ),
            task_data_edge_dim=overrides.get(
                "task_data_edge_dim", other.task_data_edge_dim
            ),
            task_device_edge_dim=overrides.get(
                "task_device_edge_dim", other.task_device_edge_dim
            ),
            task_task_edge_dim=overrides.get(
                "task_task_edge_dim", other.task_task_edge_dim
            ),
        )


@dataclass
class LayerConfig:
    hidden_channels: int = 16
    n_heads: int = 1
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None


class HeteroGAT1Layer(nn.Module):
    def __init__(self, feature_config: FeatureDimConfig, layer_config: LayerConfig):
        super(HeteroGAT1Layer, self).__init__()

        self.feature_config = feature_config
        self.layer_config = layer_config

        self.gnn_tasks_data = GATv2Conv(
            (feature_config.data_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            # edge_dim=feature_config.task_data_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATv2Conv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            # edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_devices = GATv2Conv(
            (feature_config.device_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            # edge_dim=feature_config.task_device_edge_dim,
            add_self_loops=False,
        )

        self.layer_norm1 = nn.LayerNorm(layer_config.hidden_channels)
        self.layer_norm2 = nn.LayerNorm(layer_config.hidden_channels)
        self.layer_norm3 = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )

    def forward(self, data):
        data_fused_tasks = self.gnn_tasks_data(
            (data["data"].x, data["tasks"].x),
            data["data", "to", "tasks"].edge_index,
            # data["data", "to", "tasks"].edge_attr,
        )

        tasks_fused_tasks = self.gnn_tasks_tasks(
            (data["tasks"].x, data["tasks"].x),
            data["tasks", "to", "tasks"].edge_index,
            # data["tasks", "to", "tasks"].edge_attr,
        )

        devices_fused_tasks = self.gnn_tasks_devices(
            (data["devices"].x, data["tasks"].x),
            data["devices", "to", "tasks"].edge_index,
            # data["devices", "to", "tasks"].edge_attr,
        )

        data_fused_tasks = self.layer_norm1(data_fused_tasks)
        tasks_fused_tasks = self.layer_norm2(tasks_fused_tasks)
        devices_fused_tasks = self.layer_norm3(devices_fused_tasks)

        data_fused_tasks = self.activation(data_fused_tasks)
        tasks_fused_tasks = self.activation(tasks_fused_tasks)
        devices_fused_tasks = self.activation(devices_fused_tasks)

        return torch.cat(
            [data["tasks"].x, data_fused_tasks, tasks_fused_tasks, devices_fused_tasks],
            dim=-1,
        )


class NoDeviceHeteroGAT1Layer(nn.Module):
    def __init__(self, feature_config: FeatureDimConfig, layer_config: LayerConfig):
        super(NoDeviceHeteroGAT1Layer, self).__init__()

        self.feature_config = feature_config
        self.layer_config = layer_config

        self.gnn_tasks_data = GATv2Conv(
            (feature_config.data_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            edge_dim=feature_config.task_data_edge_dim,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATv2Conv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        )

        self.layer_norm1 = nn.LayerNorm(layer_config.hidden_channels)
        self.layer_norm2 = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = (
            layer_config.hidden_channels * 2 + feature_config.task_feature_dim
        )

    def forward(self, data):
        data_fused_tasks = self.gnn_tasks_data(
            (data["data"].x, data["tasks"].x),
            data["data", "to", "tasks"].edge_index,
            data["data", "to", "tasks"].edge_attr,
        )

        tasks_fused_tasks = self.gnn_tasks_tasks(
            (data["tasks"].x, data["tasks"].x),
            data["tasks", "to", "tasks"].edge_index,
        )

        data_fused_tasks = self.layer_norm1(data_fused_tasks)
        tasks_fused_tasks = self.layer_norm2(tasks_fused_tasks)

        data_fused_tasks = self.activation(data_fused_tasks)
        tasks_fused_tasks = self.activation(tasks_fused_tasks)

        return torch.cat(
            [data["tasks"].x, data_fused_tasks, tasks_fused_tasks],
            dim=-1,
        )


class NoDeviceSAGE1Layer(nn.Module):
    def __init__(self, feature_config: FeatureDimConfig, layer_config: LayerConfig):
        super(NoDeviceSAGE1Layer, self).__init__()

        self.feature_config = feature_config
        self.layer_config = layer_config

        self.gnn_tasks_data = SAGEConv(
            (feature_config.data_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            project=True,
        )

        self.gnn_tasks_tasks = SAGEConv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            project=True,
        )

        self.layer_norm1 = nn.LayerNorm(layer_config.hidden_channels)
        self.layer_norm2 = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = (
            layer_config.hidden_channels * 2 + feature_config.task_feature_dim
        )

    def forward(self, data):
        data_fused_tasks = self.gnn_tasks_data(
            (data["data"].x, data["tasks"].x),
            data["data", "to", "tasks"].edge_index,
        )

        tasks_fused_tasks = self.gnn_tasks_tasks(
            (data["tasks"].x, data["tasks"].x),
            data["tasks", "to", "tasks"].edge_index,
        )

        data_fused_tasks = self.layer_norm1(data_fused_tasks)
        tasks_fused_tasks = self.layer_norm2(tasks_fused_tasks)

        data_fused_tasks = self.activation(data_fused_tasks)
        tasks_fused_tasks = self.activation(tasks_fused_tasks)

        return torch.cat(
            [data["tasks"].x, data_fused_tasks, tasks_fused_tasks],
            dim=-1,
        )


class DataTaskGAT2Layer(nn.Module):
    def __init__(self, feature_config: FeatureDimConfig, layer_config: LayerConfig):
        super(DataTaskGAT2Layer, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.data_task_conv = HeteroConv(
            {
                ("data", "to", "tasks"): GATConv(
                    (feature_config.data_feature_dim, feature_config.task_feature_dim),
                    layer_config.hidden_channels,
                    heads=layer_config.n_heads,
                    concat=True,
                    residual=True,
                    dropout=0,
                    edge_dim=feature_config.task_data_edge_dim,
                    add_self_loops=False,
                ),
                ("tasks", "to", "data"): GATConv(
                    (feature_config.task_feature_dim, feature_config.data_feature_dim),
                    layer_config.hidden_channels,
                    heads=layer_config.n_heads,
                    concat=True,
                    residual=True,
                    dropout=0,
                    edge_dim=feature_config.task_data_edge_dim,
                    add_self_loops=False,
                ),
            }
        )

        hidden_channels_with_heads = layer_config.hidden_channels * layer_config.n_heads

        self.task_data_conv = HeteroConv(
            {
                ("tasks", "to", "data"): GATConv(
                    (hidden_channels_with_heads, hidden_channels_with_heads),
                    layer_config.hidden_channels,
                    heads=layer_config.n_heads,
                    concat=False,
                    add_self_loops=False,
                    residual=True,
                    dropout=0,
                    edge_dim=feature_config.task_data_edge_dim,
                ),
                ("data", "to", "tasks"): GATConv(
                    (hidden_channels_with_heads, hidden_channels_with_heads),
                    layer_config.hidden_channels,
                    heads=layer_config.n_heads,
                    concat=False,
                    add_self_loops=False,
                    residual=True,
                    dropout=0,
                    edge_dim=feature_config.task_data_edge_dim,
                ),
            }
        )

        self.norm_tasks = nn.LayerNorm(hidden_channels_with_heads)
        self.norm_data = nn.LayerNorm(hidden_channels_with_heads)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = layer_config.hidden_channels

    def forward(self, data: HeteroData | Batch):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = (
            data.edge_attr_dict if hasattr(data, "edge_attr_dict") else None
        )
        x_dict = self.data_task_conv(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {
            "tasks": self.norm_tasks(x_dict["tasks"]),
            "data": self.norm_data(x_dict["data"]),
        }
        x_dict = {node_type: self.activation(x) for node_type, x in x_dict.items()}
        x_dict = self.task_data_conv(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {node_type: self.activation(x) for node_type, x in x_dict.items()}

        return x_dict


class TaskTaskGATkLayer(nn.Module):
    """
    Tasks-to-tasks encodes task -> dependency information

    This module performs k layers of GAT convolutions on the task nodes:
    - k accumulations of task -> dependency information
    - k accumulations of task -> dependant information

    These results are then concatenated and returned as the output (hidden_channels * 2)
    """

    def __init__(
        self,
        input_dim: int,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        k: int = 2,
        skip_connection: bool = True,
        use_edge_features: bool = False,
    ):
        super(TaskTaskGATkLayer, self).__init__()

        if not use_edge_features:
            edge_dim = None
        else:
            edge_dim = feature_config.task_task_edge_dim

        # Create k layers for dependency and dependent paths
        self.conv_dependency_layers = nn.ModuleList()
        self.conv_dependent_layers = nn.ModuleList()

        # First layer gets input_dim
        self.conv_dependency_layers.append(
            GATv2Conv(
                (input_dim, input_dim),
                layer_config.hidden_channels,
                heads=layer_config.n_heads,
                concat=False,
                residual=True,
                dropout=0,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )

        self.conv_dependent_layers.append(
            GATv2Conv(
                (input_dim, input_dim),
                layer_config.hidden_channels,
                heads=layer_config.n_heads,
                concat=False,
                residual=True,
                dropout=0,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )

        # Remaining layers get hidden_channels as input
        for _ in range(1, k):
            self.conv_dependency_layers.append(
                GATv2Conv(
                    (layer_config.hidden_channels, layer_config.hidden_channels),
                    layer_config.hidden_channels,
                    heads=layer_config.n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )

            self.conv_dependent_layers.append(
                GATv2Conv(
                    (layer_config.hidden_channels, layer_config.hidden_channels),
                    layer_config.hidden_channels,
                    heads=layer_config.n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )

        self.norm_dependency = nn.LayerNorm(layer_config.hidden_channels)
        self.norm_dependant = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_dim = layer_config.hidden_channels * 2

        self.use_edge_features = use_edge_features
        self.k = k

        self.skip_connection = skip_connection
        if skip_connection:
            self.output_dim += input_dim  # Use input_dim for skip connection

    def forward(self, task_embedding, data: HeteroData | Batch):
        tasks = task_embedding
        edge_dependency_index = data.edge_index_dict["tasks", "to", "tasks"]
        edge_dependant_index = edge_dependency_index.flip(0)

        if self.use_edge_features:
            edge_attr = data.edge_attr_dict["tasks", "to", "tasks"]
        else:
            edge_attr = None

        tasks_dependency = tasks
        tasks_dependant = tasks

        # Process through all layers
        for i in range(self.k):
            tasks_dependency = self.conv_dependency_layers[i](
                tasks_dependency, edge_dependency_index, edge_attr
            )
            tasks_dependency = self.norm_dependency(tasks_dependency)
            tasks_dependency = self.activation(tasks_dependency)

            tasks_dependant = self.conv_dependent_layers[i](
                tasks_dependant, edge_dependant_index, edge_attr
            )
            tasks_dependant = self.norm_dependant(tasks_dependant)
            tasks_dependant = self.activation(tasks_dependant)

        if self.skip_connection:
            task_embedding = torch.cat(
                [tasks_dependency, tasks_dependant, tasks], dim=-1
            )
        else:
            task_embedding = torch.cat([tasks_dependency, tasks_dependant], dim=-1)

        return task_embedding


class TaskTaskGAT1Layer(nn.Module):
    """
    Tasks-to-tasks encodes task -> dependency information

    This module performs one layer of GAT convolutions on the task nodes:
    - One accumulation of task -> dependency information
    - One accumulation of task -> dependant information

    These results are then concatenated and returned as the output (hidden_channels * 2)
    """

    def __init__(
        self,
        input_dim: int,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        skip_connection: bool = True,
        use_edge_features: bool = False,
    ):
        super(TaskTaskGAT1Layer, self).__init__()

        if not use_edge_features:
            edge_dim = None
        else:
            edge_dim = feature_config.task_task_edge_dim

        self.conv_dependency = GATv2Conv(
            (input_dim, input_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent = GATv2Conv(
            (input_dim, input_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        self.norm_dependency = nn.LayerNorm(layer_config.hidden_channels)
        self.norm_dependant = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_dim = layer_config.hidden_channels * 2

        self.use_edge_features = use_edge_features

        self.skip_connection = skip_connection
        if skip_connection:
            self.output_dim += input_dim  # Use input_dim for skip connection

    def forward(self, task_embedding, data: HeteroData | Batch):
        tasks = task_embedding
        edge_dependency_index = data.edge_index_dict["tasks", "to", "tasks"]
        edge_dependant_index = edge_dependency_index.flip(0)

        if self.use_edge_features:
            edge_attr = data.edge_attr_dict["tasks", "to", "tasks"]
        else:
            edge_attr = None

        tasks_dependency = self.conv_dependency(tasks, edge_dependency_index, edge_attr)
        tasks_dependency = self.norm_dependency(tasks_dependency)
        tasks_dependency = self.activation(tasks_dependency)

        tasks_dependant = self.conv_dependent(tasks, edge_dependant_index, edge_attr)
        tasks_dependant = self.norm_dependant(tasks_dependant)
        tasks_dependant = self.activation(tasks_dependant)

        if self.skip_connection:
            task_embedding = torch.cat(
                [tasks_dependency, tasks_dependant, tasks], dim=-1
            )
        else:
            task_embedding = torch.cat([tasks_dependency, tasks_dependant], dim=-1)

        return task_embedding


class OutputHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
        debug: bool = False
    ):
        super(OutputHead, self).__init__()

        #print(initialization)
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
        layers.append(
            instantiate(activation) if activation else nn.LeakyReLU(negative_slope=0.01)
        )
        layers.append(layer2_init(nn.Linear(hidden_channels, output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.debug:
            printf("[OutputHead] input {x.shape}")
        return self.network(x)


class LogitStabilizer(nn.Module):
    def __init__(self, init_tau=2.0, learnable=True):
        super(LogitStabilizer, self).__init__()
        t = torch.tensor(float(init_tau)).log()
        self.log_tau = nn.Parameter(t, requires_grad=learnable)

    @property
    def tau(self):
        return self.log_tau.exp().clamp_min(1.0)
    
    def forward(self, logits):
        logits = logits - logits.mean(dim=-1, keepdim=True)
        logits = logits / self.tau
        return logits


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
        debug: bool = False
    ):
        super(LogitsOutputHead, self).__init__(
            input_dim,
            hidden_channels,
            output_dim,
            activation=activation,
            initialization=initialization,
            layer_norm=layer_norm,
            debug=debug
        )
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

class OldOutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, logits=False):
        super(OldOutputHead, self).__init__()

        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = layer_init(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CombineTwoLayer(nn.Module):
    def __init__(self, x_shape, y_shape, hidden_shape, output_shape):
        super(CombineTwoLayer, self).__init__()
        self.fc_x = layer_init(nn.Linear(x_shape, hidden_shape))
        self.fc_y = layer_init(nn.Linear(y_shape, hidden_shape))
        self.fc_c = layer_init(nn.Linear(hidden_shape, output_shape))
        self.layer_norm_x = nn.LayerNorm(hidden_shape)
        self.layer_norm_y = nn.LayerNorm(hidden_shape)
        self.layer_norm_c = nn.LayerNorm(output_shape)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, y):
        x = self.fc_x(x)
        x = self.layer_norm_x(x)
        x = self.activation(x)

        y = self.fc_y(y)
        y = self.layer_norm_y(y)
        y = self.activation(y)

        z = x + y
        z = self.fc_c(z)
        z = self.layer_norm_c(z)
        z = self.activation(z)

        return z


class CombineThreeLayer(nn.Module):
    def __init__(self, x_shape, y_shape, z_shape, hidden_shape, output_shape):
        super(CombineThreeLayer, self).__init__()
        self.fc_x = layer_init(nn.Linear(x_shape, hidden_shape))
        self.fc_y = layer_init(nn.Linear(y_shape, hidden_shape))
        self.fc_z = layer_init(nn.Linear(z_shape, hidden_shape))
        self.fc_c = layer_init(nn.Linear(hidden_shape, output_shape))
        self.layer_norm_x = nn.LayerNorm(hidden_shape)
        self.layer_norm_y = nn.LayerNorm(hidden_shape)
        self.layer_norm_z = nn.LayerNorm(hidden_shape)
        self.layer_norm_c = nn.LayerNorm(output_shape)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, y, z):
        x = self.fc_x(x)
        x = self.layer_norm_x(x)
        x = self.activation(x)

        y = self.fc_y(y)
        y = self.layer_norm_y(y)
        y = self.activation(y)

        z = self.fc_z(z)
        z = self.layer_norm_z(z)
        z = self.activation(z)

        c = x + y + z
        c = self.fc_c(c)
        c = self.layer_norm_c(c)
        c = self.activation(c)

        return z


class DeviceGlobalLayer(nn.Module):
    def __init__(self, feature_config: FeatureDimConfig, layer_config: LayerConfig):
        super(DeviceGlobalLayer, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        # Linear layer from dvice dim to hidden dim
        self.device_layer = layer_init(
            nn.Linear(feature_config.device_feature_dim, layer_config.hidden_channels)
        )

        self.norm_tasks = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        ##Make
        # self.edges_base = np.edge_index

    def forward(self, data: HeteroData | Batch):
        devices = data["devices"].x
        # print("devices", devices.shape)
        device_embeddings = self.device_layer(devices)
        device_embeddings = self.norm_tasks(device_embeddings)
        device_embeddings = self.activation(device_embeddings)

        return device_embeddings


class DataTaskGATkLayer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        k: int = 1,
        skip_connection: bool = True,
    ):
        super(DataTaskGATkLayer, self).__init__()

        self.layer_config = layer_config
        self.k = k

        # Create ModuleLists for the alternating direction convolutions
        self.conv_data_task_layers = nn.ModuleList()
        self.conv_task_data_layers = nn.ModuleList() if k > 1 else None

        # Layer norms for each direction
        self.layer_norm_task = nn.ModuleList()
        self.layer_norm_data = nn.ModuleList() if k > 1 else None

        # First layer: Data -> Task
        self.conv_data_task_layers.append(
            GATv2Conv(
                (feature_config.data_feature_dim, feature_config.task_feature_dim),
                layer_config.hidden_channels,
                heads=layer_config.n_heads,
                concat=False,
                residual=True,
                dropout=0,
                edge_dim=feature_config.task_data_edge_dim,
                add_self_loops=False,
            )
        )
        self.layer_norm_task.append(nn.LayerNorm(layer_config.hidden_channels))

        # For k > 1, we need additional layers in alternating directions
        for i in range(1, k):
            if i % 2 == 1:  # Task -> Data layer
                if i == 1:
                    self.conv_task_data_layers = nn.ModuleList()
                    self.layer_norm_data = nn.ModuleList()

                self.conv_task_data_layers.append(
                    GATv2Conv(
                        (layer_config.hidden_channels, layer_config.hidden_channels),
                        layer_config.hidden_channels,
                        heads=layer_config.n_heads,
                        concat=False,
                        residual=True,
                        dropout=0,
                        edge_dim=feature_config.task_data_edge_dim,
                        add_self_loops=False,
                    )
                )
                self.layer_norm_data.append(nn.LayerNorm(layer_config.hidden_channels))
            else:  # Data -> Task layer (second round onwards)
                self.conv_data_task_layers.append(
                    GATv2Conv(
                        (layer_config.hidden_channels, layer_config.hidden_channels),
                        layer_config.hidden_channels,
                        heads=layer_config.n_heads,
                        concat=False,
                        residual=True,
                        dropout=0,
                        edge_dim=feature_config.task_data_edge_dim,
                        add_self_loops=False,
                    )
                )
                self.layer_norm_task.append(nn.LayerNorm(layer_config.hidden_channels))

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_dim = layer_config.hidden_channels

        self.skip_connection = skip_connection
        if skip_connection:
            self.output_dim += feature_config.task_feature_dim

    def forward(self, data: HeteroData | Batch):
        task_features = data["tasks"].x
        data_features = data["data"].x
        data_task_edges = data["data", "to", "tasks"].edge_index
        task_data_edges = data["tasks", "to", "data"].edge_index
        data_task_edges_attr = data["data", "to", "tasks"].edge_attr

        # Initial task embeddings from first layer
        task_embeddings = self.conv_data_task_layers[0](
            (data_features, task_features),
            data_task_edges,
            data_task_edges_attr,
        )
        task_embeddings = self.layer_norm_task[0](task_embeddings)
        task_embeddings = self.activation(task_embeddings)

        # Process through remaining layers if k > 1
        data_embeddings = None
        for i in range(1, self.k):
            if i % 2 == 1:  # Task -> Data direction
                data_embeddings = self.conv_task_data_layers[(i - 1) // 2](
                    (task_embeddings, data_features),
                    task_data_edges,
                    data_task_edges_attr,
                )
                data_embeddings = self.layer_norm_data[(i - 1) // 2](data_embeddings)
                data_embeddings = self.activation(data_embeddings)
                data_features = data_embeddings
            else:  # Data -> Task direction
                task_embeddings = self.conv_data_task_layers[i // 2](
                    (data_features, task_embeddings),
                    data_task_edges,
                    data_task_edges_attr,
                )
                task_embeddings = self.layer_norm_task[i // 2](task_embeddings)
                task_embeddings = self.activation(task_embeddings)

        # Add skip connection if requested
        if self.skip_connection:
            task_embeddings = torch.cat([task_embeddings, data["tasks"].x], dim=-1)

        return task_embeddings


class DeviceCandidateGAT(nn.Module):
    def __init__(
        self,
        device_embed_dim: int,
        candidate_embed_dim: int,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
    ):
        super(DeviceCandidateGAT, self).__init__()

        self.layer_config = layer_config

        self.conv_device_candidate = GATv2Conv(
            (device_embed_dim, candidate_embed_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            add_self_loops=False,
        )

        self.output_layer = layer_init(
            nn.Linear(layer_config.hidden_channels, layer_config.hidden_channels)
        )
        self.output_dim = layer_config.hidden_channels

        self.layer_norm = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, device_embeddings, candidate_embedding, edge_index):
        # print("device_embeddings", device_embeddings.shape)
        # print("candidate_embeddings", candidate_embeddings.shape)
        x = self.conv_device_candidate(
            (device_embeddings, candidate_embedding), edge_index
        )
        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.output_layer(x)

        return x


class DeviceAssignmentNet2Layer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
    ):
        super(DeviceAssignmentNet2Layer, self).__init__()

        self.feature_config = feature_config
        self.layer_config = layer_config
        self.n_devices = n_devices

        # Returns embeddings for tasks and data nodes at depth 2
        # Output feature dim:
        # dict of ("tasks": hidden_channels, "data": hidden_channels)
        self.data_task_layer = DataTaskGAT2Layer(feature_config, layer_config)

        # Returns concatenated embeddings for tasks at depth 2
        # Two directions of task -> task information (dependency and dependant)
        # Output feature dim: hidden_channels * 2
        self.task_task_layer = TaskTaskGATkLayer(
            layer_config.hidden_channels, feature_config, layer_config
        )

        self.device_layer = DeviceGlobalLayer(feature_config, layer_config)

        # Combination layer
        self.combine_layer = CombineThreeLayer(
            self.layer_config.hidden_channels,
            self.layer_config.hidden_channels * 2,
            self.layer_config.hidden_channels,
            self.layer_config.hidden_channels * 2,
            self.layer_config.hidden_channels,
        )

        # Output head
        self.output_head = OutputHead(
            self.layer_config.hidden_channels * 4,
            self.layer_config.hidden_channels,
            n_devices - 1,
            logits=True,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        data_task_embedding = self.data_task_layer(data)
        task_embeddings = data_task_embedding["tasks"]

        task_embeddings = self.task_task_layer(task_embeddings, data)

        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        data_batch = data["data"].batch if isinstance(data, Batch) else None
        device_batch = data["devices"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        device_embeddings = self.device_layer(data)

        task_counts = torch.clip(counts[0], min=1)
        data_counts = torch.clip(counts[1], min=1)

        task_pooling = torch.div(
            global_add_pool(task_embeddings, task_batch), task_counts
        )
        data_pooling = torch.div(
            global_add_pool(data_task_embedding["data"], data_batch), data_counts
        )
        device_pooling = global_mean_pool(device_embeddings, device_batch)

        # task_pooling = global_mean_pool(task_embeddings, task_batch)
        # data_pooling = global_mean_pool(data_task_embedding["data"], data_batch)
        # device_pooling = global_mean_pool(device_embeddings, device_batch)

        global_embedding = self.combine_layer(
            data_pooling, task_pooling, device_pooling
        )
        global_embedding = global_embedding.squeeze(0)

        # print("candidate_embedding", candidate_embedding.shape)
        # print("global_embedding", global_embedding.shape)

        candidate_embedding = torch.cat([candidate_embedding, global_embedding], dim=-1)

        # print("merged_shape", candidate_embedding.shape)

        d_logits = self.output_head(candidate_embedding)

        return d_logits


class ValueNetkLayer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
        task_task_layers: int = 1,
        data_task_layers: int = 1,
        skip_connection: bool = True,
    ):
        super(ValueNetkLayer, self).__init__()

        self.feature_config = feature_config
        self.layer_config = layer_config
        self.n_devices = n_devices

        if task_task_layers >= 1:
            self.data_task_layer = DataTaskGATkLayer(
                feature_config,
                layer_config,
                k=data_task_layers,
                skip_connection=skip_connection,
            )

        if task_task_layers >= 1:
            self.task_task_layer = TaskTaskGATkLayer(
                layer_config.hidden_channels,
                feature_config,
                layer_config,
                k=task_task_layers,
                skip_connection=skip_connection,
            )

        self.device_layer = DeviceGlobalLayer(feature_config, layer_config)

        # Combination layer
        self.combine_layer = CombineThreeLayer(
            self.layer_config.hidden_channels,
            self.layer_config.hidden_channels * 2,
            self.layer_config.hidden_channels,
            self.layer_config.hidden_channels * 2,
            self.layer_config.hidden_channels,
        )

        # Output head
        self.output_head = OutputHead(
            self.layer_config.hidden_channels * 4,
            self.layer_config.hidden_channels,
            1,
            logits=False,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        data_task_embedding = self.data_task_layer(data)
        task_embeddings = data_task_embedding["tasks"]

        task_embeddings = self.task_task_layer(task_embeddings, data)

        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        data_batch = data["data"].batch if isinstance(data, Batch) else None
        device_batch = data["devices"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        device_embeddings = self.device_layer(data)

        task_counts = torch.clip(counts[0], min=1)
        data_counts = torch.clip(counts[1], min=1)

        task_pooling = torch.div(
            global_add_pool(task_embeddings, task_batch), task_counts
        )
        counts[1] = torch.clip(counts[1], min=1)
        data_pooling = torch.div(
            global_add_pool(data_task_embedding["data"], data_batch), data_counts
        )
        device_pooling = global_mean_pool(device_embeddings, device_batch)

        # task_pooling = global_mean_pool(task_embeddings, task_batch)
        # data_pooling = global_mean_pool(data_task_embedding["data"], data_batch)
        # device_pooling = global_mean_pool(device_embeddings, device_batch)

        global_embedding = self.combine_layer(
            data_pooling, task_pooling, device_pooling
        )
        global_embedding = global_embedding.squeeze(0)

        # print("candidate_embedding", candidate_embedding.shape)
        # print("global_embedding", global_embedding.shape)

        candidate_embedding = torch.cat([candidate_embedding, global_embedding], dim=-1)

        # print("merged_shape", candidate_embedding.shape)

        v = self.output_head(candidate_embedding)

        return v


class SeparateNet2Layer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
    ):
        super(SeparateNet2Layer, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config
        self.n_devices = n_devices

        self.actor = DeviceAssignmentNet2Layer(feature_config, layer_config, n_devices)
        self.critic = ValueNetkLayer(feature_config, layer_config, n_devices)

    def forward(self, data: HeteroData | Batch, counts=None):
        d_logits = self.actor(data, counts)
        v = self.critic(data, counts)
        return d_logits, v


class OldCombinedNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldCombinedNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_gat = HeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = self.hetero_gat.output_dim

        self.actor = OutputHead(
            gat_output_dim, layer_config.hidden_channels, n_devices, logits=True
        )
        self.critic = OutputHead(
            gat_output_dim, layer_config.hidden_channels, 1, logits=False
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        if next(self.parameters()).is_cuda:
            data = data.to("cuda")
        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        d_logits = self.actor(candidate_embedding)

        v = self.critic(task_embeddings)
        v = global_add_pool(v, task_batch)
        v = torch.div(v, counts[0])

        return d_logits, v


class UnRolledDeviceLayer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(UnRolledDeviceLayer, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config
        self.n_devices = n_devices
        self.output_dim = n_devices * feature_config.device_feature_dim

    def forward(self, data: HeteroData | Batch):
        # Device features are n_devices x device_feature_dim

        device_features = data["devices"].x
        device_features = device_features.reshape(
            -1, self.n_devices * self.feature_config.device_feature_dim
        )

        return device_features


class DataTaskGraphConv(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(DataTaskGraphConv, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.data_task_conv = GraphConv(
            (feature_config.data_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            aggr="add",
            bias=True,
        )

        self.layer_norm = nn.LayerNorm(layer_config.hidden_channels)

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = layer_config.hidden_channels

    def forward(self, data: HeteroData | Batch):
        data_task_edges = data["data", "to", "tasks"].edge_index

        task_agg = self.data_task_conv(
            (data["data"].x, data["tasks"].x), data_task_edges
        )

        task_agg = self.layer_norm(task_agg)

        task_agg = self.activation(task_agg)

        return task_agg


class TaskTaskEdgeConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_config: LayerConfig,
        k: int = 1,
        agg_type: str = "add",
        skip_connection: bool = False,
    ):
        super(TaskTaskEdgeConv, self).__init__()
        if k < 1:
            raise ValueError("Number of layers k must be at least 1.")

        self.skip_connection = skip_connection
        self.k = k
        self.layer_config = layer_config
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        current_dim = input_dim
        for i in range(k):
            mlp = nn.Sequential(
                layer_init(nn.Linear(current_dim * 2, layer_config.hidden_channels)),
                nn.LeakyReLU(negative_slope=0.01),
                layer_init(
                    nn.Linear(
                        layer_config.hidden_channels, layer_config.hidden_channels
                    )
                ),
            )
            self.conv_layers.append(EdgeConv(mlp, aggr=agg_type))
            self.norm_layers.append(nn.LayerNorm(layer_config.hidden_channels))
            current_dim = layer_config.hidden_channels  # Input dim for the next layer

        self.output_dim = k * layer_config.hidden_channels

    def forward(self, task_features, task_edges):
        """
        Forward pass through the k EdgeConv layers.

        Args:
            task_features (Tensor): Input task node features [num_tasks, input_dim].
            task_edges (Tensor): Edge index for task-to-task connections [2, num_edges].

        Returns:
            Tensor: Concatenated output features from all layers
                    [num_tasks, k * hidden_channels].
        """
        layer_outputs = []
        current_features = task_features

        for i in range(self.k):
            current_features = self.conv_layers[i](current_features, task_edges)
            current_features = self.norm_layers[i](current_features)
            current_features = self.activation(current_features)
            if self.skip_connection:
                layer_outputs.append(current_features.clone())

        # Concatenate the outputs from all layers
        if self.skip_connection:
            final_output = torch.cat(layer_outputs, dim=-1)
        else:
            final_output = current_features

        return final_output


class HeteroConvkLayer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
        k: int = 1,
    ):
        super(HeteroConvkLayer, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        if k < 1:
            raise ValueError("Number of layers k must be at least 1.")

        self.k = k

        self.hetero_convs = nn.ModuleList()

        self.hetero_convs.append(
            HeteroConv(
                {
                    ("data", "to", "tasks"): GraphConv(
                        (
                            feature_config.data_feature_dim,
                            feature_config.task_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    ("tasks", "to", "data"): GraphConv(
                        (
                            feature_config.task_feature_dim,
                            feature_config.data_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    # ("data", "writes", "tasks"): GraphConv(
                    #     (
                    #         feature_config.data_feature_dim,
                    #         feature_config.task_feature_dim,
                    #     ),
                    #     layer_config.hidden_channels,
                    #     aggr="add",
                    # ),
                    # ("tasks", "writes", "data"): GraphConv(
                    #     (
                    #         feature_config.task_feature_dim,
                    #         feature_config.data_feature_dim,
                    #     ),
                    #     layer_config.hidden_channels,
                    #     aggr="add",
                    # ),
                    ("tasks", "to", "devices"): GraphConv(
                        (
                            feature_config.task_feature_dim,
                            feature_config.device_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    ("devices", "to", "tasks"): GraphConv(
                        (
                            feature_config.device_feature_dim,
                            feature_config.task_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    ("data", "to", "devices"): GraphConv(
                        (
                            feature_config.data_feature_dim,
                            feature_config.device_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    ("devices", "to", "data"): GraphConv(
                        (
                            feature_config.device_feature_dim,
                            feature_config.data_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    ("tasks", "to", "tasks"): GraphConv(
                        (
                            feature_config.task_feature_dim,
                            feature_config.task_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                    ("tasks", "from", "tasks"): GraphConv(
                        (
                            feature_config.task_feature_dim,
                            feature_config.task_feature_dim,
                        ),
                        layer_config.hidden_channels,
                        aggr="add",
                    ),
                },
                aggr="sum",
            )
        )

        for i in range(1, k):
            self.hetero_convs.append(
                HeteroConv(
                    {
                        ("data", "mapped", "tasks"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("tasks", "mapped", "data"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("tasks", "to", "devices"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("devices", "to", "tasks"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("data", "to", "devices"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("devices", "to", "data"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("tasks", "to", "tasks"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                        ("tasks", "from", "tasks"): GraphConv(
                            (
                                layer_config.hidden_channels,
                                layer_config.hidden_channels,
                            ),
                            layer_config.hidden_channels,
                            aggr="add",
                        ),
                    },
                    aggr="sum",
                )
            )
        self.device_layer_norm = nn.ModuleList()
        self.task_layer_norm = nn.ModuleList()
        self.data_layer_norm = nn.ModuleList()

        for i in range(k):
            self.device_layer_norm.append(nn.LayerNorm(layer_config.hidden_channels))
            self.task_layer_norm.append(nn.LayerNorm(layer_config.hidden_channels))
            self.data_layer_norm.append(nn.LayerNorm(layer_config.hidden_channels))

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = layer_config.hidden_channels

    def forward(self, data: HeteroData | Batch):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        # print("HeteroConvkLayer")
        # print("data", data)
        # print("hetero_convs", self.hetero_convs)

        for k in range(self.k):
            x_dict = self.hetero_convs[k](x_dict, edge_index_dict)

            for node_type in x_dict.keys():
                if node_type == "tasks":
                    x_dict[node_type] = self.task_layer_norm[k](x_dict[node_type])
                    x_dict[node_type] = self.activation(x_dict[node_type])
                elif node_type == "data":
                    x_dict[node_type] = self.data_layer_norm[k](x_dict[node_type])
                    x_dict[node_type] = self.activation(x_dict[node_type])
                elif node_type == "devices":
                    x_dict[node_type] = self.device_layer_norm[k](x_dict[node_type])
                    x_dict[node_type] = self.activation(x_dict[node_type])

        # print("data_out", x_dict)

        return x_dict


class HeteroConvStateNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
        k: int = 1,
    ):
        super(HeteroConvStateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_conv = HeteroConvkLayer(
            feature_config, layer_config, n_devices, k=k
        )

        self.output_dim = layer_config.hidden_channels * 2

    def forward(self, data: HeteroData | Batch, counts=None):
        # print("HeteroConvStateNet")

        features = self.hetero_conv(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        data_batch = data["data"].batch if isinstance(data, Batch) else None
        device_batch = data["devices"].batch if isinstance(data, Batch) else None

        time = data["time"].x
        with torch.no_grad():
            time = time / 100000
        if task_batch is None:
            time = time.squeeze(0)
        else:
            time.reshape(-1, 1)

        task_features = features["tasks"]
        data_features = features["data"]
        device_features = features["devices"]

        # print("task_features", task_features.shape)
        # print("data_features", data_features.shape)
        # print("device_features", device_features.shape)
        # print("time", time.shape)

        if task_batch is not None:
            candidate_features = task_features[data["tasks"].ptr[:-1]]
        else:
            candidate_features = task_features[0]

        task_counts = torch.clip(data["tasks_count"].x.unsqueeze(1), min=1)
        # data_counts = torch.clip(counts[1], min=1)

        # print("task_counts", task_counts.shape)

        # print("task_features", task_features.shape)
        task_features = global_add_pool(task_features, task_batch)
        # print("task_features", task_features.shape)

        task_pooling = torch.div(task_features, task_counts)

        # print("task_pooling", task_pooling.shape)

        # data_pooling = torch.div(
        #     global_add_pool(data_features, data_batch), data_counts
        # )

        device_pooling = global_mean_pool(device_features, device_batch)

        # print("task_pooling", task_pooling.shape)
        # print("data_pooling", data_pooling.shape)
        # print("device_pooling", device_pooling.shape)

        task_pooling = task_pooling.squeeze(0)
        # data_pooling = data_pooling.squeeze(0)
        device_pooling = device_pooling.squeeze(0)

        pool_all = task_pooling + device_pooling
        # pool_all = pool_all.squeeze(0)

        # print("pool_all", pool_all.shape)
        # print("candidate_features", candidate_features.shape)

        state_features = torch.cat([candidate_features, pool_all], dim=-1)

        return state_features


class AddConvStateNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(AddConvStateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.data_task_conv = DataTaskGraphConv(feature_config, layer_config, n_devices)
        data_task_dim = self.data_task_conv.output_dim

        self.task_task_conv_dependants = TaskTaskEdgeConv(
            data_task_dim, layer_config, k=1, agg_type="add"
        )

        self.task_task_conv_dependencies = TaskTaskEdgeConv(
            data_task_dim, layer_config, k=1, agg_type="add"
        )

        self.unroll_devices = UnRolledDeviceLayer(
            feature_config, layer_config, n_devices
        )

        # MLP that turns (n_devices * device_feature_dim) into [1 x hidden_channels]
        self.device_layer = nn.Sequential(
            layer_init(
                nn.Linear(
                    feature_config.device_feature_dim * n_devices + 1,
                    layer_config.hidden_channels,
                )
            ),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.stacked_task_dim = (
            self.task_task_conv_dependants.output_dim
            + self.task_task_conv_dependencies.output_dim
        )

        self.project_down = nn.Sequential(
            layer_init(nn.Linear(self.stacked_task_dim, layer_config.hidden_channels)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.output_dim = layer_config.hidden_channels * 3

    def forward(self, data: HeteroData | Batch, counts=None):
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        data_fused_tasks = self.data_task_conv(data)
        task_task_dependants = self.task_task_conv_dependants(
            data_fused_tasks, data["tasks", "to", "tasks"].edge_index
        )
        task_task_dependencies = self.task_task_conv_dependencies(
            data_fused_tasks, data["tasks", "to", "tasks"].edge_index.flip(0)
        )

        device_features = self.unroll_devices(data)
        device_features = device_features.squeeze(0)

        time = data["time"].x

        with torch.no_grad():
            time = time / 100000

        if task_batch is None:
            time = time.squeeze(0)
        else:
            time.reshape(-1, 1)

        device_features = torch.cat([device_features, time], dim=-1)

        device_features = self.device_layer(device_features)

        task_features = torch.cat(
            [
                task_task_dependants,
                task_task_dependencies,
            ],
            dim=-1,
        )

        task_features = self.project_down(task_features)

        if task_batch is not None:
            candidate_features = task_features[data["tasks"].ptr[:-1]]
        else:
            candidate_features = task_features[0]

        counts_0 = torch.clip(counts[0], min=1)
        global_state = global_add_pool(task_features, task_batch)
        global_state = torch.div(global_state, counts_0)
        global_state = global_state.squeeze(0)

        # print("global_state", global_state.shape)
        # print("device_features", device_features.shape)
        # print("candidate_features", candidate_features.shape)

        # global_state = global_state.unsqueeze(0)
        # candidate_features = candidate_features.unsqueeze(0)
        # device_features = device_features.unsqueeze(0)

        # print("global_state", global_state.shape)
        # print("device_features", device_features.shape)
        # print("candidate_features", candidate_features.shape)

        state_features = torch.cat(
            (global_state, candidate_features, device_features), dim=-1
        )

        # print("state_features", state_features)

        # print("state_features", state_features.shape)

        return state_features


class DataTaskPolicyNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(DataTaskPolicyNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.data_task_gat = DataTaskGATkLayer(feature_config, layer_config)
        data_task_dim = self.data_task_gat.output_dim

        self.task_task_gat = TaskTaskGATkLayer(
            data_task_dim, feature_config, layer_config
        )

        self.unrolled_device_layer = UnRolledDeviceLayer(
            feature_config, layer_config, n_devices
        )

        output_dim = (
            self.task_task_gat.output_dim * 2 + self.unrolled_device_layer.output_dim
        )

        self.actor_head = OutputHead(
            output_dim, layer_config.hidden_channels, n_devices - 1, logits=True
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.data_task_gat(data)
        task_embeddings = self.task_task_gat(task_embeddings, data)

        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        device_batch = data["devices"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        device_features = self.unrolled_device_layer(data)
        device_features = device_features.squeeze(0)

        counts_0 = torch.clip(counts[0], min=1)
        global_embedding = global_add_pool(task_embeddings, task_batch)
        global_embedding = torch.div(global_embedding, counts_0)

        global_embedding = global_embedding.squeeze(0)

        time = data["auxilary"]["time"]
        time = time.reshape(-1, 1)

        candidate_embedding = torch.cat(
            [candidate_embedding, global_embedding, device_features, time], dim=-1
        )

        d_logits = self.actor_head(candidate_embedding)

        return d_logits


class HeteroConvPolicyNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
        k: int = 1,
    ):
        super(HeteroConvPolicyNet, self).__init__()

        self.heteroconv_state_net = HeteroConvStateNet(
            feature_config, layer_config, n_devices, k=k
        )

        self.output_head = OutputHead(
            self.heteroconv_state_net.output_dim,
            layer_config.hidden_channels,
            n_devices - 1,
            logits=True,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        # print("HeteroConvPolicyNet")
        state_features = self.heteroconv_state_net(data, counts)
        # print("state_features", state_features.shape)
        d_logits = self.output_head(state_features)
        # print("d_logits", d_logits.shape)
        return d_logits


class HeteroConvValueNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
        k: int = 1,
    ):
        super(HeteroConvValueNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.heteroconv_state_net = HeteroConvStateNet(
            feature_config, layer_config, n_devices, k=k
        )

        self.output_head = OutputHead(
            self.heteroconv_state_net.output_dim,
            layer_config.hidden_channels,
            1,
            logits=False,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        # print("HeteroConvValueNet")
        state_features = self.heteroconv_state_net(data, counts)
        v = self.output_head(state_features)
        return v


class AddConvPolicyNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(AddConvPolicyNet, self).__init__()

        self.add_conv_state_net = AddConvStateNet(
            feature_config, layer_config, n_devices
        )

        self.output_head = OutputHead(
            self.add_conv_state_net.output_dim,
            layer_config.hidden_channels,
            n_devices - 1,
            logits=True,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        state_features = self.add_conv_state_net(data, counts)
        d_logits = self.output_head(state_features)
        return d_logits


class AddConvValueNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(AddConvValueNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.add_conv_state_net = AddConvStateNet(
            feature_config, layer_config, n_devices
        )

        self.output_head = OutputHead(
            self.add_conv_state_net.output_dim,
            layer_config.hidden_channels,
            1,
            logits=False,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        state_features = self.add_conv_state_net(data, counts)
        v = self.output_head(state_features)
        return v


class DataTaskValueNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(DataTaskValueNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.data_task_gat = DataTaskGATkLayer(feature_config, layer_config)
        data_task_dim = self.data_task_gat.output_dim

        self.task_task_gat = TaskTaskGATkLayer(
            data_task_dim, feature_config, layer_config
        )

        self.unrolled_device_layer = UnRolledDeviceLayer(
            feature_config, layer_config, n_devices
        )

        output_dim = (
            self.task_task_gat.output_dim * 2 + self.unrolled_device_layer.output_dim
        )

        self.critic_head = OutputHead(
            output_dim, layer_config.hidden_channels, 1, logits=False
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.data_task_gat(data)
        task_embeddings = self.task_task_gat(task_embeddings, data)

        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        device_batch = data["devices"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        device_features = self.unrolled_device_layer(data)
        device_features = device_features.squeeze(0)

        counts_0 = torch.clip(counts[0], min=1)
        global_embedding = global_add_pool(task_embeddings, task_batch)
        global_embedding = torch.div(global_embedding, counts_0)

        global_embedding = global_embedding.squeeze(0)

        candidate_embedding = torch.cat(
            [candidate_embedding, global_embedding, device_features], dim=-1
        )

        v = self.critic_head(candidate_embedding)

        return v


class DataTaskSeparateNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
    ):
        super(DataTaskSeparateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.actor = DataTaskPolicyNet(feature_config, layer_config, n_devices)
        self.critic = DataTaskValueNet(feature_config, layer_config, n_devices)

    def forward(self, data: HeteroData | Batch, counts=None):
        d_logits = self.actor(data, counts)
        v = self.critic(data, counts)
        return d_logits, v


class OldTaskAssignmentNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldTaskAssignmentNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_gat = NoDeviceHeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = self.hetero_gat.output_dim
        self.unrolled_device_layer = UnRolledDeviceLayer(
            feature_config, layer_config, n_devices
        )

        self.actor_head = OutputHead(
            gat_output_dim * 2 + self.unrolled_device_layer.output_dim,
            layer_config.hidden_channels,
            n_devices - 1,
            logits=True,
        )
        print(f"actor output dim: {n_devices - 1}")

    def forward(self, data: HeteroData | Batch, counts=None):
        # if next(self.parameters()).is_cuda:
        #     data = data.to("cuda")

        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        device_features = self.unrolled_device_layer(data)
        device_features = device_features.squeeze(0)

        global_embedding = global_add_pool(task_embeddings, task_batch)
        counts_0 = torch.clip(counts[0], min=1)
        global_embedding = torch.div(global_embedding, counts_0)
        global_embedding = global_embedding.squeeze(0)

        candidate_embedding = torch.cat(
            [candidate_embedding, global_embedding, device_features], dim=-1
        )

        d_logits = self.actor_head(candidate_embedding)

        return d_logits


class OldValueNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldValueNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_gat = NoDeviceHeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = self.hetero_gat.output_dim

        self.unrolled_device_layer = UnRolledDeviceLayer(
            feature_config, layer_config, n_devices
        )

        self.critic_head = OutputHead(
            gat_output_dim + self.unrolled_device_layer.output_dim,
            layer_config.hidden_channels,
            1,
            logits=False,
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        counts_0 = torch.clip(counts[0], min=1)

        device_features = self.unrolled_device_layer(data)
        device_features = device_features.squeeze(0)

        global_embedding = global_add_pool(task_embeddings, task_batch)
        global_embedding = torch.div(global_embedding, counts_0)

        global_embedding = global_embedding.squeeze(0)

        global_embedding = torch.cat([global_embedding, device_features], dim=-1)

        v = self.critic_head(global_embedding)
        # v = torch.cat([v, device_features], dim=-1)

        # v = torch.div(global_add_pool(v, task_batch), counts_0)

        return v


class OldActionValueNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldActionValueNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_gat = HeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = self.hetero_gat.output_dim

        self.critic_head = OutputHead(
            gat_output_dim, layer_config.hidden_channels, n_devices - 1, logits=True
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        v = global_add_pool(task_embeddings, task_batch)
        v = torch.div(v, counts[0])
        v = self.critic_head(v)
        return v


class OldSeparateNet(nn.Module):
    """
    Wrapper module for separate actor and critic networks using individual HeteroGAT1Layer instances.

    Unlike `OldCombinedNet`, this class assigns a distinct HeteroGAT1Layer to each of the actor and critic networks.

    Args:
        n_devices (int): The number of mappable devices. Check whether this includes the CPU.
    """

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldSeparateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.actor = OldTaskAssignmentNet(feature_config, layer_config, n_devices)
        self.critic = OldValueNet(feature_config, layer_config, n_devices)

    def forward(self, data: HeteroData | Batch, counts=None):
        # check the device of data["tasks"].x
        if next(self.actor.parameters()).is_cuda:
            data = data.to("cuda")
        d_logits = self.actor(data, counts)
        v = self.critic(data, counts)
        return d_logits, v


class OldTaskAssignmentNetwDevice(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldTaskAssignmentNetwDevice, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_gat = HeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )
        self.actor_head = OldOutputHead(
            gat_output_dim, layer_config.hidden_channels, n_devices - 1, False
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        if next(self.parameters()).is_cuda:
            data = data.to("cuda")

        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        d_logits = self.actor_head(candidate_embedding)

        return d_logits


class OldValueNetwDevice(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldValueNetwDevice, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.hetero_gat = HeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )
        self.critic_head = OldOutputHead(
            gat_output_dim, layer_config.hidden_channels, 1, logits=False
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        counts = None
        if counts is None:
            v = self.critic_head(task_embeddings)
            v = global_mean_pool(v, task_batch)
        else:
            v = self.critic_head(task_embeddings)
            v = global_add_pool(v, task_batch)
            v = torch.div(v, torch.clamp(counts[0], min=1))

        return v


class OldSeparateNetwDevice(nn.Module):
    """
    Wrapper module for separate actor and critic networks using individual HeteroGAT1Layer instances.

    Unlike `OldCombinedNet`, this class assigns a distinct HeteroGAT1Layer to each of the actor and critic networks.

    Args:
        n_devices (int): The number of mappable devices. Check whether this includes the CPU.
    """

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(OldSeparateNetwDevice, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.actor = OldTaskAssignmentNetwDevice(
            feature_config, layer_config, n_devices
        )
        self.critic = OldValueNetwDevice(feature_config, layer_config, n_devices)

    def forward(self, data: HeteroData | Batch, counts=None):
        d_logits = self.actor(data, counts)
        v = self.critic(data, counts)
        return d_logits, v


class AddConvSeparateNet(nn.Module):
    """
    Wrapper module for separate actor and critic networks using individual HeteroGAT1Layer instances.

    Unlike `OldCombinedNet`, this class assigns a distinct HeteroGAT1Layer to each of the actor and critic networks.

    Args:
        n_devices (int): The number of mappable devices. Check whether this includes the CPU.
    """

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(AddConvSeparateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.actor = AddConvPolicyNet(feature_config, layer_config, n_devices)
        self.critic = AddConvValueNet(feature_config, layer_config, n_devices)

    def forward(self, data: HeteroData | Batch, counts=None):
        d_logits = self.actor(data, counts)
        v = self.critic(data, counts)
        return d_logits, v


class HeteroConvSeparateNet(nn.Module):
    """
    Wrapper module for separate actor and critic networks using individual HeteroGAT1Layer instances.

    Unlike `OldCombinedNet`, this class assigns a distinct HeteroGAT1Layer to each of the actor and critic networks.

    Args:
        n_devices (int): The number of mappable devices. Check whether this includes the CPU.
    """

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
        k: int = 2,
    ):
        super(HeteroConvSeparateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.actor = HeteroConvPolicyNet(feature_config, layer_config, n_devices, k=k)
        self.critic = HeteroConvValueNet(feature_config, layer_config, n_devices, k=k)

    def forward(self, data: HeteroData | Batch, counts=None):
        # if any(p.is_cuda for p in self.actor.parameters()):
        #     data = data.to("cuda", non_blocking=True)
        if not isinstance(data, Batch | HeteroData):
            data = data["hetero_data"]
        d_logits = self.actor(data, counts)
        v = self.critic(data, counts)
        return d_logits, v


class VectorStateNet(nn.Module):
    """
    Simple network that takes in a task feature vector and performs k MLP layers of fixed size.
    Args:
        feature_config (FeatureDimConfig): Configuration for feature dimensions.
        layer_config (LayerConfig): Configuration for layer dimensions.
    """

    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: list[int] | int,
        add_progress: bool = False,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = True,
    ):
        super(VectorStateNet, self).__init__()
        self.feature_config = feature_config
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        self.hidden_channels = hidden_channels
        self.k = len(self.hidden_channels)


        def make_activation(activation_config):
            return (
                instantiate(activation)
                if activation
                else nn.LeakyReLU(negative_slope=0.01)
            )

        layer_init = call(initialization if initialization else kaiming_init)

        input_dim = feature_config.task_feature_dim

        self.add_progress = add_progress
        if add_progress:
            input_dim += 2

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
        task_features = torch.squeeze(task_features)
        # print("TF SHAPE", task_features.shape)

        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]
            task_features = torch.cat(
                [task_features, time_feature, progress_feature], dim=-1
            )

        task_activations = self.layers(task_features)

        return task_activations


class VectorPolicyNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: list[int],
        n_devices: int = 5,
    ):
        super(VectorPolicyNet, self).__init__()
        self.vector_state_net = VectorStateNet(feature_config, hidden_channels)

        self.actor_head = OutputHead(
            self.vector_state_net.output_dim,
            hidden_channels,
            n_devices - 1,
            logits=True,
        )

    def forward(self, td):
        task_features = td["nodes"]["tasks"]["attr"]
        # print("task_features", task_features.shape)
        state_features = self.vector_state_net(task_features)
        d_logits = self.actor_head(state_features)
        # print("d_logits", d_logits.shape)
        return d_logits


class VectorValueNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
        k: int = 0,
    ):
        super(VectorValueNet, self).__init__()
        self.vector_state_net = VectorStateNet(feature_config, layer_config, k=k)

        self.critic_head = OutputHead(
            self.vector_state_net.output_dim + 2,
            layer_config.hidden_channels,
            1,
            logits=False,
        )

    def forward(self, td):
        task_features = td["nodes"]["tasks"]["attr"]
        # print("task_features", task_features.shape)
        state_features = self.vector_state_net(task_features)

        time_feature = td["aux"]["time"] / td["aux"]["baseline"]

        # print(
        #     f"time: {td['aux']['time']}, baseline: {td['aux']['baseline']}, time_feature: {time_feature}"
        # )
        progress_feature = td["aux"]["progress"]
        state_features = state_features.squeeze(1)

        # print(f"time_feature: {time_feature}, progress_feature: {progress_feature}")
        # print(f"state_features: {state_features}")
        # state_features = state_features.squeeze(1)

        state_features = torch.cat(
            [state_features, time_feature, progress_feature], dim=-1
        )

        v = self.critic_head(state_features)
        # v = v.squeeze(1)
        # print("v", v.shape)
        return v


class VectorSeparateNet(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int,
    ):
        super(VectorSeparateNet, self).__init__()
        self.feature_config = feature_config
        self.layer_config = layer_config

        self.actor = VectorPolicyNet(feature_config, layer_config, n_devices, 0)
        self.critic = VectorValueNet(feature_config, layer_config, n_devices, 0)

    def forward(self, td: TensorDict):
        d_logits = self.actor(td)
        v = self.critic(td)
        return d_logits, v


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

        #TODO(wlr); Continue modifying this for rectangular domains

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
        self.output_dim = (
            ((self.width*self.length) * ch + 1) if self.add_progress else ((self.width*self.length) * ch)
        )
        # Initialize CNN weights
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
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
        x_flat = x_flat.view(
            flat_bs, width, length, in_channels
        ).permute(  # (flat_bs, W, L, C_in)
            0, 3, 1, 2
        )  # (flat_bs, C_in, W, L)

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
    """
    Compute the number of 2x downsamplings (H,W -> floor(H/2), floor(W/2)) such that
    we never reduce the MIN side below `minimum_resolution` on the *next* pool.
    Works for rectangular/non-power-of-two sizes.
    """
    h, w = int(length), int(width)
    layers = 0
    while min(h, w) >= 2 * minimum_resolution:
        h //= 2
        w //= 2
        layers += 1
    return layers


def _init_deconv_bilinear_(deconv: nn.ConvTranspose2d) -> None:
    """
    Initialize ConvTranspose2d as (per-channel) bilinear upsampling *only* when shapes allow:
      - groups == 1
      - in_channels == out_channels
      - kernel is square; stride == kernel_size
    Otherwise, do nothing (safer than partial/incorrect init).
    """
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
    """
    Center-align upsampled feature (B, C_up, H_up, W_up) to encoder skip (B, C_enc, H_enc, W_enc)
    by symmetric pad/crop, then concatenate along channels.

    This guards against odd/even floor effects from pooling on rectangles.
    """
    uh, uw = up_feat.shape[-2:]
    eh, ew = enc_feat.shape[-2:]
    dh, dw = eh - uh, ew - uw
    if dh > 0 or dw > 0:
        # F.pad order: (left, right, top, bottom)
        pad = [max(dw // 2, 0), max(dw - dw // 2, 0),
               max(dh // 2, 0), max(dh - dh // 2, 0)]
        up_feat = F.pad(up_feat, pad)
    elif dh < 0 or dw < 0:
        top = (-dh) // 2
        left = (-dw) // 2
        up_feat = up_feat[..., top: top + eh, left: left + ew]
    return torch.cat([up_feat, enc_feat], dim=1)


def _flatten_to_BCHW(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Accept (C,H,W) or (*batch, C, H, W). Return (B, C, H, W), batch_shape, B.
    """
    if x.dim() == 3:
        return x.unsqueeze(0), (), 1
    elif x.dim() >= 4:
        *batch, C, H, W = x.shape
        B = 1
        for d in batch: B *= int(d)
        return x.reshape(B, C, H, W), tuple(batch), B
    else:
        raise ValueError(f"Expected (C,H,W) or (*batch,C,H,W), got {tuple(x.shape)}")

def _unflatten_from_B(xB: torch.Tensor, batch_shape: Tuple[int, ...]) -> torch.Tensor:
    """Inverse of _flatten_to_BCHW for the *batch* part; keeps (C,H,W) tail intact."""
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
        for d in batch: B *= int(d)
        return x.reshape(B, P), tuple(batch), B
    else:
        raise ValueError(f"Expected (..., P), got {tuple(x.shape)}")

def _coord_mesh(H: int, W: int, device, dtype):
    """
    Normalized CoordConv channels in [-1,1], shape (2,H,W).
    x increases left->right, y increases top->bottom.
    """
    ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)  # (2,H,W)

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
    """
    Two 3x3 convs; the first uses a (possibly >1) dilation; the second uses dilation=1.
    Residual connection is identity because C_in==C_out here.
    """
    def __init__(self, C: int, dilation: int = 1, act="silu"):
        super().__init__()
        self.conv1 = ConvNormAct(C, C, k=3, dilation=dilation, act=act)
        self.conv2 = ConvNormAct(C, C, k=3, dilation=1, act=act)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class ECA(nn.Module):
    """
    Efficient Channel Attention: global avg pool -> 1D conv (k odd) -> sigmoid gate.
    Gives cheap global channel-wise calibration without spatial resampling.
    """
    def __init__(self, C: int, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.pool(x)                             # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)           # (B,1,C)
        y = self.conv(y)                            # (B,1,C)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return x * y

def _choose_gn_groups(C: int) -> int:
    """Pick a GroupNorm group count that divides C (prefer 8,4,2,1)."""
    for g in (8, 4, 2):
        if C % g == 0:
            return g
    return 1

class TinyASPP(nn.Module):
    """
    Minimal ASPP with rates {1,2,3}. All stride-1, same-padding.
    Concats parallel atrous feats and reduces back to C with a 1x1.
    """
    def __init__(self, C: int, rates=(1, 2, 3), act="silu"):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(_choose_gn_groups(C), C),
                nn.SiLU(inplace=False) if act == "silu" else nn.ReLU(inplace=False),
            ) for r in rates
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(rates) * C, C, kernel_size=1, bias=False),
            nn.GroupNorm(_choose_gn_groups(C), C),
            nn.SiLU(inplace=False) if act == "silu" else nn.ReLU(inplace=False),
        )
    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))

class ConvNormAct(nn.Module):
    def __init__(self, C_in, C_out, k=3, dilation=1, groups=1, act="silu"):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=k, padding=pad,
                              dilation=dilation, bias=False, groups=groups)
        self.norm = nn.GroupNorm(_choose_gn_groups(C_out), C_out)
        self.act = nn.SiLU(inplace=False) if act == "silu" else nn.ReLU(inplace=False)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    


# ----- SPADE-style conditioning with explicit dims ---------------------------

class SpatialModulator(nn.Module):
    """
    SPADE-style: z_spa -> (gamma_xy, beta_xy) in R^{B×C×H×W}.
    Identity at init (near 0) with nonzero grads; runtime strength knobs.
    """
    def __init__(self, C: int, H: int, W: int,
                 z_spa_dim: int,
                 ch_hidden: int = 128,
                 seed_hw: Optional[Tuple[int,int]] = None,
                 init_scale_gamma_xy: float = 0.5,
                 init_scale_beta_xy:  float = 0.5):
        super().__init__()
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.h0 = max(4, H//4) if not seed_hw else seed_hw[0]
        self.w0 = max(4, W//4) if not seed_hw else seed_hw[1]

        self.to_seed = nn.Sequential(
            nn.Linear(z_spa_dim, ch_hidden), nn.SiLU(),
            nn.Linear(ch_hidden, 2*C*self.h0*self.w0)
        )
        nn.init.normal_(self.to_seed[-1].weight, std=1e-4)
        nn.init.zeros_(self.to_seed[-1].bias)

        # keep as non-trainable parameters for backwards compat; buffers are also fine
        self.scale_gamma_xy = nn.Parameter(torch.tensor(float(init_scale_gamma_xy)), requires_grad=False)
        self.scale_beta_xy  = nn.Parameter(torch.tensor(float(init_scale_beta_xy)),  requires_grad=False)

    @torch.no_grad()
    def set_strength(self, gamma_xy: Optional[float]=None, beta_xy: Optional[float]=None):
        if gamma_xy is not None: self.scale_gamma_xy.fill_(float(gamma_xy))
        if beta_xy  is not None: self.scale_beta_xy.fill_(float(beta_xy))

    def forward(self, z) -> Tuple[torch.Tensor, torch.Tensor]:   # z: (B, z_spa_dim)

        *lead, z_dim = z.shape
        B = int(torch.prod(torch.tensor(lead))) if lead else z.shape[0]
        zf = z.reshape(-1, z_dim)
        seed = self.to_seed(zf).view(B, 2*self.C, self.h0, self.w0)
        maps = F.interpolate(seed, size=(self.H, self.W), mode='bilinear', align_corners=False)
        g_raw, b_raw = maps.chunk(2, dim=1)                      # (B,C,H,W)
        g_xy = self.scale_gamma_xy * torch.tanh(g_raw)           # bounded, ≈0
        b_xy = self.scale_beta_xy  * b_raw                       # small bias
        return g_xy, b_xy

class AdaSPADE_GN(nn.Module):
    """
    GroupNorm (affine=False) + channel FiLM(z_ch) + (optional) spatial FiLM(z_spa).
    Dimensions explicit via z_ch_dim.
    """
    def __init__(self, C: int, groups: int, spatial: SpatialModulator,
                 z_ch_dim: int,
                 ch_hidden: int = 128,
                 init_scale_gamma_c: float = 0.5, init_scale_beta_c: float = 0.5,
                 enable_spatial: bool = True, enable_channel: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(groups, C, affine=False)
        self.to_gb_c = nn.Sequential(
            nn.Linear(z_ch_dim, ch_hidden), nn.SiLU(),
            nn.Linear(ch_hidden, 2*C)
        )
        nn.init.normal_(self.to_gb_c[-1].weight, std=1e-4)
        nn.init.zeros_(self.to_gb_c[-1].bias)

        self.scale_gamma_c = nn.Parameter(torch.tensor(init_scale_gamma_c), requires_grad=False)
        self.scale_beta_c  = nn.Parameter(torch.tensor(init_scale_beta_c),  requires_grad=False)

        self.spatial = spatial
        self.enable_spatial = bool(enable_spatial)
        self.enable_channel = bool(enable_channel)

    @torch.no_grad()
    def set_strength(self, gamma_c: Optional[float]=None, beta_c: Optional[float]=None):
        if gamma_c is not None: self.scale_gamma_c.fill_(float(gamma_c))
        if beta_c  is not None: self.scale_beta_c.fill_(float(beta_c))

    def forward(self, x: torch.Tensor, z_ch: torch.Tensor, z_spa: torch.Tensor) -> torch.Tensor:
        # z_ch: (B, z_ch_dim); z_spa: (B, z_spa_dim)
        B, Cx, H, W = x.shape
        z_ch = z_ch.reshape(B, -1)
        z_spa = z_spa.reshape(B, -1)

        x = self.gn(x)

        if not self.enable_channel:
            return x
        
        g_c_raw, b_c_raw = self.to_gb_c(z_ch).chunk(2, dim=-1)   # (B,C)
        g_c = 1.0 + self.scale_gamma_c * torch.tanh(g_c_raw)
        b_c =        self.scale_beta_c  * b_c_raw

        if self.enable_spatial:
            g_xy, b_xy = self.spatial(z_spa)                     # (B,C,H,W)
        else:
            B, C, H, W = x.shape
            g_xy = x.new_zeros((B,Cx,H,W)); b_xy = x.new_zeros((B,Cx,H,W))

        gamma = g_c.unsqueeze(-1).unsqueeze(-1) * (1.0 + g_xy)
        beta  = b_c.unsqueeze(-1).unsqueeze(-1) + b_xy
        return x * gamma + beta

class DilatedResBlock_SPADE(nn.Module):
    def __init__(self, C: int, dilation: int, norm1: AdaSPADE_GN, norm2: AdaSPADE_GN):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = norm1
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.norm2 = norm2
        self.act2  = nn.SiLU()
        nn.init.zeros_(self.conv2.weight)  # identity-at-init

    def forward(self, x, z_ch, z_spa):
        h = self.act1(self.norm1(self.conv1(x), z_ch, z_spa))
        h = self.act2(self.norm2(self.conv2(h), z_ch, z_spa))
        return x + h

# ----- Backbone with explicit z dims -----------------------------------------

class DilationState(nn.Module):
    """
    Fixed-resolution backbone to be used as `layers.state`.
    Produces `("embed")` = (B, C, H, W). `output_dim = C`.
    All dimensions are explicit, including z dims.
    """
    def __init__(self,
                 feature_config,
                 hidden_channels: int,
                 width: int,
                 length: int,
                 z_ch_dim: int = 8,
                 z_spa_dim: int = 8,
                 num_blocks: int = 3,
                 dilation_schedule: Optional[List[int]] = None,
                 use_tiny_aspp: bool = False,
                 use_eca: bool = False,
                 spatial_in_all_blocks: bool = False,
                 film_in_all_blocks: bool = False,
                 spatial_last_k: int = 1,
                 film_last_k: int = 2,
                 init_gamma_c: float = 0.05, init_beta_c: float = 0.05,
                 init_gamma_xy: float = 0.05, init_beta_xy: float = 0.05,
                 debug: bool = False,
                 add_progress: bool = False,
                 **_ignored):
        super().__init__()
        if not hasattr(feature_config, "task_feature_dim"):
            raise AttributeError("feature_config must have attribute 'task_feature_dim'")

        self.width  = int(width)
        self.length = int(length)
        self.in_channels     = int(feature_config.task_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.debug = bool(debug)
        self.output_dim  = self.hidden_channels
        self.output_keys = ["embed"]
        self.add_progress = bool(add_progress)

        C_in = self.in_channels
        C    = self.hidden_channels

        self.stem = ConvNormAct(C_in, C, k=3, dilation=1, act="silu")
        if not dilation_schedule:
            dilation_schedule = [1, 2, 3, 1]

        # Effective z dims after optional progress concatenation
        zc_eff = int(z_ch_dim)  + (1 if self.add_progress else 0)
        zs_eff = int(z_spa_dim) + (1 if self.add_progress else 0)

        self.spatial = SpatialModulator(
            C=C, H=self.length, W=self.width,
            z_spa_dim=zs_eff, ch_hidden=16,
            seed_hw=(max(4,self.length//4), max(4,self.width//4)),
            init_scale_gamma_xy=init_gamma_xy, init_scale_beta_xy=init_beta_xy
        )

        groups = _choose_gn_groups(C)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            use_spa = spatial_in_all_blocks or (i >= num_blocks - spatial_last_k)
            use_film = film_in_all_blocks or (i >= num_blocks - film_last_k)

            norm1 = AdaSPADE_GN(
                C=C, groups=groups, spatial=self.spatial,
                z_ch_dim=zc_eff, ch_hidden=16,
                init_scale_gamma_c=init_gamma_c, init_scale_beta_c=init_beta_c,
                enable_spatial=use_spa,
                enable_channel=use_film
            )
            norm2 = AdaSPADE_GN(
                C=C, groups=groups, spatial=self.spatial,
                z_ch_dim=zc_eff, ch_hidden=16,
                init_scale_gamma_c=init_gamma_c, init_scale_beta_c=init_beta_c,
                enable_spatial=use_spa,
                enable_channel=use_film
            )

            self.blocks.append(
                DilatedResBlock_SPADE(
                    C, dilation=dilation_schedule[i % len(dilation_schedule)],
                    norm1=norm1, norm2=norm2
                )
            )

        self.aspp = TinyASPP(C) if use_tiny_aspp else nn.Identity()
        self.eca  = ECA(C, k_size=3) if use_eca else nn.Identity()

    @torch.no_grad()
    def set_noise_strength(self, gamma_c=None, beta_c=None, gamma_xy=None, beta_xy=None):
        if gamma_xy is not None or beta_xy is not None:
            self.spatial.set_strength(gamma_xy, beta_xy)
        for blk in self.blocks:
            blk.norm1.set_strength(gamma_c, beta_c)
            blk.norm2.set_strength(gamma_c, beta_c)

    def forward(self, observation):
        """
        Expects a TensorDict-like `observation` with:
          ('nodes','tasks','attr'): (..., H*W, Cin)
          ('aux','z_ch'): (..., z_ch_dim[+1 if add_progress])
          ('aux','z_spa'): (..., z_spa_dim[+1 if add_progress])
          optionally ('aux','progress'): (...,1) if add_progress
        Returns: (embed,)
        """
        xt   = observation[("nodes","tasks","attr")]   # (..., T, Cin)
        z_ch = observation[("aux","z_ch")]             # (..., z_ch_dim or z_ch_dim+1)
        z_spa= observation[("aux","z_spa")]            # (..., z_spa_dim or z_spa_dim+1)

        if self.add_progress:
            progress = observation[("aux","progress")] # (..., 1)
            z_ch  = torch.cat([z_ch,  progress], dim=-1)
            z_spa = torch.cat([z_spa, progress], dim=-1)

        single = (xt.dim() == 2)
        if single: xt = xt.unsqueeze(0)
        *batch_shape, T, Cin = xt.shape
        H, W = self.length, self.width
        assert T == H * W, f"tasks={T} differs from H*W={H*W}"
        assert Cin == self.in_channels

        B = 1
        for d in batch_shape: B *= int(d)
        h = xt.reshape(B, H, W, Cin).permute(0,3,1,2)  # (B,Cin,H,W)

        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h, z_ch, z_spa)
        h = self.aspp(h)
        h = self.eca(h)

        if single: h = h.squeeze(0)  # (C,H,W)
        else:
            C = h.size(1)
            h = h.view(*batch_shape, C, H, W)

        #print("DilationState output", h.shape)
        return (h,)

# ----- Policy head unchanged (no lazy) ---------------------------------------

class DilationPolicyHead(nn.Module):
    """
    Minimal actor head: ('embed')=(..., C, H, W) -> logits (…, H*W, A)
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 width: int,
                 length: int,
                 init_mode: str = "tiny",     # 'zero' | 'tiny' | 'kaiming'
                 tiny_std: float = 1e-3,
                 debug: bool = False,
                 **_ignored):
        super().__init__()
        self.width  = int(width)
        self.length = int(length)
        self.Cin    = int(input_dim)
        self.A      = int(output_dim)
        self.debug  = bool(debug)

        self.input_keys = ["embed"]
        self.output_dim = self.A

        self.proj = nn.Conv2d(self.Cin, self.A, kernel_size=1, bias=True)

        init_mode = init_mode.lower()
        if init_mode == "zero":
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        elif init_mode == "tiny":
            nn.init.normal_(self.proj.weight, std=float(tiny_std)); nn.init.zeros_(self.proj.bias)
        elif init_mode == "kaiming":
            nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear"); nn.init.zeros_(self.proj.bias)
        else:
            raise ValueError(f"init_mode must be 'zero' | 'tiny' | 'kaiming', got {init_mode!r}")

    def forward(self, embed: torch.Tensor):
        if embed.dim() == 3:
            h = embed.unsqueeze(0); single = True
        else:
            h = embed; single = False
        *B, C, H, W = h.shape
        h = embed.view(-1, C, H, W)
        logits_hw = self.proj(h)                         # (B, A, H, W)
        logits = logits_hw.permute(0, 2, 3, 1).reshape(h.size(0), H * W, self.A)
        logits = logits.view(*B, H*W, self.A)
        #print("DilationPolicyHead logits", logits.shape)
        return logits[0] if single else logits

# ----- Value head with explicit dims -----------------------------------------

class DilationValueHead(nn.Module):
    """
    embed: (B,C,H,W) or (C,H,W)
    z    : (B,z_dim)  or (z_dim,)
    returns: (B,) or scalar if single
    """
    def __init__(self,
                 input_dim: int,
                 z_dim: int = 8,
                 proj_dim: int = 8,
                 hidden_channels: int = 128,
                 tiny_std: float = 1e-3,
                 add_gap: bool = True, **_ignored):
        super().__init__()
        C = int(input_dim)
        P = int(proj_dim)
        Dz = int(z_dim)

        # 1×1 channel mixer (no bias before GN/act)
        self.mix = nn.Conv2d(C, P, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.mix.weight, nonlinearity='relu')

        # attention scorer -> (B,1,H,W); tiny init
        self.attn = nn.Conv2d(P, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.attn.weight, std=tiny_std)
        nn.init.zeros_(self.attn.bias)

        self.add_gap = bool(add_gap)

        mlp_in = (2 * P if self.add_gap else P) #+ Dz
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, 1)
        )
        nn.init.normal_(self.mlp[-1].weight, std=tiny_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, embed: torch.Tensor):
        # embed: (*batch, C, H, W) ; z: (*batch, Dz)
        *batch, C, H, W = embed.shape
        B = int(torch.tensor(batch).prod().item()) if batch else 1
        embed_f = embed.view(-1, C, H, W)                        # (B, C, H, W)
        #z_f = z.view(-1, z.shape[-1])                            # (B, Dz)

        Fm = F.silu(self.mix(embed_f))                           # (B, P, H, W)
        scores = self.attn(Fm)                                   # (B, 1, H, W)
        attn = scores.flatten(2).softmax(dim=-1).view(B, 1, H, W)
        pooled_attn = (Fm * attn).sum(dim=(2,3))                 # (B, P)
        if self.add_gap:
            pooled_gap = Fm.mean(dim=(2,3))                      # (B, P)
            pooled = torch.cat([pooled_attn, pooled_gap], dim=1) # (B, 2P)
        else:
            pooled = pooled_attn

        #v = self.mlp(torch.cat([pooled, z_f], dim=-1)).squeeze(-1)  # (B,)
        v = self.mlp(pooled).squeeze(-1)  # (B,)
        v = v.view(*batch)                                        # (*batch,)
        #print("DilationValueHead v", v.shape)
        return v

class DilationMultiValueHead(nn.Module):
    """ 
    Returns (B, H*W) or (H*W) if single values (one for each pixel)
    """

    def __init__(self,
                 input_dim: int,
                 tiny_std: float = 1e-3, **_ignored):
        super().__init__()
        C = int(input_dim)
        self.proj = nn.Conv2d(C, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.proj.weight, std=float(tiny_std))
        nn.init.zeros_(self.proj.bias)
        self.input_keys = ["embed"]
        self.output_dim = 1

    def forward(self, embed: torch.Tensor):
        if embed.dim() == 3:
            h = embed.unsqueeze(0); single = True
        else:
            h = embed; single = False
        *B, C, H, W = h.shape
        h = h.view(-1, C, H, W)
        v_hw = self.proj(h)                         # (B, 1, H, W)
        v = v_hw.squeeze(1)                         # (B, H, W)
        v = v.view(*B, H*W)
        print("DilationMultiValueHead v", v.shape)
        return v[0] if single else v


class DilationRectangularEncoder(nn.Module):

    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        width: int,
        length: int,
        add_progress: bool = False,                 
        minimum_resolution: int = 2,                
        activation: Optional[DictConfig] = None,     # kept for BC
        initialization: Optional[DictConfig] = None, # kept for BC
        debug: bool = True,
        pool_mode: str = "avg",                      # ignored; no pooling
        # New optional knobs (do not break existing calls):
        add_coord: bool = True,
        num_blocks: int = 2,
        dilation_schedule: Optional[List[int]] = None,
        use_tiny_aspp: bool = True,
        use_eca: bool = True,
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

        self.add_coord = bool(add_coord)
        C_in = self.in_channels + (2 if self.add_coord else 0)
        C = self.hidden_channels
        self.stem = ConvNormAct(C_in, C, k=3, dilation=1, act="silu")

        # Dilated residual stack
        if not dilation_schedule:
            dilation_schedule = [1, 2, 3]  # short, aperiodic cycle for ≤32x32
        self.blocks = nn.ModuleList([
            DilatedResBlock(C, dilation=dilation_schedule[i % len(dilation_schedule)], act="silu")
            for i in range(num_blocks)
        ])

        # Tiny ASPP and optional ECA gate
        self.aspp = TinyASPP(C) if use_tiny_aspp else nn.Identity()
        self.eca  = ECA(C, k_size=3) if use_eca else nn.Identity()

        self.film = nn.Linear(8, 2*self.hidden_channels, bias=True)

        self.in_channels_per_scale: List[int] = [C]
        self.output_dim = C
        self.output_keys: List[str] = ["embed"]

    def forward(self, x):
        xt = x["nodes", "tasks", "attr"]  # (..., tasks, C_in_no_coords)
        z = x["aux", "z"]

        #print(f"[Encoder] input {xt.shape}")
        #print(f"[features] ", xt[0, :])
        single = (xt.dim() == 2)
        if single:
            xt = xt.unsqueeze(0)  # (1, tasks, C)

        *batch_shape, T, Cin = xt.shape
        H, W = self.length, self.width
        assert T == H * W, f"tasks={T} differs from length*width={H*W}"
        assert Cin == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {Cin}"

        # Flatten and reshape to BCHW
        B = 1
        for d in batch_shape: B *= int(d)
        h = xt.reshape(B, H, W, Cin).permute(0, 3, 1, 2)  # (B,Cin,H,W)

        if self.add_coord:
            coords = _coord_mesh(H, W, device=h.device, dtype=h.dtype)  # (2,H,W)
            coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
            h = torch.cat([h, coords], dim=1)

        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.aspp(h)
        h = self.eca(h)

        if single:
            h = h.squeeze(0)  # (C,H,W)
            if self.debug: print(f"[Encoder] embed {h.shape}")
            return (h,)
        else:
            h = h.view(*batch_shape, *h.shape[1:])  # (*batch, C, H, W)
            if self.debug: print(f"[Encoder] embed {h.shape}")

        # zB, _, _ = _flatten_last_dim(z)
        # gamma_beta = self.film(z)
        # gamma, beta = gamma_beta.chunk(2, dim=-1)
        # print(f"H shape {h.shape}, gamma shape {gamma.shape}, beta shape {beta.shape}")
        #h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)
        
        return (h,)

class ConditionedDilationRectangularEncoder(nn.Module):

    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        width: int,
        length: int,
        add_progress: bool = False,                 
        minimum_resolution: int = 2,                
        activation: Optional[DictConfig] = None,     # kept for BC
        initialization: Optional[DictConfig] = None, # kept for BC
        debug: bool = True,
        pool_mode: str = "avg",                      # ignored; no pooling
        # New optional knobs (do not break existing calls):
        add_coord: bool = True,
        num_blocks: int = 2,
        dilation_schedule: Optional[List[int]] = None,
        use_tiny_aspp: bool = True,
        use_eca: bool = True,
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

        self.add_coord = bool(add_coord)
        C_in = self.in_channels + (2 if self.add_coord else 0)
        C = self.hidden_channels
        self.stem = ConvNormAct(C_in, C, k=3, dilation=1, act="silu")

        # Dilated residual stack
        if not dilation_schedule:
            dilation_schedule = [1, 2, 3]  # short, aperiodic cycle for ≤32x32
        self.blocks = nn.ModuleList([
            DilatedResBlock(C, dilation=dilation_schedule[i % len(dilation_schedule)], act="silu")
            for i in range(num_blocks)
        ])

        # Tiny ASPP and optional ECA gate
        self.aspp = TinyASPP(C) if use_tiny_aspp else nn.Identity()
        self.eca  = ECA(C, k_size=3) if use_eca else nn.Identity()

        self.film = nn.Linear(8, 2*self.hidden_channels, bias=True)

        self.in_channels_per_scale: List[int] = [C]
        self.output_dim = C
        self.output_keys: List[str] = ["embed"]

    def forward(self, x):
        xt = x["nodes", "tasks", "attr"]  # (..., tasks, C_in_no_coords)
        z = x["aux", "z"]

        #print(f"[Encoder] input {xt.shape}")
        #print(f"[features] ", xt[0, :])
        single = (xt.dim() == 2)
        if single:
            xt = xt.unsqueeze(0)  # (1, tasks, C)

        *batch_shape, T, Cin = xt.shape
        H, W = self.length, self.width
        assert T == H * W, f"tasks={T} differs from length*width={H*W}"
        assert Cin == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {Cin}"

        # Flatten and reshape to BCHW
        B = 1
        for d in batch_shape: B *= int(d)
        h = xt.reshape(B, H, W, Cin).permute(0, 3, 1, 2)  # (B,Cin,H,W)

        if self.add_coord:
            coords = _coord_mesh(H, W, device=h.device, dtype=h.dtype)  # (2,H,W)
            coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
            h = torch.cat([h, coords], dim=1)

        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.aspp(h)
        h = self.eca(h)

        if single:
            h = h.squeeze(0)  # (C,H,W)
            if self.debug: print(f"[Encoder] embed {h.shape}")
            return (h,)
        else:
            h = h.view(*batch_shape, *h.shape[1:])  # (*batch, C, H, W)
            if self.debug: print(f"[Encoder] embed {h.shape}")

        zB, _, _ = _flatten_last_dim(z)
        gamma_beta = self.film(z)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        print(f"H shape {h.shape}, gamma shape {gamma.shape}, beta shape {beta.shape}")
        h = gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)
        
        return (h,)


class DilationRectangularDecoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        width: int,
        length: int,
        output_dim: int,
        minimum_resolution: int = 2,                
        activation: Optional[DictConfig] = None,
        initialization: Optional[DictConfig] = None,
        layer_norm: bool = False,                    # we use GroupNorm internally
        add_progress: bool = False,
        progress_dim: int = 0,
        debug: bool = True,
        upsample_type: str = "nearest",              
        deconv_bilinear_init: bool = True,           
        # New optional knobs (safe defaults):
        num_blocks: int = 2,
        dilation_schedule: Optional[List[int]] = None,
        use_eca: bool = False,
    ):
        super().__init__()
        self.width = int(width)
        self.length = int(length)
        self.input_dim = int(input_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.add_progress = bool(add_progress)
        self.progress_dim = int(progress_dim)
        self.debug = bool(debug)

        # No upsampling in this decoder
        self.num_layers = 0

        self.film = None
        if self.add_progress:
            assert self.progress_dim > 0, "progress_dim must be > 0 when add_progress=True"
            self.film = nn.Linear(self.progress_dim, 2 * self.input_dim, bias=True)

        # Small fixed-resolution head
        if not dilation_schedule:
            dilation_schedule = [1, 2]
        self.pre = ConvNormAct(self.input_dim, self.hidden_channels, k=3, dilation=1, act="silu")
        self.blocks = nn.ModuleList([
            DilatedResBlock(self.hidden_channels, dilation=dilation_schedule[i % len(dilation_schedule)], act="silu")
            for i in range(num_blocks)
        ])
        self.eca = ECA(self.hidden_channels, k_size=3) if use_eca else nn.Identity()
        #self.out_conv = nn.Conv2d(self.hidden_channels, 2*self.output_dim, kernel_size=1)
        self.out_conv = nn.Conv2d(self.hidden_channels, self.output_dim, kernel_size=1)

        self.in_channels_per_scale: List[int] = [self.input_dim]
        self.input_keys: List[str] = ["observation", "embed"]

        #self.logit_layer = LogitsOutputHead(input_dim=2*self.output_dim, hidden_channels=16, output_dim=self.output_dim)

    def forward(self, obs, *features):
        if len(features) == 0:
            raise ValueError("Decoder expects encoder features: (*enc_feats, bottleneck_map)")
        b_map = features[-1]  # (C,H,W) or (*batch,C,H,W)

        # Normalize to BCHW
        hB, batch_shape, B = _flatten_to_BCHW(b_map)  # (B,C,H,W)
        _, C, H, W = hB.shape
        assert C == self.input_dim, f"Decoder input_dim={self.input_dim}, got bottleneck C={C}"
        if self.debug: print(f"[Decoder] b_map {hB.shape}")

        if self.add_progress:
            prog = obs["aux", "progress"]  # (..., P)
            progB, _, _ = _flatten_last_dim(prog)      # (B, P)
            gamma_beta = self.film(progB)              # (B, 2*C)
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # (B, C), (B, C)
            hB = gamma.unsqueeze(-1).unsqueeze(-1) * hB + beta.unsqueeze(-1).unsqueeze(-1)
            if self.debug: print(f"[Decoder] FiLM applied {hB.shape}")

        # Fixed-resolution head
        hB = self.pre(hB)
        for blk in self.blocks:
            hB = blk(hB)
        hB = self.eca(hB)
        logits_map = self.out_conv(hB)  # (B, A, H, W)
        if self.debug: print(f"[Decoder] logits_map {logits_map.shape}")

        if len(batch_shape) == 0:
            logits = logits_map.permute(0, 2, 3, 1).reshape(H * W, self.output_dim).squeeze(0)
        else:
            logits = logits_map.permute(0, 2, 3, 1).reshape(B, H * W, self.output_dim).view(
                *batch_shape, H * W, self.output_dim
            )
        #return self.logit_layer(logits)
        return logits

class UNetRectangularEncoder(nn.Module):

    def __init__(
        self,
        feature_config,
        hidden_channels: int,
        width: int,
        length: int,
        add_progress: bool = False,
        minimum_resolution: int = 2,
        activation: Optional[DictConfig] = None,
        initialization: Optional[DictConfig] = None,
        debug: bool = True,
        pool_mode: str = "avg",
    ):
        super().__init__()
        if not hasattr(feature_config, "task_feature_dim"):
            raise AttributeError("feature_config must have attribute 'task_feature_dim'")
        self.width = int(width)
        self.length = int(length)
        self.in_channels = int(feature_config.task_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.minimum_resolution = int(minimum_resolution)
        self.debug = debug

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
                out_ch = self.hidden_channels * (2 ** i)
                self.enc_blocks.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
                    nn.LeakyReLU(negative_slope=0.01, inplace=False),
                ))
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

        self.output_dim = channels
        self.output_keys: List[str] = [f"enc_{i}" for i in range(self.num_layers)] + ["embed"]


    def forward(self, x):
        xt = x["nodes", "tasks", "attr"] # shape: (*batch, tasks, C) or (tasks, C)

        single = (xt.dim() == 2)  # (tasks, C)
        if single:
            xt = xt.unsqueeze(0)  # -> (1, tasks, C)

        *batch_shape, tasks, in_ch = xt.shape
        assert in_ch == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {in_ch}"
        assert tasks == self.length * self.width, f"got tasks={tasks}, expected length*width={self.length*self.width}"

        B = 1
        for d in batch_shape: B *= int(d)
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
        if self.debug:
            print(f"[Encoder] bottleneck {h.shape}")

        def unflatten(t):
            return t.squeeze(0) if single else t.view(*batch_shape, *t.shape[1:])

        enc_feats = [unflatten(f) for f in enc_feats]
        b_map = unflatten(b_map)
        return (*enc_feats, b_map)


class UNetRectangularDecoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        width: int,
        length: int,
        output_dim: int,
        minimum_resolution: int = 2,
        activation: Optional[DictConfig] = None,
        initialization: Optional[DictConfig] = None,
        layer_norm: bool = False,
        add_progress: bool = False,
        progress_dim: int = 0,
        debug: bool = True,
        upsample_type: str = "nearest",
        deconv_bilinear_init: bool = True,
    ):
        super().__init__()
        self.width = int(width)
        self.length = int(length)
        self.hidden_channels = int(hidden_channels)
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)
        self.minimum_resolution = int(minimum_resolution)
        self.add_progress = bool(add_progress)
        self.progress_dim = int(progress_dim)
        self.upsample_type = str(upsample_type)
        self.deconv_bilinear_init = bool(deconv_bilinear_init)
        self.debug = debug

        self.num_layers = _compute_num_downsampling_layers(self.length, self.width, self.minimum_resolution)

        if self.num_layers == 0:
            self.in_channels_per_scale = [self.input_dim]   # just bottleneck
        else:
            skip_channels = [self.hidden_channels * (2 ** i) for i in range(self.num_layers)]
            self.in_channels_per_scale = [*skip_channels, self.input_dim]

        expected_bottleneck_ch = self.hidden_channels * (2 ** max(self.num_layers - 1, 0))
        assert self.input_dim == expected_bottleneck_ch, (
            f"Decoder input_dim={self.input_dim} must equal encoder bottleneck channels {expected_bottleneck_ch}"
        )

        if self.add_progress:
            assert self.progress_dim > 0, "progress_dim must be > 0 when add_progress=True"
            self.film = nn.Linear(self.progress_dim, 2 * self.input_dim, bias=True)

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        prev_ch = self.input_dim
        for i in reversed(range(self.num_layers)):
            out_ch = self.hidden_channels * (2 ** i)

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
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=False),
            ))
            prev_ch = out_ch

        self.input_keys: List[str] = ["observation"] + [f"enc_{i}" for i in range(self.num_layers)] + ["embed"]

        # Final projection to logits at full resolution
        final_in = self.hidden_channels if self.num_layers >= 1 else self.input_dim
        self.out_conv = nn.Conv2d(final_in, 2*self.output_dim, kernel_size=1)

        self.logit_layer = LogitsOutputHead(
            input_dim=2*self.output_dim,
            hidden_channels=self.hidden_channels,
            output_dim=self.output_dim,
        )

    def forward(self, obs, *features):
        if len(features) == 0:
            raise ValueError("Decoder expects encoder features: (*enc_feats, bottleneck_map)")
        enc_feats = features[:-1]
        b_map = features[-1]  # shape: (C,H,W) or (*batch,C,H,W)

        single = (b_map.dim() == 3)

        # Normalize shapes
        b_mapB, batch_shape, B = _flatten_to_BCHW(b_map)
        encB = [_flatten_to_BCHW(e)[0] for e in enc_feats]

        if self.add_progress:
            prog = obs["aux", "progress"]  # (..., P)
            progB, _, _ = _flatten_last_dim(prog)
            gamma_beta = self.film(progB)           # (B, 2*C)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            b_mapB = gamma * b_mapB + beta
            if self.debug:
                print(f"[Decoder] FiLM {b_mapB.shape}")

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

        logits_map = self.out_conv(h)          # (B, output_dim, H, W)
        if self.debug:
            print(f"[Decoder] logits: {logits_map.shape}")

        if single:
            # -> (H*W, output_dim)
            logits =  logits_map.permute(0, 2, 3, 1).reshape(-1, 2*self.output_dim).squeeze(0)
            logits = self.logit_layer(logits)  # (H*W, output_dim)
        else:
            _, _, H, W = logits_map.shape
            logits = logits_map.permute(0, 2, 3, 1).reshape(B, H * W, 2*self.output_dim)\
                             .view(*batch_shape, H * W, 2*self.output_dim)
            logits = self.logit_layer(logits)  # (*batch, H*W, output_dim)

        return logits


class PooledOutputHead(nn.Module):
    def __init__(
        self,
        input_dim: int,                 # shared dim before final MLP
        hidden_channels: int,           # hidden size in OutputHead
        output_dim: int,                # final dimension (e.g., 1 for V(s))
        activation: Optional[nn.Module] = None,
        initialization: Optional[dict] = None,
        layer_norm: bool = True,
        in_channels_per_scale: Optional[Sequence[int]] = None,
        debug: bool = False
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

        if in_channels_per_scale is not None:
            self._build(list(int(c) for c in in_channels_per_scale))

        self.in_channels_per_scale: Optional[List[int]] = (
            list(in_channels_per_scale) if in_channels_per_scale is not None else None
        )

        self._oh_kwargs = dict(
            input_dim=self.proj_dim,
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
        self._proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(Ci),
                nn.Linear(Ci, self.proj_dim, bias=False),
            ) for Ci in in_dims
        ])

        for m in self._proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self._head = OutputHead(**self._oh_kwargs)

        self._built = True

    def forward(self, *encoder_outputs: torch.Tensor) -> torch.Tensor:
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
            z = proj(z)                                  # (B, D)
            zs.append(z)
            if self.debug:
                print(f"[PooledOutputHead] features {fB.shape}")

        v = torch.stack(zs, dim=1).sum(dim=1)            # (B, D)
        yB = self._head(v)                                # (B, output_dim)
        return _unflatten_from_B(yB, batch_shape_ref or ())


class UNetEncoder(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        hidden_channels: int,
        width: int,
        length: int, 
        add_progress: bool = False,
        activation: DictConfig = None,
        initialization: DictConfig = None,
    ):
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
                    inplace=False,
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
                inplace=False,
                negative_slope=0.01,
            ),
        )
        self.output_dim = channels + (1 if add_progress else 0)
        self.output_keys.append("embed")

    # def forward(self, obs):
    #     single = obs.batch_size == torch.Size([])
    #     x = obs["nodes", "tasks", "attr"]
    #     if single:
    #         x = x.unsqueeze(0)
    #     bsz = x.size(0)
    #     x = x.view(bsz, self.width, self.width, self.in_channels).permute(0, 3, 1, 2)
    #     enc_feats = []
    #     for block in self.enc_blocks:
    #         x = block(x)
    #         enc_feats.append(x)
    #         x = self.pool(x)
    #     b = self.bottleneck(x)
    #     b = b.flatten(start_dim=1)
    #     if single:
    #         b = b.squeeze(0)

    #     if self.add_progress:
    #         # Add time and progress features
    #         progress_feature = obs["aux", "progress"]
    #         b = torch.cat([b, progress_feature], dim=-1)
    #     return (*enc_feats, b)

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
        x_flat = x_flat.view(
            flat_bs, width, width, in_channels
        ).permute(  # (flat_bs, W, W, C_in)
            0, 3, 1, 2
        )  # (flat_bs, C_in, W, W)

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


class UNetDecoder(nn.Module):
    input_keys = ["observation"]

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        width: int,
        length: int, 
        output_dim: int,
        activation: DictConfig = None,
        initialization: DictConfig = None,
        layer_norm: bool = False,
    ):
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
            in_ch = (
                hidden_channels * (2 ** (i + 1))
                if i < self.num_layers - 1
                else hidden_channels * (2**i)
            )
            out_ch = hidden_channels * (2**i)
            self.up_blocks.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                )
            )
        for i in range(self.num_layers):
            self.input_keys.append(f"enc_{i}")
        self.input_keys.append("embed")
        self.out_conv = nn.Conv2d(hidden_channels, 2*output_dim, kernel_size=1)

        print(f"Initialization Dict: {initialization}")
        print(f"Activation Dict: {activation}")

        self.logit_layer = LogitsOutputHead(input_dim=2*output_dim, hidden_channels=16, output_dim=output_dim)
        assert (
            input_dim == self.up_blocks[0].in_channels
        ), f"Input dimension mismatch: expected {self.up_blocks[0].in_channels}, got {input_dim}"

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
        logits = self.logit_layer(logits)

        if single:
            logits = logits.squeeze(0)
        else:
            logits = logits.view(*batch_shape, self.width * self.width, -1)
        return logits