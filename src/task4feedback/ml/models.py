from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
import torch
from typing import Optional, Self

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import observation_to_heterodata
from dataclasses import dataclass

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


def layer_init(layer, a=0.01, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight, a=a, nonlinearity="leaky_relu")
    torch.nn.init.constant_(layer.bias, bias_const)
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

        with record_function("CONVERT_DATA"):
            with torch.no_grad():
                data = self._convert_to_heterodata(obs, is_batch)

        with record_function("FORWARD_PASS"):
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

        with record_function("CONVERT_DATA"):
            with torch.no_grad():
                data, task_count, data_count = self._convert_to_heterodata(
                    obs, is_batch, actions=actions
                )

        with record_function("FORWARD_PASS"):
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
    def __init__(self, input_dim, hidden_channels, output_dim, logits=True):
        super(OutputHead, self).__init__()

        self.fc1 = layer_init(nn.Linear(input_dim, hidden_channels))
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_channels, output_dim)

        if logits:
            # nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
            nn.init.uniform_(self.fc2.weight, a=-0.001, b=0.001)
            nn.init.constant_(self.fc2.bias, 0.0)
        else:
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.fc2(x)

        # print("x", x.shape, flush=True)
        return x


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

        with record_function("HeteroConvStateNet"):
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
        with record_function("hetero_conv"):
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
        hidden_channels: list[int],
        add_progress: bool = False,
    ):
        super(VectorStateNet, self).__init__()
        self.feature_config = feature_config
        self.hidden_channels = hidden_channels
        self.k = len(self.hidden_channels)

        if self.k == 0:
            self.layers = nn.Identity()
            self.output_dim = feature_config.task_feature_dim
        else:
            # Build k MLP layers
            layers = []
            input_dim = feature_config.task_feature_dim
            layer_channels = input_dim
            for i in range(self.k):
                layer_channels = hidden_channels[i]
                layers.append(layer_init(nn.Linear(input_dim, layer_channels)))
                layers.append(nn.LayerNorm(layer_channels))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                input_dim = layer_channels

            self.layers = nn.Sequential(*layers)
            self.output_dim = layer_channels

        self.add_progress = add_progress
        if self.add_progress:
            self.output_dim += 2

    def forward(self, tensordict: TensorDict):
        task_features = tensordict["nodes", "tasks", "attr"]
        task_features = torch.squeeze(task_features)
        # print("TF SHAPE", task_features.shape)

        task_activations = self.layers(task_features)

        if self.add_progress:
            time_feature = tensordict["aux", "time"] / tensordict["aux", "baseline"]
            progress_feature = tensordict["aux", "progress"]

            task_activations = torch.cat(
                [task_activations, time_feature, progress_feature], dim=-1
            )

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
