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

from tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
    GATConv,
    global_mean_pool,
    global_add_pool,
    HeteroConv,
    SAGEConv,
)
import numpy as np


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


class HeteroDataWrapper(nn.Module):
    def __init__(self, network: nn.Module, device: Optional[str] = "cpu"):
        super(HeteroDataWrapper, self).__init__()
        self.network = network
        if device is None:
            self.device = (
                torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
            )
        else:
            self.device = device

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
        # print("obs", obs.shape)
        # if actions is not None:
        #     print("actions", actions.shape)
        if not is_batch:
            if actions is not None:
                _obs = observation_to_heterodata(obs, actions=actions)
            else:
                _obs = observation_to_heterodata(obs)
            return (
                _obs,
                obs["nodes", "tasks", "count"],
                obs["nodes", "data", "count"],
            )

        # print("BATCH DIMS", obs.batch_dims)
        # print("BATCH SIZE", obs.batch_size)

        # flatten and save the batch size
        self.batch_size = obs.batch_size
        # obs = obs.reshape(-1)

        _h_data = []
        # print("obs", obs.shape)
        for i in range(obs.batch_size[0]):
            # print("obs[i]", obs[i].shape)
            if actions is not None:
                _obs = observation_to_heterodata(obs[i], actions=actions[i])
            else:
                _obs = observation_to_heterodata(obs[i])
            _h_data.append(_obs)

        return (
            Batch.from_data_list(_h_data),
            obs["nodes", "tasks", "count"],
            obs["nodes", "data", "count"],
        )

    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data, task_count, data_count = self._convert_to_heterodata(
            obs, is_batch, actions=actions
        )
        data = data.to(self.device)
        out = self.network(data, (task_count, data_count))

        # rehape the output to the original batch size
        # if is_batch:
        #    out = out.reshape(self.batch_size[0], -1, out.shape[-1])
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
            edge_dim=feature_config.task_data_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATv2Conv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_devices = GATv2Conv(
            (feature_config.device_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_device_edge_dim,
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
            data["data", "to", "tasks"].edge_attr,
        )

        tasks_fused_tasks = self.gnn_tasks_tasks(
            (data["tasks"].x, data["tasks"].x),
            data["tasks", "to", "tasks"].edge_index,
            data["tasks", "to", "tasks"].edge_attr,
        )

        devices_fused_tasks = self.gnn_tasks_devices(
            (data["devices"].x, data["tasks"].x),
            data["devices", "to", "tasks"].edge_index,
            data["devices", "to", "tasks"].edge_attr,
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


class TaskTaskGAT2Layer(nn.Module):
    """
    Tasks-to-tasks encodes task -> dependency information

    This module performs two depth 2 GAT convolutions on the task nodes:
    - Two accumulations of task -> dependency information
    - Two accumulations of task -> dependant information

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
        super(TaskTaskGAT2Layer, self).__init__()

        if not use_edge_features:
            edge_dim = None
        else:
            edge_dim = feature_config.task_task_edge_dim

        self.conv_dependency_1 = GATv2Conv(
            (input_dim, input_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent_1 = GATv2Conv(
            (input_dim, input_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        self.conv_dependency_2 = GATv2Conv(
            (layer_config.hidden_channels, layer_config.hidden_channels),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent_2 = GATv2Conv(
            (layer_config.hidden_channels, layer_config.hidden_channels),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            edge_dim=feature_config.task_task_edge_dim,
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

        tasks_dependency = self.conv_dependency_1(
            tasks, edge_dependency_index, edge_attr
        )
        tasks_dependency = self.norm_dependency(tasks_dependency)
        tasks_dependency = self.activation(tasks_dependency)

        tasks_dependant = self.conv_dependent_1(tasks, edge_dependant_index, edge_attr)
        tasks_dependant = self.norm_dependant(tasks_dependant)
        tasks_dependant = self.activation(tasks_dependant)

        tasks_dependency = self.conv_dependency_2(
            tasks_dependency, edge_dependency_index, edge_attr
        )
        tasks_dependant = self.conv_dependent_2(
            tasks_dependant, edge_dependant_index, edge_attr
        )
        tasks_dependant = self.activation(tasks_dependant)
        tasks_dependency = self.activation(tasks_dependency)

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
    def __init__(self, input_dim, hidden_dim, output_dim, logits=True):
        super(OutputHead, self).__init__()

        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

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


class DataTaskGAT(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        skip_connection: bool = True,
    ):
        super(DataTaskGAT, self).__init__()

        self.layer_config = layer_config

        self.conv_data_task = GATv2Conv(
            (feature_config.data_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_data_edge_dim,
            add_self_loops=False,
        )

        self.output_layer = layer_init(
            nn.Linear(layer_config.hidden_channels, layer_config.hidden_channels)
        )

        self.output_dim = layer_config.hidden_channels

        self.skip_connection = skip_connection
        if skip_connection:
            self.output_dim += feature_config.task_feature_dim

        self.layer_norm = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data: HeteroData | Batch):
        task_features = data["tasks"].x
        data_features = data["data"].x
        data_task_edges = data["data", "to", "tasks"].edge_index
        data_task_edges_attr = data["data", "to", "tasks"].edge_attr

        task_embeddings = self.conv_data_task(
            (data_features, task_features),
            data_task_edges,
            data_task_edges_attr,
        )
        task_embeddings = self.layer_norm(task_embeddings)
        task_embeddings = self.activation(task_embeddings)

        if self.skip_connection:
            task_embeddings = torch.cat([task_embeddings, task_features], dim=-1)

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
        self.task_task_layer = TaskTaskGAT2Layer(
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


class ValueNet2Layer(nn.Module):
    def __init__(
        self,
        feature_config: FeatureDimConfig,
        layer_config: LayerConfig,
        n_devices: int = 5,
    ):
        super(ValueNet2Layer, self).__init__()

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
        self.task_task_layer = TaskTaskGAT2Layer(
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
        self.critic = ValueNet2Layer(feature_config, layer_config, n_devices)

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

        # print("device_features", data["devices"].x.shape)
        device_features = data["devices"].x
        # print("device_features", device_features.shape)
        device_features = device_features.view(
            -1, self.n_devices * self.feature_config.device_feature_dim
        )

        return device_features


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

        self.data_task_gat = DataTaskGAT(feature_config, layer_config)
        data_task_dim = self.data_task_gat.output_dim

        self.task_task_gat = TaskTaskGAT2Layer(
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

        candidate_embedding = torch.cat(
            [candidate_embedding, global_embedding, device_features], dim=-1
        )

        d_logits = self.actor_head(candidate_embedding)

        return d_logits


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

        self.data_task_gat = DataTaskGAT(feature_config, layer_config)
        data_task_dim = self.data_task_gat.output_dim

        self.task_task_gat = TaskTaskGAT2Layer(
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
