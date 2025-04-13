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
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, HeteroConv
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
    def __init__(self, network: nn.Module, device: None):
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
            return _obs

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

        return Batch.from_data_list(_h_data)

    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs, is_batch, actions=actions)
        data = data.to(self.device)
        return self.network(data)


class ActorWrapper(HeteroDataWrapper):
    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs, is_batch, actions=actions)
        data = data.to(self.device)
        # Compute task embeddings from the hetero-GAT network
        task_embeddings = self.network(data)
        # Extract candidate embedding based on batch presence
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None
        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]
        # Return the candidate embedding wrapped in a dict with the expected key 'embed'
        return {"embed": candidate_embedding}


class CriticEmbedWrapper(HeteroDataWrapper):
    def forward(self, obs: TensorDict, actions: Optional[TensorDict] = None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs, is_batch, actions=actions)
        data = data.to(self.device)
        # Compute task embeddings from the hetero-GAT network
        task_embeddings = self.network(data)

        # Extract candidate embedding based on batch presence
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        # Aggregate node embeddings to get one embedding per graph/sample
        # This ensures the resulting embedding tensor has shape [batch_size, feature_dim]
        pooled_embeddings = global_mean_pool(task_embeddings, task_batch)
        # Return the pooled embedding wrapped in a dict with the expected key 'embed'
        return {"embed": pooled_embeddings}


class CriticHeadWrapper(nn.Module):
    def __init__(self, critic: nn.Module, device: str):
        super().__init__()
        self.critic = critic
        self.device = device

    def forward(self, obs: TensorDict):
        v = self.critic(obs)
        # v = global_mean_pool(v, obs["task_batch"])
        return {"state_value": v}


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
        print(f"task_feature_dim: {observer.task_feature_dim}")
        print(f"data_feature_dim: {observer.data_feature_dim}")
        print(f"device_feature_dim: {observer.device_feature_dim}")
        print(f"task_data_edge_dim: {observer.task_data_edge_dim}")
        print(f"task_device_edge_dim: {observer.task_device_edge_dim}")
        print(f"task_task_edge_dim: {observer.task_task_edge_dim}")

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

        self.gnn_tasks_data = GATConv(
            (feature_config.data_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_data_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATConv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_devices = GATConv(
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

        # with torch.no_grad():
        #     mean_devices = torch.mean(data["devices"].x, dim=0, keepdim=True)
        #     data["devices"].x = data["devices"].x / mean_devices

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
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    residual=True,
                    dropout=0,
                    edge_dim=feature_config.task_data_edge_dim,
                ),
                ("data", "to", "tasks"): GATConv(
                    (hidden_channels_with_heads, hidden_channels_with_heads),
                    layer_config.hidden_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    residual=True,
                    dropout=0,
                    edge_dim=feature_config.task_data_edge_dim,
                ),
            }
        )

        self.norm_tasks = nn.LayerNorm(layer_config.hidden_channels)
        self.norm_data = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.output_dim = layer_config.hidden_channels

    def forward(self, data: HeteroData | Batch):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = (
            data.edge_attr_dict if hasattr(data, "edge_attr_dict") else None
        )
        x_dict = self.data_task_conv(x_dict, edge_index_dict, edge_attr_dict)

        x_dict = {node_type: self.activation(x) for node_type, x in x_dict.items()}
        x_dict = {
            "tasks": self.norm_tasks(x_dict["tasks"]),
            "data": self.norm_data(x_dict["data"]),
        }
        x_dict = self.task_data_conv(x_dict, edge_index_dict, edge_attr_dict)

        return x_dict


class TaskTaskGAT2Layer(nn.Module):
    """
    Tasks-to-tasks encodes task -> dependency information

    This module performs two depth 2 GAT convolutions on the task nodes:
    - Two accumulations of task -> dependency information
    - Two accumulations of task -> dependant information

    These results are then concatenated and returned as the output (hidden_channels * 2)
    """

    def __init__(self, feature_config: FeatureDimConfig, layer_config: LayerConfig):
        super(TaskTaskGAT2Layer, self).__init__()

        self.conv_dependency_1 = GATConv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent_1 = GATConv(
            (feature_config.task_feature_dim, feature_config.task_feature_dim),
            layer_config.hidden_channels,
            heads=layer_config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.conv_dependency_2 = GATConv(
            (layer_config.hidden_channels, layer_config.hidden_channels),
            layer_config.hidden_channels,
            heads=1,
            concat=True,
            residual=True,
            dropout=0,
            edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent_2 = GATConv(
            (layer_config.hidden_channels, layer_config.hidden_channels),
            layer_config.hidden_channels,
            heads=1,
            concat=True,
            residual=True,
            edge_dim=feature_config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.norm_dependency = nn.LayerNorm(layer_config.hidden_channels)
        self.norm_dependant = nn.LayerNorm(layer_config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_dim = layer_config.hidden_channels * 2

    def forward(self, task_embedding, data: HeteroData | Batch):
        tasks = task_embedding
        edge_dependency_index = data.edge_index_dict["tasks", "to", "tasks"]
        edge_dependant_index = edge_dependency_index.flip(0)
        edge_attr = data.edge_attr_dict["tasks", "to", "tasks"]

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

        return torch.cat([tasks_dependency, tasks_dependant], dim=-1)


class OutputHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OutputHead, self).__init__()

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

    def forward(self, x, y):
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
        gat_output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )

        self.actor = OutputHead(gat_output_dim, layer_config.hidden_channels, n_devices)
        self.critic = OutputHead(gat_output_dim, layer_config.hidden_channels, 1)

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

        if counts is None:
            v = self.critic(task_embeddings)
            v = global_mean_pool(v, task_batch)
        else:
            v = self.critic(task_embeddings)
            v = global_add_pool(v, task_batch)
            v = torch.div(v, counts)

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

        self.hetero_gat = HeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )
        self.actor_head = OutputHead(
            gat_output_dim, layer_config.hidden_channels, n_devices - 1
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

        # print("candidate_embedding", candidate_embedding)

        d_logits = self.actor_head(candidate_embedding)

        # print(f"d_logits: {d_logits}, {d_logits.shape}")

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

        self.hetero_gat = HeteroGAT1Layer(feature_config, layer_config)
        gat_output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )
        self.critic_head = OutputHead(gat_output_dim, layer_config.hidden_channels, 1)

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        if counts is None:
            v = self.critic_head(task_embeddings)
            v = global_mean_pool(v, task_batch)
        else:
            v = self.critic_head(task_embeddings)
            v = global_add_pool(v, task_batch)
            v = torch.div(v, counts)

        # print(f"v: {v}, {v.shape}")

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
        gat_output_dim = (
            layer_config.hidden_channels * 3 + feature_config.task_feature_dim
        )

        self.critic_head = OutputHead(
            gat_output_dim, layer_config.hidden_channels, n_devices - 1
        )

    def forward(self, data: HeteroData | Batch, counts=None):
        task_embeddings = self.hetero_gat(data)
        task_batch = data["tasks"].batch if isinstance(data, Batch) else None

        if counts is None:
            v = global_mean_pool(task_embeddings, task_batch)
            v = self.critic_head(v)

        else:
            v = global_mean_pool(task_embeddings, task_batch)
            v = torch.div(v, counts)
            v = self.critic_head(v)

        # print(f"av: {v}, {v.shape}")
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
