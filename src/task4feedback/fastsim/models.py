# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch_geometric.data import HeteroData
# # from torch_geometric.nn import HeteroConv, GATConv, TransformerConv
# # from torch.distributions.categorical import Categorical
# # from torch.utils.tensorboard import SummaryWriter
# # from torch_geometric.nn import global_mean_pool
# # import numpy as np


# # # class HeteroGAT(nn.Module):

# # #     def __init__(self, hidden_channels, data):
# # #         super(HeteroGAT, self).__init__()

# # #         metadata = data.metadata()
# # #         self.convs = nn.ModuleList()

# # #         # Task feature size
# # #         self.in_channels_tasks = data["tasks"].x.shape[1]
# # #         self.in_channels_data = data["data"].x.shape[1]
# # #         self.in_channels_devices = data["devices"].x.shape[1]

# # #         self.task_data_edge_dim = data[("data", "used_by", "tasks")].edge_attr.shape[1]
# # #         self.task_device_edge_dim = data[
# # #             ("devices", "variant", "tasks")
# # #         ].edge_attr.shape[1]
# # #         self.task_task_edge_dim = data[
# # #             ("tasks", "depends_on", "tasks")
# # #         ].edge_attr.shape[1]

# # #         self.n_heads = 4

# # #         # Data - Tasks with edge features
# # #         self.gnn_tasks_data = GATConv(
# # #             (self.in_channels_data, self.in_channels_tasks),
# # #             hidden_channels,
# # #             heads=self.n_heads,
# # #             concat=True,
# # #             dropout=0.0,
# # #             edge_dim=self.task_data_edge_dim,
# # #             add_self_loops=False,
# # #         )

# # #         gnn_tasks_data_dim = hidden_channels * self.n_heads
# # #         gnn_tasks_devices_dim = hidden_channels * self.n_heads

# # #         self.gnn_tasks_devices = GATConv(
# # #             (self.in_channels_devices, self.in_channels_tasks),
# # #             hidden_channels,
# # #             heads=self.n_heads,
# # #             concat=True,
# # #             dropout=0.0,
# # #             edge_dim=self.task_device_edge_dim,
# # #             add_self_loops=False,
# # #         )

# # #         self.gnn_tasks_tasks = GATConv(
# # #             gnn_tasks_data_dim + gnn_tasks_devices_dim,
# # #             2*hidden_channels,
# # #             heads=self.n_heads,
# # #             concat=True,
# # #             dropout=0,
# # #             edge_dim=self.task_task_edge_dim,
# # #             add_self_loops=True,
# # #         )

# # #         self.batch_norm1 = nn.BatchNorm1d(hidden_channels * self.n_heads)
# # #         self.batch_norm2 = nn.BatchNorm1d(hidden_channels * self.n_heads)
# # #         self.batch_norm3 = nn.BatchNorm1d(2*hidden_channels * self.n_heads)

# # #     def forward(self, data):
# # #         data_fused_tasks = self.gnn_tasks_data(
# # #             (data["data"].x, data["tasks"].x),
# # #             data["data", "used_by", "tasks"].edge_index,
# # #             data["data", "used_by", "tasks"].edge_attr,
# # #         )
# # #         data_fused_tasks = self.batch_norm1(data_fused_tasks)
# # #         data_fused_tasks = torch.relu(data_fused_tasks)

# # #         device_fused_tasks = self.gnn_tasks_devices(
# # #             (data["devices"].x, data["tasks"].x),
# # #             data["devices", "variant", "tasks"].edge_index,
# # #             data["devices", "variant", "tasks"].edge_attr,
# # #         )
# # #         device_fused_tasks = self.batch_norm2(device_fused_tasks)
# # #         device_fused_tasks = torch.relu(device_fused_tasks)

# # #         tasks = torch.cat([data_fused_tasks, device_fused_tasks], dim=1)
# # #         residual = tasks

# # #         x = self.gnn_tasks_tasks(
# # #             tasks,
# # #             data["tasks", "depends_on", "tasks"].edge_index,
# # #             data["tasks", "depends_on", "tasks"].edge_attr,
# # #         )
# # #         x = x + residual
# # #         x = self.batch_norm3(x)
# # #         x = torch.relu(x)
# # #         x = torch.dropout(x, p=0.1, train=self.training)

# # #         return x

# class HeteroGAT(nn.Module):
#     def __init__(self, hidden_channels, data):
#         super(HeteroGAT, self).__init__()

#         # Extract metadata and input feature sizes
#         self.in_channels_tasks = data["tasks"].x.shape[1]
#         self.in_channels_data = data["data"].x.shape[1]
#         self.in_channels_devices = data["devices"].x.shape[1]

#         # Extract edge feature dimensions
#         self.task_data_edge_dim = data[("data", "used_by", "tasks")].edge_attr.shape[1]
#         self.task_device_edge_dim = data[("devices", "variant", "tasks")].edge_attr.shape[1]
#         self.task_task_edge_dim = data[("tasks", "depends_on", "tasks")].edge_attr.shape[1]

#         self.n_heads = 4

#         # Define GATConv layers
#         self.gnn_tasks_data = GATConv(
#             (self.in_channels_data, self.in_channels_tasks),
#             hidden_channels,
#             heads=self.n_heads,
#             concat=True,
#             dropout=0.0,
#             edge_dim=self.task_data_edge_dim,
#             add_self_loops=False,
#         )

#         self.gnn_tasks_devices = GATConv(
#             (self.in_channels_devices, self.in_channels_tasks),
#             hidden_channels,
#             heads=self.n_heads,
#             concat=True,
#             dropout=0.0,
#             edge_dim=self.task_device_edge_dim,
#             add_self_loops=False,
#         )

#         # The output dimension is (hidden_channels * n_heads) * 2
#         self.gnn_tasks_tasks = GATConv(
#             hidden_channels * self.n_heads * 2,
#             hidden_channels * 2,  # Adjusted to maintain dimensionality
#             heads=self.n_heads,
#             concat=True,
#             dropout=0.0,
#             edge_dim=self.task_task_edge_dim,
#             add_self_loops=True,
#         )

#         # Batch normalization layers
#         self.batch_norm1 = nn.BatchNorm1d(hidden_channels * self.n_heads)
#         self.batch_norm2 = nn.BatchNorm1d(hidden_channels * self.n_heads)
#         self.batch_norm3 = nn.BatchNorm1d(hidden_channels * self.n_heads * 2)

#         # Activation function
#         self.activation = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, data):
#         # Process data to tasks
#         data_fused_tasks = self.gnn_tasks_data(
#             (data["data"].x, data["tasks"].x),
#             data["data", "used_by", "tasks"].edge_index,
#             data["data", "used_by", "tasks"].edge_attr,
#         )
#         data_fused_tasks = self.batch_norm1(data_fused_tasks)
#         data_fused_tasks = self.activation(data_fused_tasks)

#         # Process devices to tasks
#         device_fused_tasks = self.gnn_tasks_devices(
#             (data["devices"].x, data["tasks"].x),
#             data["devices", "variant", "tasks"].edge_index,
#             data["devices", "variant", "tasks"].edge_attr,
#         )
#         device_fused_tasks = self.batch_norm2(device_fused_tasks)
#         device_fused_tasks = self.activation(device_fused_tasks)

#         # Concatenate the processed features
#         tasks = torch.cat([data_fused_tasks, device_fused_tasks], dim=1)
#         residual = tasks  # Residual connection

#         # Process tasks to tasks
#         x = self.gnn_tasks_tasks(
#             tasks,
#             data["tasks", "depends_on", "tasks"].edge_index,
#             data["tasks", "depends_on", "tasks"].edge_attr,
#         )
#         x = x + residual  # Add residual connection
#         x = self.batch_norm3(x)
#         x = self.activation(x)
#         x = torch.dropout(x, p=0.1, train=self.training)

#         return x


# # class TaskDeviceAssignment(nn.Module):

# #     def __init__(self, ndevices, priority_levels, hidden_channels, data):
# #         super(TaskAssignmentNet, self).__init__()

# #         self.hetero_gat = HeteroGAT(hidden_channels, data)
# #         self.ndevices = ndevices
# #         self.critic_fc = nn.Linear(hidden_channels, hidden_channels)
# #         self.critic_fc2 = nn.Linear(hidden_channels, 1)

# #         self.fc = nn.Linear(hidden_channels, hidden_channels)
# #         self.fcd = nn.Linear(hidden_channels, ndevices)

# #     def forward(self, data, task_batch=None):
# #         x = self.hetero_gat(data)

# #         # Critic Head
# #         xc = torch.relu(self.critic_fc(x))
# #         xc = torch.dropout(xc, p=0.1, train=self.training)
# #         v = self.critic_fc2(xc)
# #         v = global_mean_pool(v, task_batch)

# #         # Actor Heads
# #         x = x[: data["candidate_list"].shape[0]]
# #         x = torch.relu(self.fc(x))
# #         x = torch.dropout(x, p=0.1, train=self.training)
# #         # Output logits for device assignment
# #         d = torch.relu(self.fcd(x))

# #         return d, v


# # def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
# #     torch.nn.init.orthogonal_(layer.weight, std)
# #     torch.nn.init.constant_(layer.bias, bias_const)
# #     return layer


# # class TaskAssignmentNet(nn.Module):

# #     def __init__(self, ndevices, priority_levels, hidden_channels, data):
# #         super(TaskAssignmentNet, self).__init__()

# #         self.hetero_gat = HeteroGAT(hidden_channels, data)
# #         self.ndevices = ndevices
# #         self.critic_fc = layer_init(nn.Linear(hidden_channels*8, hidden_channels))
# #         self.critic_fc2 = layer_init(nn.Linear(hidden_channels, 1))

# #         self.batch_norm = nn.BatchNorm1d(hidden_channels)
# #         self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
# #         self.batch_norm3 = nn.BatchNorm1d(hidden_channels)
# #         self.batch_norm4 = nn.BatchNorm1d(hidden_channels)
# #         self.batch_norm5 = nn.BatchNorm1d(hidden_channels)

# #         self.fc = nn.Linear(hidden_channels, hidden_channels)
# #         self.fc2 = nn.Linear(hidden_channels*8, hidden_channels)
# #         self.fc3 = nn.Linear(hidden_channels*8, hidden_channels)
# #         self.fcp = nn.Linear(hidden_channels, priority_levels)
# #         self.fcd = nn.Linear(hidden_channels, ndevices)

# #     def forward(self, data, task_batch=None):
# #         x = self.hetero_gat(data)

# #         # Critic Head
# #         xc = torch.relu(self.batch_norm(self.critic_fc(x)))
# #         xc = torch.dropout(xc, p=0.1, train=self.training)
# #         v = self.critic_fc2(xc)
# #         v = global_mean_pool(v, task_batch)

# #         # Actor Heads
# #         x = x[: data["candidate_list"].shape[0]]
# #         # x = torch.relu(self.batch_norm2(self.fc(x)))
# #         # x = torch.dropout(x, p=0.1, train=self.training)

# #         x2 = torch.relu(self.batch_norm4(self.fc2(x)))
# #         p = self.fcp(x2)

# #         x3 = torch.relu(self.batch_norm5(self.fc3(x)))
# #         d = self.fcd(x3)

# #         return p, d, v

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.nn import GATConv, global_mean_pool
# from torch_geometric.data import HeteroData
# from torch.nn import functional as F
# import numpy as np

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

# class HeteroGAT(nn.Module):
#     def __init__(self, hidden_channels, data):
#         super(HeteroGAT, self).__init__()

#         # Extract feature sizes
#         self.in_channels_tasks = data["tasks"].x.shape[1]
#         self.in_channels_data = data["data"].x.shape[1]
#         self.in_channels_devices = data["devices"].x.shape[1]

#         # Extract edge feature dimensions
#         self.task_data_edge_dim = data[("data", "used_by", "tasks")].edge_attr.shape[1]
#         self.task_device_edge_dim = data[("devices", "variant", "tasks")].edge_attr.shape[1]
#         self.task_task_edge_dim = data[("tasks", "depends_on", "tasks")].edge_attr.shape[1]

#         self.n_heads = 4

#         # Define GATConv layers
#         self.gnn_tasks_data = GATConv(
#             (self.in_channels_data, self.in_channels_tasks),
#             hidden_channels,
#             heads=self.n_heads,
#             concat=True,
#             dropout=0.0,
#             edge_dim=self.task_data_edge_dim,
#             add_self_loops=False,
#         )

#         self.gnn_tasks_devices = GATConv(
#             (self.in_channels_devices, self.in_channels_tasks),
#             hidden_channels,
#             heads=self.n_heads,
#             concat=True,
#             dropout=0.0,
#             edge_dim=self.task_device_edge_dim,
#             add_self_loops=False,
#         )

#         # The output dimension is (hidden_channels * n_heads) * 2
#         self.gnn_tasks_tasks = GATConv(
#             hidden_channels * self.n_heads * 2,
#             hidden_channels * 2,
#             heads=self.n_heads,
#             concat=True,
#             dropout=0.0,
#             edge_dim=self.task_task_edge_dim,
#             add_self_loops=True,
#         )

#         # Batch normalization layers
#         self.batch_norm1 = nn.BatchNorm1d(hidden_channels * self.n_heads)
#         self.batch_norm2 = nn.BatchNorm1d(hidden_channels * self.n_heads)
#         self.batch_norm3 = nn.BatchNorm1d(hidden_channels * self.n_heads * 2)

#         # Activation function
#         self.activation = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, data):
#         # Process data to tasks
#         data_fused_tasks = self.gnn_tasks_data(
#             (data["data"].x, data["tasks"].x),
#             data["data", "used_by", "tasks"].edge_index,
#             data["data", "used_by", "tasks"].edge_attr,
#         )
#         data_fused_tasks = data_fused_tasks
#         data_fused_tasks = self.batch_norm1(data_fused_tasks)
#         data_fused_tasks = self.activation(data_fused_tasks)

#         # Process devices to tasks
#         device_fused_tasks = self.gnn_tasks_devices(
#             (data["devices"].x, data["tasks"].x),
#             data["devices", "variant", "tasks"].edge_index,
#             data["devices", "variant", "tasks"].edge_attr,
#         )
#         device_fused_tasks = self.batch_norm2(device_fused_tasks)
#         device_fused_tasks = self.activation(device_fused_tasks)

#         # Concatenate the processed features
#         tasks = torch.cat([data_fused_tasks, device_fused_tasks], dim=1)
#         residual = tasks  # Residual connection

#         # Process tasks to tasks
#         x = self.gnn_tasks_tasks(
#             tasks,
#             data["tasks", "depends_on", "tasks"].edge_index,
#             data["tasks", "depends_on", "tasks"].edge_attr,
#         )
#         x = x + residual  # Add residual connection
#         x = self.batch_norm3(x)
#         x = self.activation(x)
#         # x = torch.dropout(x, p=0.1, train=self.training)

#         return x

# class ActorCriticHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(ActorCriticHead, self).__init__()
#         self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.activation = nn.LeakyReLU(negative_slope=0.01)
#         self.fc2 = layer_init(nn.Linear(hidden_dim, output_dim))

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.activation(x)
#         # x = torch.dropout(x, p=0.1, train=self.training)
#         x = self.fc2(x)
#         return x

# class TaskAssignmentNet(nn.Module):
#     def __init__(self, ndevices, priority_levels, hidden_channels, data):
#         super(TaskAssignmentNet, self).__init__()

#         # Initialize HeteroGAT
#         self.hetero_gat = HeteroGAT(hidden_channels, data)
#         self.ndevices = ndevices
#         self.priority_levels = priority_levels

#         # Critic Head
#         critic_input_dim = hidden_channels * 8  # Adjust based on HeteroGAT output
#         self.critic_head = ActorCriticHead(critic_input_dim, hidden_channels, 1)

#         # Actor Head for Priority
#         self.actor_p_head = ActorCriticHead(critic_input_dim, hidden_channels, priority_levels)

#         # Actor Head for Device Assignment
#         self.actor_d_head = ActorCriticHead(critic_input_dim, hidden_channels, ndevices)

#     def forward(self, data, task_batch=None):
#         # Get features from HeteroGAT
#         x = self.hetero_gat(data)

#         # Critic Head
#         v = self.critic_head(x)
#         v = global_mean_pool(v, task_batch)

#         # Actor Head for Priority
#         p_logits = self.actor_p_head(x)

#         # Actor Head for Device Assignment
#         d_logits = self.actor_d_head(x)

#         return p_logits, d_logits, v

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import numpy as np


def layer_init(layer, a=0.01, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight, a=a, nonlinearity="leaky_relu")
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, GATConv):
            layer_init(m)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm weights and biases
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class HeteroGAT(nn.Module):
    def __init__(self, hidden_channels, data):
        super(HeteroGAT, self).__init__()

        self.in_channels_tasks = data["tasks"].x.shape[1]
        self.in_channels_data = data["data"].x.shape[1]
        self.in_channels_devices = data["devices"].x.shape[1]

        self.task_data_edge_dim = data[("data", "used_by", "tasks")].edge_attr.shape[1]
        self.task_device_edge_dim = data[
            ("devices", "variant", "tasks")
        ].edge_attr.shape[1]
        self.task_task_edge_dim = data[
            ("tasks", "depends_on", "tasks")
        ].edge_attr.shape[1]

        self.n_heads = 2

        self.gnn_tasks_data = GATConv(
            (self.in_channels_data, self.in_channels_tasks),
            hidden_channels,
            heads=self.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=self.task_data_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_devices = GATConv(
            (self.in_channels_devices, self.in_channels_tasks),
            hidden_channels,
            heads=self.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=self.task_device_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATConv(
            self.in_channels_tasks,
            hidden_channels,
            heads=self.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=self.task_task_edge_dim,
            add_self_loops=True,
        )

        self.linear = nn.Linear(
            (hidden_channels * 3),
            self.in_channels_tasks,
        )

        # Layer normalization layers
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        # self.batch_norm1 = nn.BatchNorm1d(hidden_channels)

        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        # self.batch_norm2 = nn.BatchNorm1d(hidden_channels)

        # self.layer_norm3 = nn.LayerNorm(self.in_channels_tasks * self.n_heads)
        self.layer_norm4 = nn.LayerNorm(hidden_channels)
        # self.batch_norm4 = nn.BatchNorm1d(hidden_channels)

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data):
        # Process data to tasks

        data_fused_tasks = self.gnn_tasks_data(
            (data["data"].x, data["tasks"].x),
            data["data", "used_by", "tasks"].edge_index,
            data["data", "used_by", "tasks"].edge_attr,
        )

        # Check device placements for all
        # print(f"data_fused_tasks device: {data_fused_tasks.device}")

        data_fused_tasks = self.layer_norm1(data_fused_tasks)
        # print(f"data_fused_tasks after layer_norm1 device: {data_fused_tasks.device}")
        data_fused_tasks = self.activation(data_fused_tasks)
        # print(f"data_fused_tasks after activation device: {data_fused_tasks.device}")

        # Process devices to tasks
        device_fused_tasks = self.gnn_tasks_devices(
            (data["devices"].x, data["tasks"].x),
            data["devices", "variant", "tasks"].edge_index,
            data["devices", "variant", "tasks"].edge_attr,
        )
        device_fused_tasks = self.layer_norm2(device_fused_tasks)
        device_fused_tasks = self.activation(device_fused_tasks)

        task_fused_tasks = self.gnn_tasks_tasks(
            data["tasks"].x,
            data["tasks", "depends_on", "tasks"].edge_index,
            data["tasks", "depends_on", "tasks"].edge_attr,
        )
        task_fused_tasks = self.layer_norm4(task_fused_tasks)
        task_fused_tasks = self.activation(task_fused_tasks)

        # Concatenate the processed feature
        x = torch.cat(
            [data["tasks"].x, task_fused_tasks, data_fused_tasks, device_fused_tasks],
            dim=1,
        )

        # x = self.linear(tasks)
        # x = self.layer_norm3(x)
        # x = self.activation(x)

        # x = torch.dropout(x, p=0.1, train=self.training)

        return x


class HeteroVec(nn.Module):
    def __init__(self, hidden_channels, data):
        super(HeteroVec, self).__init__()

        self.hidden_channels = hidden_channels

        self.in_channels_tasks = data["tasks"].x.shape[1]
        self.in_channels_data = data["data"].x.shape[1]
        self.in_channels_devices = data["devices"].x.shape[1]

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.layer_norm_output = nn.LayerNorm(hidden_channels)
        self.linear_output = nn.Linear(
            (self.in_channels_tasks + self.in_channels_devices * 4),
            self.hidden_channels,
        )

    def forward(self, data):
        # Get subgraph for candidate task

        subset_dict = {"tasks": torch.tensor([0])}

        subgraph = data.subgraph(subset_dict)

        candidate_data = subgraph["data"].x[
            subgraph[("data", "used_by", "tasks")].edge_index[0]
        ]

        candidate_devices = subgraph["devices"].x[
            subgraph[("devices", "variant", "tasks")].edge_index[0]
        ]
        candidate_task = subgraph["tasks"].x

        flattened_task = torch.flatten(candidate_task)
        flattened_devices = torch.flatten(candidate_devices)

        x = torch.cat([flattened_task, flattened_devices], dim=0)

        x = self.linear_output(x)
        x = self.layer_norm_output(x)
        x = self.activation(x)

        return x


class ActorCriticHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCriticHead, self).__init__()
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        # self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = layer_init(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        # x = torch.dropout(x, p=0.1, train=self.training)
        x = self.fc2(x)
        return x


class VectorTaskAssignmentNet(nn.Module):
    def __init__(self, ndevices, priority_levels, hidden_channels, data):
        super(VectorTaskAssignmentNet, self).__init__()

        self.in_channels_tasks = data["tasks"].x.shape[1]
        self.in_channels_data = data["data"].x.shape[1]
        self.in_channels_devices = data["devices"].x.shape[1]

        self.hetero_vec = HeteroVec(hidden_channels, data)
        self.ndevices = ndevices
        self.priority_levels = priority_levels
        self.n_heads = 3

        # input dimension
        critic_input_dim = hidden_channels * 3 + self.in_channels_tasks
        actor_input_dim = hidden_channels * 3 + self.in_channels_tasks

        # Critic Head
        self.critic_head = ActorCriticHead(critic_input_dim, hidden_channels, 1)

        # Actor Head for Priority
        self.actor_p_head = ActorCriticHead(
            actor_input_dim, hidden_channels, priority_levels
        )

        # Actor Head for Device Assignment
        self.actor_d_head = ActorCriticHead(actor_input_dim, hidden_channels, ndevices)

    def forward(self, data, task_batch=None):
        # Get features from HeteroGAT
        x = self.hetero_vec(data)

        # Critic Head
        v = self.critic_head(x)
        v = global_mean_pool(v, task_batch)

        z = data[: len(data["candidate_list"])]

        # Actor Head for Priority
        p_logits = self.actor_p_head(z)

        # Actor Head for Device Assignment
        d_logits = self.actor_d_head(z)

        return p_logits, d_logits, v


class TaskAssignmentNet(nn.Module):
    def __init__(self, ndevices, priority_levels, hidden_channels, data, device):
        super(TaskAssignmentNet, self).__init__()

        self.in_channels_tasks = data["tasks"].x.shape[1]
        self.in_channels_data = data["data"].x.shape[1]
        self.in_channels_devices = data["devices"].x.shape[1]

        self.task_data_edge_dim = data[("data", "used_by", "tasks")].edge_attr.shape[1]
        self.task_device_edge_dim = data[
            ("devices", "variant", "tasks")
        ].edge_attr.shape[1]
        self.task_task_edge_dim = data[
            ("tasks", "depends_on", "tasks")
        ].edge_attr.shape[1]

        self.hetero_gat_actor = HeteroGAT(hidden_channels, data).to(device)
        self.hetero_gat_critic = HeteroGAT(hidden_channels, data).to(device)
        self.ndevices = ndevices
        self.priority_levels = priority_levels

        # input dimension
        critic_input_dim = hidden_channels * 3 + self.in_channels_tasks
        actor_input_dim = hidden_channels * 3 + self.in_channels_tasks

        # Critic Head
        self.critic_head = ActorCriticHead(critic_input_dim, hidden_channels, 1).to(
            device
        )

        # Actor Head for Priority
        self.actor_p_head = ActorCriticHead(
            critic_input_dim, hidden_channels, priority_levels
        ).to(device)

        # Actor Head for Device Assignment
        self.actor_d_head = ActorCriticHead(
            actor_input_dim, hidden_channels, ndevices
        ).to(device)

    def forward(self, data, task_batch=None):
        # Get features from HeteroGAT
        x = self.hetero_gat_critic(data)

        # Critic Head
        v = self.critic_head(x)
        v = global_mean_pool(v, task_batch)

        z = self.hetero_gat_actor(data)

        if task_batch is not None:
            z = z[data["tasks"].ptr[:-1]]
        else:
            z = z[0]

        # Actor Head for Priority
        p_logits = self.actor_p_head(z)

        # Actor Head for Device Assignment
        d_logits = self.actor_d_head(z)

        return p_logits, d_logits, v


class TaskAssignmentNetDeviceOnly(nn.Module):
    def __init__(self, ndevices, hidden_channels, data):
        super(TaskAssignmentNetDeviceOnly, self).__init__()

        self.in_channels_tasks = data["tasks"].x.shape[1]
        self.in_channels_data = data["data"].x.shape[1]
        self.in_channels_devices = data["devices"].x.shape[1]

        self.task_data_edge_dim = data[("data", "used_by", "tasks")].edge_attr.shape[1]
        self.task_device_edge_dim = data[
            ("devices", "variant", "tasks")
        ].edge_attr.shape[1]
        self.task_task_edge_dim = data[
            ("tasks", "depends_on", "tasks")
        ].edge_attr.shape[1]

        self.hetero_gat_actor = HeteroGAT(hidden_channels, data)
        self.hetero_gat_critic = HeteroGAT(hidden_channels, data)
        self.ndevices = ndevices

        # input dimension
        critic_input_dim = hidden_channels * 3 + self.in_channels_tasks
        actor_input_dim = hidden_channels * 3 + self.in_channels_tasks

        # Critic Head
        self.critic_head = ActorCriticHead(critic_input_dim, hidden_channels, 1)

        # Actor Head for Device Assignment
        self.actor_d_head = ActorCriticHead(actor_input_dim, hidden_channels, ndevices)

    def forward(self, data, task_batch=None):
        # Get features from HeteroGAT
        x = self.hetero_gat_critic(data)

        # Critic Head
        v = self.critic_head(x)
        v = global_mean_pool(v, task_batch)

        z = self.hetero_gat_actor(data)

        if task_batch is not None:
            z = z[data["tasks"].ptr[:-1]]
        else:
            z = z[0]

        # Actor Head for Device Assignment
        d_logits = self.actor_d_head(z)

        return d_logits, v
