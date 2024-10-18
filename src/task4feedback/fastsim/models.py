import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, TransformerConv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import global_mean_pool


class HeteroGAT(nn.Module):

    def __init__(self, hidden_channels, data):
        super(HeteroGAT, self).__init__()

        metadata = data.metadata()
        self.convs = nn.ModuleList()

        # Task feature size
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

        self.n_heads = 4

        # Data - Tasks with edge features
        self.gnn_tasks_data = GATConv(
            (self.in_channels_data, self.in_channels_tasks),
            hidden_channels,
            heads=self.n_heads,
            concat=True,
            dropout=0.0,
            edge_dim=self.task_data_edge_dim,
            add_self_loops=False,
        )

        gnn_tasks_data_dim = hidden_channels * self.n_heads
        gnn_tasks_devices_dim = hidden_channels * self.n_heads

        self.gnn_tasks_devices = GATConv(
            (self.in_channels_devices, self.in_channels_tasks),
            hidden_channels,
            heads=self.n_heads,
            concat=True,
            dropout=0.0,
            edge_dim=self.task_device_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATConv(
            gnn_tasks_data_dim + gnn_tasks_devices_dim,
            hidden_channels,
            heads=self.n_heads,
            concat=False,
            dropout=0,
            edge_dim=self.task_task_edge_dim,
            add_self_loops=True,
        )

    def forward(self, data):
        data_fused_tasks = self.gnn_tasks_data(
            (data["data"].x, data["tasks"].x),
            data["data", "used_by", "tasks"].edge_index,
            data["data", "used_by", "tasks"].edge_attr,
        )
        data_fused_tasks = torch.relu(data_fused_tasks)

        device_fused_tasks = self.gnn_tasks_devices(
            (data["devices"].x, data["tasks"].x),
            data["devices", "variant", "tasks"].edge_index,
            data["devices", "variant", "tasks"].edge_attr,
        )
        device_fused_tasks = torch.relu(device_fused_tasks)

        tasks = torch.cat([data_fused_tasks, device_fused_tasks], dim=1)

        x = self.gnn_tasks_tasks(
            tasks,
            data["tasks", "depends_on", "tasks"].edge_index,
            data["tasks", "depends_on", "tasks"].edge_attr,
        )
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)

        return x


class TaskAssignmentNet(nn.Module):

    def __init__(self, ndevices, priority_levels, hidden_channels, data):
        super(TaskAssignmentNet, self).__init__()

        self.hetero_gat = HeteroGAT(hidden_channels, data)
        self.ndevices = ndevices
        self.critic_fc = nn.Linear(hidden_channels, hidden_channels)
        self.critic_fc2 = nn.Linear(hidden_channels, 1)

        self.fc = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fcp = nn.Linear(hidden_channels, priority_levels)
        self.fcd = nn.Linear(hidden_channels, ndevices)

    def forward(self, data, task_batch=None):
        x = self.hetero_gat(data)

        # Critic Head
        xc = torch.relu(self.critic_fc(x))
        xc = torch.dropout(xc, p=0.1, train=self.training)
        v = self.critic_fc2(xc)
        v = global_mean_pool(v, task_batch)

        # Actor Heads
        x = x[: data["candidate_list"].shape[0]]
        x = torch.relu(self.fc(x))
        x = torch.dropout(x, p=0.1, train=self.training)
        x2 = torch.relu(self.fc2(x))
        x2 = torch.dropout(x2, p=0.2, train=self.training)
        x3 = torch.relu(self.fc3(x))
        x3 = torch.dropout(x3, p=0.2, train=self.training)
        p = torch.relu(self.fcp(x2))
        d = torch.relu(self.fcd(x3))

        return p, d, v
