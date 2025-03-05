from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
from task4feedback.legacy_graphs import *
import torch
from typing import Optional, Self

from torchrl.envs import EnvBase
from task4feedback.interface.wrappers import (
    DefaultObserverFactory,
    CompiledDefaultObserverFactory,
    SimulatorDriver,
    SimulatorFactory,
    create_graph_spec,
    start_logger,
    observation_to_heterodata,
    observation_to_heterodata_truncate,
)

# from task4feedback.interface.wrappers import (
#     observation_to_heterodata_truncate as observation_to_heterodata,
# )
from torchrl.data import Composite, TensorSpec, Unbounded, Binary, Bounded
from torchrl.envs.utils import make_composite_from_td
from tensordict.nn import set_composite_lp_aggregate
from torchrl.envs import check_env_specs
from tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, HeteroConv
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import StepCounter, TrajCounter, TransformedEnv
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor
import tensordict
from tensordict.nn import (
    TensorDictModule,
    ProbabilisticTensorDictModule,
    TensorDictSequential,
)
import torchrl
import torch_geometric
import aim
from aim.pytorch import track_gradients_dists, track_params_dists
# start_logger()


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


def make_test_cholesky_graph():
    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        placement_info.add(
            (Device(Architecture.GPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )
        placement_info.add(
            (Device(Architecture.CPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=1),
        )

        return placement_info

    data_config = CholeskyDataGraphConfig(data_size=1000000)
    config = CholeskyConfig(blocks=4, task_config=task_config)
    tasks, data = make_graph(config, data_config=data_config)
    return tasks, data


class FastSimEnv(EnvBase):
    def __init__(self, simulator_factory, seed: int = 0, device="cpu"):
        super().__init__(device=device)

        self.simulator_factory = simulator_factory
        self.simulator = simulator_factory.create(seed)

        # self.time_spec = Unbounded(shape=(1,), device=self.device, dtype=torch.int64)
        self.observation_spec = self._create_observation_spec()
        self.action_spec = self._create_action_spec()
        self.reward_spec = self._create_reward_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

        self.workspace = self.simulator.observer.new_observation_buffer()

    def _get_baseline(self):
        # simulator_copy = self.simulator.fresh_copy()
        # simulator_copy.initialize()
        # simulator_copy.initialize_data()
        # simulator_copy.disable_external_mapper()
        # final_state = simulator_copy.run()
        # assert final_state == fastsim.ExecutionState.COMPLETE, (
        #     f"Baseline returned unexpected final state: {final_state}"
        # )
        # return simulator_copy.time()
        return 10000

    def _create_observation_spec(self) -> TensorSpec:
        obs = self.simulator.observer.get_observation()
        comp = make_composite_from_td(obs)
        comp = Composite(observation=comp)
        return comp

    def _create_action_spec(self, ndevices: int = 5) -> TensorSpec:
        n_devices = self.simulator_factory.graph_spec.max_devices
        out = Bounded(
            shape=(1,),
            device=self.device,
            dtype=torch.int64,
            low=torch.tensor(0, device=self.device),
            high=torch.tensor(n_devices, device=self.device),
        )
        out = Composite(action=out)
        return out

    def _create_reward_spec(self) -> TensorSpec:
        return Unbounded(shape=(1,), device=self.device, dtype=torch.float32)

    def _get_observation(self) -> TensorDict:
        obs = self.simulator.observer.get_observation()
        td = TensorDict(observation=obs)
        return td

    def _step(self, td: TensorDict) -> TensorDict:
        chosen_device = td["action"].item()
        local_id = 0
        device = chosen_device
        state = self.simulator.get_state()
        mapping_priority = 0
        reserving_priority = mapping_priority
        launching_priority = mapping_priority
        actions = [
            fastsim.Action(local_id, device, reserving_priority, launching_priority)
        ]
        self.simulator.simulator.map_tasks(actions)
        simulator_status = self.simulator.run_until_external_mapping()

        # terminated = torch.tensor((1,), device=self.device, dtype=torch.bool)
        done = torch.tensor((1,), device=self.device, dtype=torch.bool)
        reward = torch.tensor((1,), device=self.device, dtype=torch.float32)
        # time = torch.tensor((1,), device=self.device, dtype=torch.int64)

        # terminated[0] = simulator_status == fastsim.ExecutionState.COMPLETE
        done[0] = simulator_status == fastsim.ExecutionState.COMPLETE
        reward[0] = 0

        obs = self._get_observation()
        time = obs["observation"]["aux"]["time"].item()

        if not done:
            assert simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING, (
                f"Unexpected simulator status: {simulator_status}"
            )
        else:
            obs = self._reset()
            baseline_time = self._get_baseline()
            print(f"Baseline time: {baseline_time}")
            print(f"Simulator time: {time}")
            reward[0] = 1 + (baseline_time - time) / baseline_time

        out = obs
        out.set("reward", reward)
        out.set("done", done)
        # out.set("time", torch.tensor([time], device=self.device, dtype=torch.int64))
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        self.simulator = self.simulator_factory.create()
        simulator_status = self.simulator.run_until_external_mapping()
        assert simulator_status == fastsim.ExecutionState.EXTERNAL_MAPPING, (
            f"Unexpected simulator status: {simulator_status}"
        )

        obs = self._get_observation()
        # obs.set("time", obs["observation"]["aux"]["time"])
        # print("Reset: ", obs["observation"]["aux"]["candidates"]["idx"])
        # print("Reset: ", obs["observation"]["nodes"]["tasks"]["glb"])

        return obs

    def _set_seed(self, seed: Optional[int] = None):
        pass


def make_env():
    s = uniform_connected_devices(5, 1000000000, 1, 2000)
    tasks, data = make_test_cholesky_graph()
    d = DataBlocks.create_from_legacy_data(data, s)
    m = Graph.create_from_legacy_graph(tasks, data)
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)
    print(f"Max devices: {spec.max_devices}")
    print(f"N tasks: {len(tasks)}")
    print(f"N data: {len(data)}")
    env = FastSimEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory), device="cpu"
    )
    return TransformedEnv(
        env,
        StepCounter(),
        TrajCounter(),
    )


@dataclass
class HeteroGATConfig:
    task_feature_dim: int = 12
    data_feature_dim: int = 5
    device_feature_dim: int = 12
    task_data_edge_dim: int = 3
    task_device_edge_dim: int = 2
    task_task_edge_dim: int = 1
    hidden_channels: int = 64
    n_heads: int = 1

    @staticmethod
    def from_observer(
        observer: ExternalObserver, n_heads: int = 1, hidden_channels: int = 16
    ):
        return HeteroGATConfig(
            task_feature_dim=observer.task_feature_dim,
            data_feature_dim=observer.data_feature_dim,
            device_feature_dim=observer.device_feature_dim,
            task_data_edge_dim=observer.task_data_edge_dim,
            task_device_edge_dim=observer.task_device_edge_dim,
            task_task_edge_dim=observer.task_task_edge_dim,
            hidden_channels=hidden_channels,
            n_heads=n_heads,
        )


class DatatoTaskLayer(nn.Module):
    """
    Performs a single GAT convolution from data features onto task features.
    """

    def __init__(self, config: HeteroGATConfig):
        super(DatatoTaskLayer, self).__init__()
        self.config = config
        self.gnn_tasks_data = GATConv(
            (config.data_feature_dim, config.task_feature_dim),
            config.hidden_channels,
            heads=config.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=config.task_data_edge_dim,
            add_self_loops=False,
        )
        self.layer_norm_data_tasks = nn.LayerNorm(config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, obs: HeteroData) -> torch.Tensor:
        data_fused_tasks = self.gnn_tasks_data(
            (obs["data"].x, obs["tasks"].x),
            obs["data", "uses", "tasks"].edge_index,
            obs["data", "uses", "tasks"].edge_attr,
        )
        data_fused_tasks = self.layer_norm_data_tasks(data_fused_tasks)
        data_fused_tasks = self.activation(data_fused_tasks)
        return data_fused_tasks


class DeprecatedHeteroGAT(nn.Module):
    def __init__(self, n_heads: int, config: HeteroGATConfig):
        super(DeprecatedHeteroGAT, self).__init__()
        self.config = config

        self.conv_1 = HeteroConv(
            {
                ("data", "to", "tasks"): GATConv(
                    (config.data_feature_dim, config.task_feature_dim),
                    config.hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_data_edge_dim,
                    add_self_loops=False,
                ),
                ("tasks", "to", "tasks"): GATConv(
                    (config.task_feature_dim, config.task_feature_dim),
                    config.hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_task_edge_dim,
                    add_self_loops=False,
                ),
                ("devices", "to", "tasks"): GATConv(
                    (config.device_feature_dim, config.task_feature_dim),
                    config.hidden_channels,
                    heads=n_heads,
                    concat=False,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_device_edge_dim,
                    add_self_loops=False,
                ),
            },
            aggr="cat",
        )

        # self.linear = nn.Linear(config.hidden_channels * 3 + config.task_feature_dim, config.task_feature_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        self.layernorm_post_gat = nn.LayerNorm(
            config.task_feature_dim + config.hidden_channels * 3
        )
        # self.layernorm_post_linear = nn.LayerNorm(config.task_feature_dim)

    def forward(self, data):
        tasks = data.x_dict["tasks"]
        x_dict = self.conv_1(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        x = x_dict["tasks"]
        x = torch.cat([tasks, x], dim=-1)
        x = self.layernorm_post_gat(x)
        x = self.activation(x)
        # x = self.linear(x)
        # x = self.layernorm_post_linear(x)
        # x = self.activation(x)
        return x


class DataTaskBipartiteLayer(nn.Module):
    def __init__(self, n_heads: int, config: HeteroGATConfig):
        super(DataTaskBipartiteLayer, self).__init__()
        self.config = config
        self.data_task_conv = HeteroConv(
            {
                ("data", "to", "tasks"): GATConv(
                    (config.data_feature_dim, config.task_feature_dim),
                    config.hidden_channels,
                    heads=n_heads,
                    concat=True,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_data_edge_dim,
                    add_self_loops=False,
                ),
                ("tasks", "to", "data"): GATConv(
                    (config.task_feature_dim, config.data_feature_dim),
                    config.hidden_channels,
                    heads=n_heads,
                    concat=True,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_data_edge_dim,
                    add_self_loops=False,
                ),
            }
        )

        hidden_channels_with_heads = config.hidden_channels * n_heads

        self.task_data_conv = HeteroConv(
            {
                ("tasks", "to", "data"): GATConv(
                    (hidden_channels_with_heads, hidden_channels_with_heads),
                    config.hidden_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_data_edge_dim,
                ),
                ("data", "to", "tasks"): GATConv(
                    (hidden_channels_with_heads, hidden_channels_with_heads),
                    config.hidden_channels,
                    heads=1,
                    concat=False,
                    add_self_loops=False,
                    residual=True,
                    dropout=0,
                    edge_dim=config.task_data_edge_dim,
                ),
            }
        )

        self.norm_tasks = nn.LayerNorm(config.hidden_channels)
        self.norm_data = nn.LayerNorm(config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, data: HeteroData | Batch):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = (
            data.edge_attr_dict if hasattr(data, "edge_attr_dict") else None
        )

        # print(x_dict)
        # print(edge_index_dict)
        # print(edge_attr_dict)

        x_dict = self.data_task_conv(x_dict, edge_index_dict, edge_attr_dict)

        x_dict = {node_type: self.activation(x) for node_type, x in x_dict.items()}
        x_dict = {
            "tasks": self.norm_tasks(x_dict["tasks"]),
            "data": self.norm_data(x_dict["data"]),
        }
        x_dict = self.task_data_conv(x_dict, edge_index_dict, edge_attr_dict)

        return x_dict


class TaskTaskLayer(nn.Module):
    """
    Tasks-to-tasks encodes task -> dependency information

    This module performs two depth 2 GAT convolutions on the task nodes:
    - Two accumulations of task -> dependency information
    - Two accumulations of task -> dependant information

    These results are then concatenated and returned as the output (hidden_channels * 2)
    """

    def __init__(self, input_dim: int, n_heads: int, config: HeteroGATConfig):
        super(TaskTaskLayer, self).__init__()

        self.conv_dependency_1 = GATConv(
            (input_dim, input_dim),
            config.hidden_channels,
            heads=n_heads,
            concat=True,
            residual=True,
            dropout=0,
            edge_dim=config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent_1 = GATConv(
            (input_dim, input_dim),
            config.hidden_channels,
            heads=n_heads,
            concat=True,
            residual=True,
            dropout=0,
            edge_dim=config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.conv_dependency_2 = GATConv(
            (config.hidden_channels * n_heads, config.hidden_channels * n_heads),
            config.hidden_channels,
            heads=1,
            concat=True,
            residual=True,
            dropout=0,
            edge_dim=config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.conv_dependent_2 = GATConv(
            (config.hidden_channels * n_heads, config.hidden_channels * n_heads),
            config.hidden_channels,
            heads=1,
            concat=True,
            residual=True,
            edge_dim=config.task_task_edge_dim,
            add_self_loops=False,
        )

        self.norm_dependency = nn.LayerNorm(config.hidden_channels)
        self.norm_dependant = nn.LayerNorm(config.hidden_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

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
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = layer_init(nn.Linear(hidden_dim, output_dim))
        # self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(x.shape)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # print(x.shape)
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


class DeprecatedDeviceAssignmentNet(nn.Module):
    def __init__(
        self,
        config: HeteroGATConfig,
        n_devices: int,
    ):
        super(DeprecatedDeviceAssignmentNet, self).__init__()
        self.config = config

        self.hetero_gat = DeprecatedHeteroGAT(config.n_heads, config)
        gat_output_dim = config.hidden_channels * 3 + config.task_feature_dim
        self.actor_head = OutputHead(gat_output_dim, config.hidden_channels, n_devices)

    def _is_batch(self, obs: TensorDict) -> bool:
        # print("Batch size: ", obs.batch_size)
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(self, obs: TensorDict) -> HeteroData:
        # print("Counts", obs["nodes"]["tasks"]["count"])
        if not self._is_batch(obs):
            _obs = observation_to_heterodata(obs)
            return _obs

        _h_data = []
        for i in range(obs.batch_size[0]):
            _obs = observation_to_heterodata(obs[i])
            _h_data.append(_obs)

        return Batch.from_data_list(_h_data)

    def forward(self, obs: TensorDict, batch=None):
        data = self._convert_to_heterodata(obs)
        task_embeddings = self.hetero_gat(data)
        is_batch = self._is_batch(obs)

        task_batch = data["tasks"].batch if is_batch else None

        if task_batch is not None:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            candidate_embedding = task_embeddings[0]

        d_logits = self.actor_head(candidate_embedding)

        return d_logits


class DeprecatedValueNet(nn.Module):
    def __init__(self, config: HeteroGATConfig, n_devices: int):
        super(DeprecatedValueNet, self).__init__()
        self.config = config

        self.hetero_gat = DeprecatedHeteroGAT(config.n_heads, config)
        gat_output_dim = config.hidden_channels * 3 + config.task_feature_dim
        self.critic_head = OutputHead(gat_output_dim, config.hidden_channels, 1)

    def _is_batch(self, obs: TensorDict) -> bool:
        print("Batch size: ", obs.batch_size)
        print("Obs0: ", obs[0].batch_size)
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(self, obs: TensorDict) -> HeteroData:
        # print("INPUT TENSOR SHAPE", obs.shape)
        # print("Counts", obs["nodes"]["tasks"]["count"])
        if not self._is_batch(obs):
            # print("NOT BATCH")
            _obs = observation_to_heterodata(obs)
            return _obs

        _h_data = []
        # print("BATCH", obs.batch_size[0])
        for i in range(obs.batch_size[0]):
            _obs = observation_to_heterodata(obs[i], idx=i)
            _h_data.append(_obs)

        return Batch.from_data_list(_h_data)

    def forward(self, obs: TensorDict, batch=None):
        data = self._convert_to_heterodata(obs)
        # print("Length of data: ", len(data))
        task_embeddings = self.hetero_gat(data)

        # print("Task Embedding Shape: ", task_embeddings.shape)
        is_batch = self._is_batch(obs)

        task_batch = data["tasks"].batch if is_batch else None
        v = self.critic_head(task_embeddings)
        v = global_mean_pool(v, task_batch)

        # print(v.shape)
        return v


class DeviceAssignmentNet(nn.Module):
    def __init__(self, config: HeteroGATConfig, n_devices: int = 5):
        super(DeviceAssignmentNet, self).__init__()

        self.config = config

        # Returns embeddings for tasks and data nodes at depth 2
        # Output feature dim:
        # dict of ("tasks": hidden_channels, "data": hidden_channels)
        self.data_task_layer = DataTaskBipartiteLayer(1, config)

        # Returns concatenated embeddings for tasks at depth 2
        # Two directions of task -> task information (dependency and dependant)
        # Output feature dim: hidden_channels * 2
        self.task_task_layer = TaskTaskLayer(config.hidden_channels, 1, config)

        # Combination layer
        self.combine_layer = CombineTwoLayer(
            config.hidden_channels * 2,
            config.hidden_channels,
            config.hidden_channels,
            config.hidden_channels,
        )

        # Output head
        self.output_head = OutputHead(
            config.hidden_channels * 2,
            config.hidden_channels,
            n_devices,
        )

    def _is_batch(self, obs: TensorDict) -> bool:
        # print("Batch size: ", obs.batch_size)
        if not obs.batch_size:
            return False
        return True

    def _convert_to_heterodata(self, obs: TensorDict) -> HeteroData:
        if not self._is_batch(obs):
            _obs = observation_to_heterodata(obs)
            return _obs

        _h_data = []
        for i in range(obs.batch_size[0]):
            _obs = observation_to_heterodata(obs[i])
            _h_data.append(_obs)

        return Batch.from_data_list(_h_data)

    def forward(self, obs: TensorDict, batch=None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs)
        data_task_embeddings = self.data_task_layer(data)
        task_embeddings = data_task_embeddings["tasks"]
        task_embeddings = self.task_task_layer(task_embeddings, data)

        if is_batch:
            candidate_embedding = task_embeddings[data["tasks"].ptr[:-1]]
        else:
            batch = None
            candidate_embedding = task_embeddings[0]

        # task_batch = data["tasks"].batch if is_batch else None
        # data_batch = data["data"].batch if is_batch else None

        # task_pool = global_mean_pool(task_embeddings, task_batch)
        # data_pool = global_mean_pool(data_task_embeddings["data"], data_batch)

        # global_embedding = self.combine_layer(task_pool, data_pool)
        candidate_embedding = (
            candidate_embedding.unsqueeze(0) if not is_batch else candidate_embedding
        )

        # output_embedding = torch.cat([global_embedding, candidate_embedding], dim=-1)
        output_embedding = candidate_embedding

        x = self.output_head(output_embedding)
        return x


class ValueNet(nn.Module):
    def __init__(self, config: HeteroGATConfig, n_devices: int = 5):
        super(ValueNet, self).__init__()

        self.config = config

        # Returns embeddings for tasks and data nodes at depth 2
        # Output feature dim:
        # dict of ("tasks": hidden_channels, "data": hidden_channels)
        self.data_task_layer = DataTaskBipartiteLayer(1, config)

        # Returns concatenated embeddings for tasks at depth 2
        # Two directions of task -> task information (dependency and dependant)
        # Output feature dim: hidden_channels * 2
        self.task_task_layer = TaskTaskLayer(config.hidden_channels, 1, config)

        # Combination layer
        self.combine_layer = CombineTwoLayer(
            config.hidden_channels * 2,
            config.hidden_channels,
            config.hidden_channels,
            config.hidden_channels,
        )

        # Output head
        self.output_head = OutputHead(
            config.hidden_channels,
            config.hidden_channels,
            1,
        )

    def _is_batch(self, obs: TensorDict) -> bool:
        # print("Batch size: ", obs.batch_size)
        if not obs.batch_size:
            return False
        return True

    # def _convert_to_heterodata(self, obs: TensorDict) -> HeteroData:
    #     if not self._is_batch(obs):
    #         _obs = observation_to_heterodata(obs)
    #         return _obs

    #     _h_data = []
    #     for i in range(obs.batch_size[0]):
    #         _obs = observation_to_heterodata(obs[i])
    #         _h_data.append(_obs)

    #     return Batch.from_data_list(_h_data)

    def _convert_to_heterodata(self, obs: TensorDict) -> HeteroData:
        # print("INPUT TENSOR SHAPE", obs.shape)
        # print("Counts", obs["nodes"]["tasks"]["count"])
        if not self._is_batch(obs):
            # print("NOT BATCH")
            _obs = observation_to_heterodata(obs)
            return _obs

        _h_data = []
        # print("BATCH", obs.batch_size[0])
        # print("SHAPE", obs["nodes"]["tasks"]["count"].shape)
        for i in range(obs.batch_size[0]):
            _obs = observation_to_heterodata(obs[i], idx=i)
            _h_data.append(_obs)

        return Batch.from_data_list(_h_data)

    def forward(self, obs: TensorDict, batch=None):
        is_batch = self._is_batch(obs)
        data = self._convert_to_heterodata(obs)
        data_task_embeddings = self.data_task_layer(data)
        task_embeddings = data_task_embeddings["tasks"]
        task_embeddings = self.task_task_layer(task_embeddings, data)

        task_batch = data["tasks"].batch if is_batch else None
        data_batch = data["data"].batch if is_batch else None

        task_pool = global_mean_pool(task_embeddings, task_batch)
        data_pool = global_mean_pool(data_task_embeddings["data"], data_batch)

        global_embedding = self.combine_layer(task_pool, data_pool)

        x = self.output_head(global_embedding)
        return x


if __name__ == "__main__":
    from torchrl.envs import ParallelEnv
    from torchrl.modules import ValueOperator

    t = time.perf_counter()
    workers = 4

    # penv = ParallelEnv(
    #     workers,
    #     [make_env for _ in range(workers)],
    #     use_buffers=True,
    # )
    penv = make_env()

    network_conf = HeteroGATConfig.from_observer(
        penv.simulator.observer, hidden_channels=64, n_heads=2
    )
    action_spec = penv.action_spec

    _internal_policy_module = TensorDictModule(
        DeprecatedDeviceAssignmentNet(network_conf, n_devices=5),
        in_keys=["observation"],
        out_keys=["logits"],
    )

    policy_module = ProbabilisticActor(
        module=_internal_policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        default_interaction_type=tensordict.nn.InteractionType.RANDOM,
        cache_dist=True,
        return_log_prob=True,
    )

    value_module = ValueOperator(
        module=DeprecatedValueNet(network_conf, n_devices=5),
        in_keys=["observation"],
    )

    # _internal_policy_module = torch_geometric.compile(
    #     _internal_policy_module, dynamic=False
    # )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"Number of trainable parameters: {count_parameters(_internal_policy_module)}"
    )

    frames_per_batch = 1000
    subbatch_size = 250
    num_epochs = 4

    # policy_module = torch_geometric.compile(policy_module, dynamic=False)
    # value_module = torch_geometric.compile(value_module, dynamic=False)

    collector = SyncDataCollector(
        make_env,
        policy_module,
        frames_per_batch=frames_per_batch,
        exploration_type=torchrl.envs.utils.ExplorationType.RANDOM,
    )

    # collector = MultiSyncDataCollector(
    #     [make_env for _ in range(workers)],
    #     policy_module,
    #     frames_per_batch=frames_per_batch,
    # )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=1000),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=1, lmbda=1, value_network=value_module, average_gae=False
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coef=0.5,
        loss_critic_type="l2",
    )

    aim_run = aim.Run(experiment="debug-ppo-torchrl")

    optim = torch.optim.Adam(loss_module.parameters(), lr=2.5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, T_max=frames_per_batch, eta_min=0
    # )
    logs = defaultdict(list)

    epoch_idx = 0  # Global counter for tracking steps across iterations

    for i, tensordict_data in enumerate(collector):
        print("Optimization Step: ", i)

        with torch.no_grad():
            # print(tensordict_data["next", "reward"].shape)
            # print(tensordict_data["next", "done"].shape)
            # print(tensordict_data)
            advantage_module(tensordict_data)

        # print("Counts", tensordict_data["observation"]["nodes"]["tasks"]["count"])
        # tensordict_data = tensordict_data.reshape(-1)
        # print(
        #    "Reshaped Counts", tensordict_data["observation"]["nodes"]["tasks"]["count"]
        # )

        # print(tensordict_data["logits"].requires_grad)
        # print(tensordict_data["action"].requires_grad)

        print(tensordict_data["value_target"])
        print(tensordict_data["next", "reward"][tensordict_data["next", "done"]])
        print(tensordict_data["state_value"])

        # print(
        #     "Reward / Cumulative Diff",
        #     tensordict_data["advantage"].view(-1)
        #     - (
        #         tensordict_data["value_target"].view(-1)
        #         - tensordict_data["state_value"].view(-1)
        #     ),
        # )
        print(
            "Advantage Diff",
            tensordict_data["advantage"]
            - (tensordict_data["value_target"] - tensordict_data["state_value"]),
        )

        # print(tensordict_data["advantage"])
        # print(tensordict_data["state_value"])
        # # print(tensordict_data["traj_count"])
        # print(tensordict_data["next", "step_count"])
        # print(tensordict_data["next", "done"])
        # print(tensordict_data["done"])
        # import sys

        # sys.exit(0)

        episode_reward = tensordict_data["next", "reward"].mean().item()

        # Log mean reward per episode
        aim_run.track(episode_reward, name="reward/episode_mean", step=i)
        aim_run.track(
            tensordict_data["next", "reward"].max().item(),
            name="reward/episode_max",
            step=epoch_idx,
        )
        aim_run.track(
            tensordict_data["next", "reward"].min().item(),
            name="reward/episode_min",
            step=epoch_idx,
        )

        non_zero_rewards = tensordict_data["next", "reward"][
            tensordict_data["next", "reward"] != 0
        ]
        if len(non_zero_rewards) > 0:
            avg_non_zero_reward = non_zero_rewards.mean().item()
            aim_run.track(
                avg_non_zero_reward, name="reward/average_non_zero", step=epoch_idx
            )
            print("Average non-zero reward: ", avg_non_zero_reward)

        for j in range(num_epochs):
            print("Epoch: ", j)
            aim_run.track(
                tensordict_data["advantage"].mean().item(),
                name="advantage/mean",
                step=epoch_idx,
            )
            aim_run.track(
                tensordict_data["advantage"].std().item(),
                name="advantage/std",
                step=epoch_idx,
            )

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)
            nbatches = frames_per_batch // subbatch_size

            batch_loss_objective = 0
            batch_loss_critic = 0
            batch_loss_entropy = 0
            batch_loss_total = 0
            batch_grad_norm = 0

            for k in range(nbatches):
                print("Batch: ", k)
                subdata = replay_buffer.sample(subbatch_size)
                loss_vals = loss_module(subdata)
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optim.zero_grad()
                loss_value.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_norm=0.5, norm_type=2
                )
                optim.step()

                # Accumulate batch losses for logging
                batch_loss_objective += loss_vals["loss_objective"].item()
                batch_loss_critic += loss_vals["loss_critic"].item()
                batch_loss_entropy += loss_vals["loss_entropy"].item()
                batch_loss_total += loss_value.item()
                batch_grad_norm += grad_norm.item()

            aim_run.track(
                batch_loss_objective / nbatches, name="loss/objective", step=epoch_idx
            )
            aim_run.track(
                batch_loss_critic / nbatches, name="loss/critic", step=epoch_idx
            )
            aim_run.track(
                batch_loss_entropy / nbatches, name="loss/entropy", step=epoch_idx
            )
            aim_run.track(
                batch_loss_total / nbatches, name="loss/total", step=epoch_idx
            )
            aim_run.track(
                batch_grad_norm / nbatches, name="gradients/norm", step=epoch_idx
            )

            # aim_run.track(
            #     scheduler.get_last_lr()[0], name="learning_rate", step=epoch_idx
            # )

            epoch_idx += 1

        logs["reward"].append(episode_reward)

        track_params_dists(policy_module, aim_run)
        track_gradients_dists(policy_module, aim_run)

        track_gradients_dists(value_module, aim_run)
        track_params_dists(value_module, aim_run)
        # scheduler.step()
        aim_run.track(time.perf_counter() - t, name="time/total_seconds", step=i)
        collector.update_policy_weights_()

    collector.shutdown()
    aim_run.close()
