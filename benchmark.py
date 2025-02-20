# Python standard library imports
import os
import random
from dataclasses import dataclass
from typing import Optional, Self
import time 

# Core numeric/scientific libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch Geometric imports
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# TorchRL and TensorDict imports
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor
from torch.distributions.categorical import Categorical
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.collectors.distributed import DistributedSyncDataCollector

from torch.profiler import profile, record_function, ProfilerActivity

# FastSim imports
from task4feedback.fastsim.interface import (
    SimulatorHandler,
    uniform_connected_devices,
    TNoiseType,
    CMapperType,
    RoundRobinPythonMapper,
    Phase,
    PythonMapper,
    Action,
    start_logger,
    ExecutionState,
)
from task4feedback.fastsim.models import TaskAssignmentNet, VectorTaskAssignmentNet, ActorCriticHead
from task4feedback.types import *
from task4feedback.graphs import *

# Visualization and tracking
from aim import Run, Distribution, Figure
from aim.pytorch import track_gradients_dists, track_params_dists
from matplotlib import pyplot as plt

# TorchRL imports for parallel environments
from torchrl.envs import EnvBase, ParallelEnv, SerialEnv
from tensordict import TensorDict, TensorDictBase
from functorch import vmap
from torchrl.envs import check_env_specs
from torchrl.data import (
    Composite,
    TensorSpec,
    Unbounded,
    Binary,
    Bounded,
)
from torchrl.envs.utils import make_composite_from_td
from tensordict.nn import set_composite_lp_aggregate

set_composite_lp_aggregate(True).set()


@dataclass 
class Args:
    devices: int = 4
    algorithm: str = "ppo"
    app: str = "cholesky"
    blocks: int = 2
    tag: str = "test"
    cuda: bool = False
    seed: int = 0
    torch_deterministic: bool = True
    vcus: float = 0.5
    
args = Args()

# Initialize Aim run for tracking
run = Run(experiment="ppo")
run.add_tag("ppo_test")



device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print(f"Using device: {device}")

run["hparams"] = vars(args)

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic


def _observation_to_heterodata(observation: TensorDict) -> HeteroData:
    hetero_data = HeteroData()
    
    for node_type, node_data in observation["nodes"].items():
        hetero_data[f"{node_type}"].x = node_data["attr"]
    
    for edge_key, edge_data in observation["edges"].items():
        target, source = edge_key.split("_")
        
        hetero_data[source, 'uses', target].edge_index = edge_data["idx"]
        hetero_data[source, 'uses', target].edge_attr = edge_data["attr"]
        
    return hetero_data

def initialize_simulator(blocks=args.blocks) -> SimulatorHandler:
    """Initializes the simulator with a Cholesky decomposition graph."""

    def task_config(task_id: TaskID) -> TaskPlacementInfo:
        placement_info = TaskPlacementInfo()
        placement_info.add(
            (Device(Architecture.GPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
        )
        placement_info.add(
            (Device(Architecture.CPU, -1),),
            TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
        )
        return placement_info

    data_config = CholeskyDataGraphConfig(data_size=1 * 1024 * 1024 * 1024)
    config = CholeskyConfig(blocks=blocks, task_config=task_config)
    tasks, data = make_graph(config, data_config=data_config)

    mem = 1600 * 1024 * 1024 * 1024
    bandwidth = (20 * 1024 * 1024 * 1024) / 10**4
    latency = 1
    devices = uniform_connected_devices(args.devices, mem, latency, bandwidth)

    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=TNoiseType.NONE,
        cmapper_type=CMapperType.EFT_DEQUEUE,
        pymapper=RoundRobinPythonMapper(args.devices),
        seed=100,
    )

    return H

forward_acc = 0
sim_acc = 0
obs_acc = 0

class FastSimEnv(EnvBase):

    def _make_batch_variable(self, spec, key=None):
        if isinstance(spec, Composite):
            new_spec = Composite(
                **{
                    key: self._make_batch_variable(subspec, key=key)
                    for key, subspec in spec.items()
                },
                shape=spec.shape,
                device=spec.device,
            )  # keep composite shape and device
            return new_spec
        elif isinstance(spec, TensorSpec):
            if "index" in key:
                return Unbounded(
                    shape=(*spec.shape[:1], -1),
                    device=spec.device,
                    dtype=spec.dtype,
                )
            return Unbounded(
                shape=(-1, *spec.shape[1:]),
                device=spec.device,
                dtype=spec.dtype,
            )
        else:
            return spec

    def _create_observation_space(self):
        candidates = self.base_sim.get_mapping_candidates()
        local_graph = self.base_sim.observer.get_local_graph_tensordict(candidates)

        comp = make_composite_from_td(local_graph)
        comp = self._make_batch_variable(comp)
        comp = Composite(observation=comp)
        return comp

    def _create_action_spec(self, n_devices: int = 4) -> TensorSpec:
        out = Bounded(
            shape=(1),
            device=self.device,
            dtype=torch.int32,
            low=torch.tensor(0, device=self.device),
            high=torch.tensor(n_devices - 1, device=self.device),
        )
        out = Composite(action=out)
        return out

    def _create_reward_spec(self) -> TensorSpec:
        return Unbounded(
            shape=(1),
            device=self.device,
            dtype=torch.float32,
        )

    def _reset_simulator(self):
        env = self.simulator_handle.copy(self.base_sim)
        env.set_c_mapper(self.simulator_handle.get_new_c_mapper())
        self.sim = env
        self.sim.step()

    def _get_observation(self):
        candidates = self.sim.get_mapping_candidates()
        local_graph = self.sim.observer.get_local_graph_tensordict(candidates)
        td = TensorDict(observation=local_graph)
        return td

    def __init__(self, simulator_handle, device="cpu"):
        super().__init__(device=device)
        self.batch_size = torch.Size([])

        self.simulator_handle = simulator_handle

        sim = simulator_handle.create_simulator()
        enable_logging = False
        sim.initialize(use_data=True)
        sim.randomize_durations()
        # sim.randomize_priorities()

        sim.enable_python_mapper()
        self.base_sim = sim
        self.observation_spec = self._create_observation_space()
        self.reward_spec = self._create_reward_spec()
        self.action_spec = self._create_action_spec()
        self.done_spec = Binary(shape=(1,), device=self.device, dtype=torch.bool)

    def _reset(self, td: Optional[TensorDict] = None):
        self._reset_simulator()
        return self._get_observation()

    def _step(self, td: TensorDict) -> TensorDict:

        t = time.time()

        candidate = td["observation"]["aux"]["candidates"][0].item()
        candidate = self.sim.get_mapping_candidates()[0]
        chosen_device = td["action"].item()
        
        obs1_t = time.time() - t
        t = time.time()

        #print("I choose: ", chosen_device, candidate)

        action_list = [Action(candidate, 0, 3, 0, 0)]

        obs, immediate_reward, done, terminated, info = self.sim.step(action_list)
        etime = self.sim.get_current_time()

        reward = torch.tensor(
            [immediate_reward], device=self.device, dtype=torch.float32
        )
        obs2_t = time.time() - t
        t = time.time()

        if not done:
            obs = self._get_observation()
        else:
            obs = self._reset()

        done = torch.tensor(done, device=self.device)

        tensordict = obs
        tensordict.set("reward", reward)
        tensordict.set("done", done)
        tensordict.set("terminated", terminated)
        tensordict.set("time", torch.tensor([etime], device=self.device))
        obs3_t = time.time() - t
        
        global forward_acc
        global sim_acc
        global obs_acc
        
        sim_acc += obs2_t
        obs_acc = obs1_t + obs3_t + obs_acc

        return tensordict

    def _set_seed(self, seed: int) -> Self:
        pass


@dataclass
class HeteroGATConfig:
    task_feature_dim: int = 12
    data_feature_dim: int = 5
    device_feature_dim: int = 12
    task_data_edge_dim: int = 3
    task_device_edge_dim: int = 2
    task_task_edge_dim: int = 1
    hidden_channels: int = 16
    n_heads: int = 2


class HeteroGAT(nn.Module):
    def __init__(self, config: HeteroGATConfig):
        super(HeteroGAT, self).__init__()

        self.in_channels_tasks = config.task_feature_dim
        self.in_channels_data = config.data_feature_dim
        self.in_channels_devices = config.device_feature_dim

        self.task_data_edge_dim = config.task_data_edge_dim
        self.task_device_edge_dim = config.task_device_edge_dim
        self.task_task_edge_dim = config.task_task_edge_dim

        self.hidden_channels = config.hidden_channels

        self.n_heads = config.n_heads

        self.gnn_tasks_data = GATConv(
            (self.in_channels_data, self.in_channels_tasks),
            self.hidden_channels,
            heads=self.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=self.task_data_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_devices = GATConv(
            (self.in_channels_devices, self.in_channels_tasks),
            self.hidden_channels,
            heads=self.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=self.task_device_edge_dim,
            add_self_loops=False,
        )

        self.gnn_tasks_tasks = GATConv(
            self.in_channels_tasks,
            self.hidden_channels,
            heads=self.n_heads,
            concat=False,
            residual=True,
            dropout=0,
            edge_dim=self.task_task_edge_dim,
            add_self_loops=True,
        )

        self.linear = nn.Linear(
            (self.hidden_channels * 3),
            self.in_channels_tasks,
        )

        # Layer normalization layers
        self.layer_norm_data_tasks = nn.LayerNorm(self.hidden_channels)
        self.layer_norm_device_tasks = nn.LayerNorm(self.hidden_channels)
        self.layer_norm_tasks_tasks = nn.LayerNorm(self.hidden_channels)

        # Activation function
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, obs: HeteroData) -> torch.Tensor:
        # Process data to tasks        
        data_fused_tasks = self.gnn_tasks_data(
            (obs["data"].x, obs["tasks"].x),
            obs["data", "uses", "tasks"].edge_index,
            obs["data", "uses", "tasks"].edge_attr,
        )
        data_fused_tasks = self.layer_norm_data_tasks(data_fused_tasks)
        data_fused_tasks = self.activation(data_fused_tasks)

        # Process devices to tasks
        device_fused_tasks = self.gnn_tasks_devices(
            (obs["devices"].x, obs["tasks"].x),
            obs["devices", "uses", "tasks"].edge_index,
            obs["devices", "uses", "tasks"].edge_attr,
        )
        device_fused_tasks = self.layer_norm_device_tasks(device_fused_tasks)
        device_fused_tasks = self.activation(device_fused_tasks)

        task_fused_tasks = self.gnn_tasks_tasks(
            obs["tasks"].x,
            obs["tasks", "uses", "tasks"].edge_index,
            obs["tasks", "uses", "tasks"].edge_attr,
        )
        task_fused_tasks = self.layer_norm_tasks_tasks(task_fused_tasks)
        task_fused_tasks = self.activation(task_fused_tasks)

        # Concatenate the processed feature
        x = torch.cat(
            [obs["tasks"].x, task_fused_tasks, data_fused_tasks, device_fused_tasks],
            dim=1,
        )

        return x


class TaskAssignmentNetDeviceOnly(nn.Module):
    def __init__(self, ndevices, config: HeteroGATConfig):
        super(TaskAssignmentNetDeviceOnly, self).__init__()

        self.in_channels_tasks = config.task_feature_dim
        self.in_channels_data = config.data_feature_dim
        self.in_channels_devices = config.device_feature_dim

        self.task_data_edge_dim = config.task_data_edge_dim
        self.task_device_edge_dim = config.task_device_edge_dim
        self.task_task_edge_dim = config.task_task_edge_dim

        self.hetero_gat_actor = HeteroGAT(config)
        self.hetero_gat_critic = HeteroGAT(config)
        self.ndevices = ndevices

        self.hidden_channels = config.hidden_channels
        
        #import time 
        #self.time = time 

        # input dimension
        critic_input_dim = self.hidden_channels * 3 + self.in_channels_tasks
        actor_input_dim = self.hidden_channels * 3 + self.in_channels_tasks

        # Critic Head
        self.critic_head = ActorCriticHead(critic_input_dim, self.hidden_channels, 1)

        # Actor Head for Device Assignment
        self.actor_d_head = ActorCriticHead(
            actor_input_dim, self.hidden_channels, ndevices
        )
        
    def _is_batch(self, data: TensorDict) -> bool:
        if not data.batch_size:
            return False
        return True
        
    def _convert_to_heterodata(self, data: TensorDict) -> HeteroData:
        if not self._is_batch(data):
            #print("NO BATCH SIZE")
            _obs = _observation_to_heterodata(data)
            return _obs
        
        #print("BATCH SIZE", data.batch_size[0])
        _h_data = []
        for i in range(data.batch_size[0]):
            _obs = _observation_to_heterodata(data[i])
            _h_data.append(_obs)
        
        return Batch.from_data_list(_h_data)

    def forward(self, data, task_batch=None):
        #t = self.time.time()
        #print("STARTING FORWARD PASS", data.shape, flush=True)
        #print(data.shape)
        
        #print(data)        
        
        hdata = self._convert_to_heterodata(data)
        
        #print(hdata)
        
        # Get features from HeteroGAT
        #x = self.hetero_gat_critic(hdata)

        # Critic Head
        #v = self.critic_head(x)
        #v = global_mean_pool(v, task_batch)

        z = self.hetero_gat_actor(hdata)

        if self._is_batch(data):
            z = z[hdata["tasks"].ptr[:-1]]
        else:
            z = z[0]

        # Actor Head for Device Assignment
        d_logits = self.actor_d_head(z)

        #print("ENDING FORWARD PASS", flush=True)
        
        #print(d_logits)
        #t = self.time.time() - t
        global forward_acc
        forward_acc += t

        return d_logits


config = HeteroGATConfig()

_internal_policy_module = TensorDictModule(
    TaskAssignmentNetDeviceOnly(args.devices, config),
    in_keys=["observation"],
    out_keys=["logits"],
)

action_spec = Bounded(
    shape=(1,),
    device=device,
    dtype=torch.int64,
    low=torch.tensor(0, device=device),
    high=torch.tensor(args.devices - 1, device=device),
)

policy_module = ProbabilisticActor(
    _internal_policy_module,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=Categorical,
    # distribution_kwargs={"logits": "logits"},
    cache_dist=True,
    return_log_prob=True,
)

torch.set_num_threads(8)

if __name__ == "__main__":
    # freeze_support()
    #f = FastSimEnv(initialize_simulator())

    # print("Running policy:", policy_module(f.reset()))    
    #policy_module = torch.compile(policy_module)

    create_env = lambda: FastSimEnv(initialize_simulator(), device=device)

    workers = 4
    env = create_env()
    #senv = SerialEnv(workers, create_env)
    #penv = ParallelEnv(workers, create_env)
    
    
    # r = env.rollout(1, policy_module)
    
    # print(r)
    model = TaskAssignmentNetDeviceOnly(args.devices, config)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable_params)




    
    t = time.time()
    collector = SyncDataCollector(create_env, policy_module, frames_per_batch=1000, total_frames=-1)
    #collector = MultiSyncDataCollector([create_env for _ in range(workers)], policy_module, frames_per_batch=1000, total_frames=-1)
    #collector = DistributedSyncDataCollector([create_env for _ in range(workers)], policy_module, frames_per_batch=1000, total_frames=11000)
    for data in collector:
        print(data.shape)
        #print(data["collector", "traj_ids"])
        #print(data["logits"])
        break
    print("Time taken: ", time.time() - t)
    
    t = time.time()
    i = 0
    
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    for data in collector:
        print(data.shape)
        #print(data["collector", "traj_ids"])
        #print(data["logits"])
        i += 1
        if i == 10:
            break
    print("Time taken: ", time.time() - t)
    
    print("FORWARD ACC: ", forward_acc)
    print("SIM ACC: ", sim_acc)
    print("OBS ACC: ", obs_acc)
    
    
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

    

    # print("LENGTH OF ", len(d))
