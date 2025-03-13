from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import *
from task4feedback.interface import TaskTuple
import time
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *

from dataclasses import dataclass
from task4feedback.ml.ppo import *
from typing import Callable, List, Self
import numpy as np
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
    StaticExternalMapper,
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
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, HeteroConv
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


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


@dataclass
class JacobiConfig:
    L: int = 4
    n: int = 4 
    steps: int = 1
    n_part: int = 4
    randomness: float = 0
    permute_idx: int = 0
    
class JacobiVariant(VariantBuilder):
    @staticmethod
    def build_variant(arch: DeviceType, task: TaskTuple) -> Optional[VariantTuple]:
        memory_usage = 0
        vcu_usage = 1
        expected_time = 1000
        if arch == DeviceType.GPU:
            return VariantTuple(arch, memory_usage, vcu_usage, expected_time)
        else:
            return None 
        
        
def build_jacobi_graph(config: JacobiConfig) -> JacobiGraph:
    
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)
    
    jgraph = JacobiGraph(geom, config.steps)
    
    jgraph.apply_variant(JacobiVariant)
    
    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)
    # print(partition)
    jgraph.set_cell_locations(partition)
    
    return jgraph
    
def make_jacobi_env(config: JacobiConfig):
    gmsh.initialize()
    s = uniform_connected_devices(5, 1000000000, 1, 2000)
    jgraph = build_jacobi_graph(config)
    


    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(m, d, s)
    env = RuntimeEnv(
        SimulatorFactory(input, spec, DefaultObserverFactory), device="cpu"
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())
    
    return env

if __name__ == "__main__":
    
    jacobi_config = JacobiConfig(steps=10)

    def make_env() -> RuntimeEnv:
        return make_jacobi_env(jacobi_config)
    
    env = make_env()
    
    
    
    #start_logger()
    
    sim = env.simulator
    jgraph = env.simulator_factory.input.graph 
    mappings = jgraph.get_mapping_from_locations()
    print(mappings)
    
    sim.external_mapper = StaticExternalMapper(mapping_dict=mappings)
    sim.enable_external_mapper()
    sim.run()
    
    print(f"Final state: {sim.status}")
    print(f"Final time: {sim.time}")
    
    @dataclass
    class EnvironmentState:
        time: int 
        compute_tasks: List[fastsim.ComputeTask]
        data_tasks: List[fastsim.DataTask]
        compute_tasks_by_state: dict
        data_tasks_by_state: dict
        mapping_dict: dict
        data_task_source_device: dict
        data_task_virtual: dict
        data_task_block: dict
        
    
    def parse_state(env: RuntimeEnv, time: Optional[int] = None):
        if time is None:
            time = env.simulator.time
        graph = env.simulator_factory.input.graph
        data = env.simulator_factory.input.data
        sim = env.simulator
        assert(graph.ctasks is not None)
        compute_tasks = graph.ctasks.get_compute_tasks()
        data_tasks = graph.ctasks.get_data_tasks()
        simulator_state = sim.state 
        
        compute_tasks_by_state = defaultdict(lambda: list())
        mapping_dict = {}
        
        for task in compute_tasks:
            task_state = simulator_state.get_state_at(task.id, time)
            compute_tasks_by_state[task_state].append(task)
            device_id = simulator_state.get_mapping(task.id)
            mapping_dict[task.id] = device_id
            
        data_tasks_by_state = defaultdict(lambda: list())
        data_task_source_device = {}
        data_task_virtual = {}
        data_task_block = {}
        for task in data_tasks:
            task_state = simulator_state.get_state_at(task.id, time)
            data_tasks_by_state[task_state].append(task)
            associated_compute_task_id = task.get_compute_task()
            device_id = simulator_state.get_mapping(associated_compute_task_id)
            source_device = simulator_state.get_data_task_source(task.id)
            data_task_source_device[task.id] = source_device
            is_virtual = simulator_state.is_data_task_virtual(task.id)
            data_task_virtual[task.id] = is_virtual
            mapping_dict[task.id] = device_id
            data_task_block[task.id] = task.get_data_id()
            #print(task.id, task_state, device_id, source_device, is_virtual)
            
        return EnvironmentState(
            time=time,
            compute_tasks=compute_tasks,
            data_tasks=data_tasks,
            compute_tasks_by_state=compute_tasks_by_state,
            data_tasks_by_state=data_tasks_by_state,
            mapping_dict=mapping_dict,
            data_task_source_device=data_task_source_device,
            data_task_virtual=data_task_virtual,
            data_task_block=data_task_block,
        )
    
    state_list = []
    for t in range(0, 40000, 500):
        state = parse_state(env, time=t)
        state_list.append(state)
        
    def animate_state_list(graph, state_list):
        geom = graph.geom
        fig, ax = create_mesh_plot(geom)
        
        
    # feature_config = FeatureDimConfig.from_observer(env.observer)
    # layer_config = LayerConfig(hidden_channels=64, n_heads=2)
    # model = OldSeparateNet(
    #     feature_config=feature_config,
    #     layer_config=layer_config,
    #     n_devices=5,
    # )
    # # model = torch.compile(model, dynamic=False)

    # config = PPOConfig(train_device="cuda")
    # run_ppo_torchrl(model, make_env, config)
