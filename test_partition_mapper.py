from task4feedback.graphs.mesh.base import *
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.mesh.plot import *
from task4feedback.graphs.base import *
from task4feedback.graphs.jacobi import *
from task4feedback.graphs.dynamic_jacobi import *
from task4feedback.interface import TaskTuple
import time
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *

from dataclasses import dataclass
from task4feedback.ml.algorithms.ppo import *
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

start_logger()
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def build_jacobi_graph(config: JacobiConfig) -> JacobiGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = DynamicJacobiGraph(geom, config, JacobiVariant)

    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)
    # offset by 1 to ignore cpu
    partition = [x + 1 for x in partition]

    jgraph.set_cell_locations(partition)

    return jgraph


def make_jacobi_env(config: JacobiConfig):
    gmsh.initialize()
    s = uniform_connected_devices(5, 100000000000, 0, 1)
    jgraph = build_jacobi_graph(config)

    d = jgraph.get_blocks()
    m = jgraph
    spec = create_graph_spec()
    input = SimulatorInput(
        m, d, s, transition_conditions=fastsim.BatchTransitionConditions(5, 5, 16)
    )
    env = IncrementalEFT(
        SimulatorFactory(input, spec, CandidateObserverFactory), device="cpu"
    )

    # print(f"Runtime environment created: {env}")

    return env


if __name__ == "__main__":
    jacobi_config = DynamicJacobiConfig(
        n=4,
        L=1,
        steps=5,
        randomness=1,
        interior_size=1000,
        boundary_interior_ratio=1,
    )

    def make_env() -> RuntimeEnv:
        return make_jacobi_env(jacobi_config)

    for i in range(1):
        env = make_env()
        sim = env.simulator
        import time

        # sim.enable_external_mapper()
        # sim.set_steps(4)
        sim.disable_external_mapper()
        sim.run() 
        print(f"Final state: {sim.status}")
        print(f"Final time: {sim.time}")
        
        animate_mesh_graph(env, show=True)
        
        # for k in range(10):
        #     start_t = time.perf_counter()
        #     s = sim.copy()
        #     end_t = time.perf_counter()
        #     print(f"Copy completed in (ms) {(end_t - start_t) * 1000:.2f} ms")
        
        # print(f"Running simulation step {k + 1}")
        # start_t = time.perf_counter()
        # sim.run()
        # end_t = time.perf_counter()
        # print(
        #     f"Simulation step {k + 1} completed in (ms) {(end_t - start_t) * 1000:.2f} ms"
        # )

        #env.rollout(10000)

        # s = sim.copy()
        # s.start_drain()
        # print(f"Running simulation step {k + 1} on copy")
        # start_t = time.perf_counter()
        # s.run()
        # end_t = time.perf_counter()

        # s.stop_drain()
        # s.run()

        # print(f"Simulation completed in (ms) {(end_t - start_t) * 1000:.2f} ms")
        # print(f"Final state: {sim.status}, {s.status}")
        # print(f"Final time: {sim.time}, {s.time}")

    # relabel_dict = {}
    # for i in range(1, 5):
    #     relabel_dict[i] = i

    # # Form all permutations of relabel_dict

    # import itertools

    # p = list(itertools.permutations(relabel_dict.values()))
    # print(p)

    # jgraph.randomize_locations(1, location_list=[1, 2, 3, 4])
    # print(mappings)

    # with open("cell_locations.pkl", "rb") as f:
    #     cell_locations = pickle.load(f)

    # print("Cell locations saved to cell_locations.pkl")

    # sim.external_mapper = PartitionMapper(cell_to_mapping=cell_locations, level_start=0)
    # sim.enable_external_mapper()
    # # sim.disable_external_mapper()
    # sim.run()

    # print(f"Final state: {sim.status}")
    # print(f"Final time: {sim.time}")

    # animate_mesh_graph(env, time_interval=250, show=False, title="test_part_mapper")
