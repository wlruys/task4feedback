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
from task4feedback.ml.test_envs import *

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def build_jacobi_graph(config: JacobiConfig, randomize=True) -> JacobiGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = DynamicJacobiGraph(geom, config)

    jgraph.apply_variant(JacobiVariant)

    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)
    partition = [x + 1 for x in partition]

    jgraph.set_cell_locations(partition)

    if randomize:
        jgraph.randomize_locations(
            config.randomness, location_list=[1, 2, 3, 4], verbose=True
        )

    location_map = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
    }
    jgraph.permute_locations(location_map, config.permute_idx)

    return jgraph


def make_jacobi_env(config: JacobiConfig):
    gmsh.initialize()
    s = uniform_connected_devices(5, 1000000000000, 1, 1000)
    jgraph = build_jacobi_graph(config)

    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()
    spec = create_graph_spec(
        max_data=200, max_tasks=200, max_edges_tasks_data=200, max_edges_tasks_tasks=200
    )
    input = SimulatorInput(
        m, d, s, transition_conditions=fastsim.BatchTransitionConditions(5, 2, 16)
    )

    internal_mapper = fastsim.DequeueEFTMapper
    external_mapper = ExternalMapper

    chunks = 50
    partition = jgraph.mincut_per_levels(arch=DeviceType.GPU, level_chunks=chunks)
    aligned, perms, flips = jgraph.align_partitions()

    max_levels = max(list(jgraph.level_to_task.keys()))
    print("Max levels:", max_levels)

    level_to_mapping = {}
    for i in range(max_levels + 1):
        chunk_idx = i // chunks
        level_to_mapping[i] = aligned[chunk_idx]
    print("Level to mapping:", level_to_mapping)

    external_mapper = LevelPartitionMapper(level_cell_mapping=level_to_mapping)

    print("Aligned partitions:", aligned)

    env = MapperRuntimeEnv(
        SimulatorFactory(
            input,
            spec,
            XYObserverFactory,
            internal_mapper=internal_mapper,
            external_mapper=external_mapper,
        ),
        device="cpu",
    )
    env.enable_external_mapper()
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())

    return env


if __name__ == "__main__":
    jacobi_config = DynamicJacobiConfig(L=1, n=8, steps=50)
    env = make_jacobi_env(jacobi_config)

    d = env.rollout(3500)
