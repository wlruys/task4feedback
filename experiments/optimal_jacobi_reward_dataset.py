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

from functools import partial

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
from task4feedback.ml.test_envs import *
from task4feedback.ml.iql import *
import random


def build_jacobi_graph(config: JacobiConfig, metis_init=False, nparts=4) -> JacobiGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    random_state = random.getstate()
    random.seed(10)

    jgraph = JacobiGraph(geom, config)
    jgraph.apply_variant(JacobiVariant)

    if metis_init:
        partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=nparts)
        # offset by 1 to ignore cpu
        partition = [x + 1 for x in partition]
        jgraph.set_cell_locations(partition)

    # jgraph.randomize_locations(config.randomness, location_list=[1, 2, 3, 4])

    location_map = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
    }
    jgraph.permute_locations(location_map, config.permute_idx)

    random.setstate(random_state)

    return jgraph


def make_env(
    graph_builder,
    graph_config,
    system_config,
    observer_factory_t=XYDataObserverFactory,
    change_priority=False,
    change_locations=False,
    seed=1000,
):
    gmsh.initialize()

    runtime_env_t = GeneralizedIncrementalEFT

    n_devices = system_config["n_devices"]
    bandwidth = system_config["bandwidth"]
    latency = system_config["latency"]
    s = uniform_connected_devices(n_devices, 100000000000000, latency, bandwidth)
    jgraph = graph_builder(graph_config)

    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()

    spec = create_graph_spec()
    input = SimulatorInput(
        m, d, s, transition_conditions=fastsim.BatchTransitionConditions(5, 2, 16)
    )

    mapper_env_t = wrap_runtime_env_for_external_mapper(runtime_env_t)

    cell_locations = jgraph.get_cell_locations()

    random_state = random.getstate()
    random.seed(10)
    jgraph.randomize_locations(1, location_list=[1, 2, 3, 4])
    random.setstate(random_state)

    external_mapper = PartitionMapper(
        cell_to_mapping=cell_locations,
        level_start=0,
    )
    internal_mapper = fastsim.DequeueEFTMapper

    env = mapper_env_t(
        simulator_factory=SimulatorFactory(
            input,
            spec,
            observer_factory_t,
            external_mapper=external_mapper,
            internal_mapper=internal_mapper,
        ),
        device="cpu",
        change_priority=False,
        seed=seed,
        change_locations=True,
        animate_each_episode=False,
        permute_locations=False,
        title=f"Jacobi_permute={graph_config.permute_idx}",
    )
    env.enable_external_mapper()
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())

    return env


def collect_runs(
    configured_env_list, graphs_per_batch: int = 1, batches: int = 1, workers: int = 1
):
    if len(configured_env_list) == 0:
        raise ValueError("configured_env_list is empty")

    env = configured_env_list[0]()

    n_tasks = len(env.simulator.input.graph)

    # Create a replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=n_tasks * graphs_per_batch * batches * len(configured_env_list)
        ),
        sampler=SamplerWithoutReplacement(),
    )

    for k, configured_env_func in enumerate(configured_env_list):
        print(f"Collecting runs for env {k + 1}/{len(configured_env_list)}")
        env_func = configured_env_func
        # Collect samples
        print(n_tasks)
        print(n_tasks * graphs_per_batch * workers)

        collector = SyncDataCollector(
            env_func,
            frames_per_batch=n_tasks * graphs_per_batch * workers,
            env_device="cpu",
            policy_device="cpu",
        )
        collector.set_seed(0)
        for i, batch in enumerate(collector):
            replay_buffer.extend(batch)
            if i == 0:
                break

        collector.shutdown()

    print("Collecting runs done")
    return replay_buffer


if __name__ == "__main__":
    # Define the graph builder and system configuration
    graph_builder = partial(build_jacobi_graph, metis_init=True)
    system_config = {
        "n_devices": 5,
        "bandwidth": 1,
        "latency": 0,
    }

    interior_size = 5000
    boundary_interior_ratio = 0.2
    steps = 10

    # Define the graph configurations
    graph_configs = [
        JacobiConfig(
            L=1,
            n=4,
            steps=steps,
            n_part=4,
            randomness=0,
            permute_idx=i,
            boundary_interior_ratio=boundary_interior_ratio,
            interior_size=interior_size,
        )
        for i in range(24)
    ]

    # Create the configured environments
    configured_envs = [
        partial(make_env, graph_builder, config, system_config)
        for config in graph_configs
    ]

    # Collect runs
    replay_buffer = collect_runs(configured_envs)

    # Save the replay buffer to a file
    filename = f"geft_runs_4x4_{steps}.rpb"
    replay_buffer.save(filename)
