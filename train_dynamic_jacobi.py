from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *

from dataclasses import dataclass
from task4feedback.ml.ppo import *
from typing import Callable, List, Self
import numpy as np
from task4feedback import fastsim2 as fastsim
from task4feedback.interface import *
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
from task4feedback.graphs.sweep import *
from task4feedback.graphs import *
from task4feedback.graphs.mesh import *
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
import wandb
from task4feedback.graphs.dynamic_jacobi import *

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def build_sweep_graph(config: SweepConfig) -> SweepGraph:
    mesh = generate_quad_mesh(L=config.L, n=config.n)
    geom = build_geometry(mesh)

    jgraph = DynamicJacobiGraph(geom, config)

    # jgraph.apply_variant(SweepVariant)

    partition = metis_partition(geom.cells, geom.cell_neighbors, nparts=4)
    # partition = block_cyclic(
    #    geom, n_col_parts=2, n_row_parts=2, parts_per_column=1, parts_per_row=1
    # )

    # offset by 1 to ignore cpu
    partition = [x + 1 for x in partition]
    jgraph.set_cell_locations(partition)

    # jgraph.randomize_locations(config.randomness, location_list=[1, 2, 3, 4])

    return jgraph


def make_sweep_env(config: SweepConfig):
    gmsh.initialize()
    s = uniform_connected_devices(5, 100000000000000, 1, 2000)
    jgraph = build_sweep_graph(config)

    d = jgraph.get_blocks()
    m = jgraph
    m.finalize_tasks()
    spec = create_graph_spec()
    input = SimulatorInput(
        m, d, s, transition_conditions=fastsim.BatchTransitionConditions(5, 2, 16)
    )
    env = EFTIncrementalEnv(
        SimulatorFactory(input, spec, XYDataObserverFactory),
        device="cpu",
        change_priority=True,
        seed=10000,
        change_locations=True,
    )
    env = TransformedEnv(env, StepCounter())
    env = TransformedEnv(env, TrajCounter())

    return env


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


if __name__ == "__main__":
    config = DynamicJacobiConfig(
        interior_size=10000000,
        boundary_interior_ratio=0.2,
        randomness=1,
        L=1,
        n=4,
        steps=50,
    )

    def make_env():
        return make_sweep_env(config)

    env = make_env()

    wandb.init(
        project="dynamic_jacobi",
        name="batch10_4_50_xy_data_net16_ent01_heads2_random_continue_incremental",
    )

    # # load model
    # model = torch.load("model_150.pth")
    # print("loaded model")
    # print(model)
    # inner_model = model[0].module.module.network
    # model = inner_model

    feature_config = FeatureDimConfig.from_observer(env.observer)
    layer_config = LayerConfig(hidden_channels=16, n_heads=2)
    model = OldSeparateNet(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=5,
    )
    model = torch.compile(model, dynamic=False)

    mconfig = PPOConfig(
        train_device="cpu", states_per_collection=800 * 10, workers=5, ent_coef=0.01
    )
    run_ppo_torchrl(model, make_env, mconfig, model_name="model")
