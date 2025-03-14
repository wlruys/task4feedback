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
from task4feedback.ml.test_envs import *
from task4feedback.ml.iql import *
from task4feedback.ml.models import *


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
            return VariantTuple(arch, memory_usage, vcu_usage, expected_time)


if __name__ == "__main__":
    randomness = 0

    rtc = False

    config_list = [
        JacobiConfig(L=4, n=4, steps=14, n_part=4, randomness=randomness, permute_idx=i)
        for i in range(2)
    ]

    def make_graph(config):
        gmsh.initialize()
        graph = build_jacobi_graph(config)
        graph.apply_variant(JacobiVariant)
        return graph

    def make_sys():
        s = uniform_connected_devices(5, 1000000000, 1, 2000)
        return s

    eval_set = (make_graph, make_sys, config_list)

    # Load the replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=224 * 50 * 24),
        sampler=SamplerWithoutReplacement(),
    )
    # Load the replay buffer from file
    if rtc:
        print("Loading RTC replay buffer")
        replay_buffer.load(
            "eft_runs_4x4_16_0.9375.rpb",
        )
    else:
        print("Loading Non-RTC replay buffer")
        replay_buffer.load(
            "eft_runs_4x4_16_0.9375_ntc.rpb",
        )

    graph_func = make_graph
    sys_func = make_sys

    def env_wrapper(config) -> RuntimeEnv:
        graph = graph_func(config)
        s = sys_func()
        d = graph.get_blocks()
        m = graph
        m.finalize_tasks()
        spec = create_graph_spec()

        if rtc:
            input = SimulatorInput(
                m,
                d,
                s,
                transition_conditions=fastsim.RangeTransitionConditions(5, 5, 16),
            )
        else:
            input = SimulatorInput(
                m, d, s, transition_conditions=fastsim.DefaultTransitionConditions()
            )
        env = RuntimeEnv(
            SimulatorFactory(
                input,
                spec,
                DefaultObserverFactory,
            ),
            device="cpu",
        )
        env = TransformedEnv(env, StepCounter())
        env = TransformedEnv(env, TrajCounter())

        return env

    env = env_wrapper(config_list[0])
    feature_config = FeatureDimConfig.from_observer(env.observer)
    layer_config = LayerConfig(hidden_channels=64, n_heads=2)
    actor_model = OldTaskAssignmentNet(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=5,
    )
    value_model = OldValueNet(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=5,
    )
    action_value_model = OldActionValueNet(
        feature_config=feature_config,
        layer_config=layer_config,
        n_devices=5,
    )

    run_iql(
        replay_buffer, actor_model, value_model, action_value_model, eval_set, rtc=rtc
    )
