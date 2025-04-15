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
from torchrl.data.replay_buffers import ReplayBuffer, TensorDictReplayBuffer
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
from torchrl.collectors.utils import split_trajectories


if __name__ == "__main__":
    n_tasks = 320 / 2
    n_scenarios = 24

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=n_tasks * n_scenarios,
            device="cpu",
        ),
        sampler=SamplerWithoutReplacement(),
    )

    replay_buffer.load("geft_runs_4x4_10.rpb")
    print("Replay buffer loaded")

    # Split by trajectory
    traj = split_trajectories(
        replay_buffer.storage._storage, done_key="done", trajectory_key="a"
    )
    # traj = traj.squeeze(0)

    print("Trajectories loaded")
    print("Number of trajectories: ", len(traj))
    print("Number of samples: ", len(traj[0]))
    print("Shape of samples: ", traj[0].shape)
    print("Shape of traj: ", traj.shape)
    r = None
    best_imp = None
    fig = plt.figure(figsize=(10, 5))
    imp = np.zeros(len(traj))
    for i in range(len(traj)):
        tr = traj[i]
        # Get the rewards

        rewards = tr["next", "reward"].cpu().numpy()
        ai = tr["next", "observation", "aux", "time"][-1].cpu().numpy()
        improvement = rewards.sum()
        imp[i] = ai.item()
        print("Improvement: ", improvement)
        if best_imp is None or improvement < best_imp:
            best_imp = improvement
            r = rewards
        # print(rewards)

        # Plot the rewards
    plt.plot(r, label="Rewards")

    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.title("Rewards over time")
    plt.legend()
    plt.show()

    # Plot cumulative rewards
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(r), label="Cumulative Rewards")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Rewards over time")
    plt.legend()
    plt.show()

# # Print max and min of even improvement indicies
# even_imp = imp[::2]
# print(
#     "Min/Max of even improvement indicies: ",
#     even_imp.min(),
#     even_imp.max(),
#     even_imp.max() / even_imp.min(),
# )

# odd_imp = imp[1::2]
# print(
#     "Min/Max of odd improvement indicies: ",
#     odd_imp.min(),
#     odd_imp.max(),
#     odd_imp.max() / odd_imp.min(),
# )
