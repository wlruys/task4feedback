from typing import Optional, Self
from dataclasses import dataclass
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import os
from aim import Run, Distribution, Figure
from aim.pytorch import track_gradients_dists, track_params_dists
from matplotlib import pyplot as plt

from task4feedback.types import *
from task4feedback.graphs import *
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
from task4feedback.fastsim.models import TaskAssignmentNet, VectorTaskAssignmentNet


# Initialize Aim run for tracking
run = Run(experiment="ppo")
run.add_tag("ppo_test")


def init_weights(m):
    """Initializes LayerNorm layers."""
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


@dataclass
class Args:
    hidden_dim: int = 64
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False  # This seems unused, consider removing if not needed later
    wandb_project_name: str = "cleanRL"  # Unused, consider removing
    wandb_entity: str = None  # Unused, consider removing
    capture_video: bool = False  # Unused, consider removing
    env_id: str = "CartPole-v1"  # Unused, consider removing
    total_timesteps: int = 500000  # Unused, but good to keep for potential scaling
    learning_rate: float = 2.5e-4
    num_envs: int = 4  # Unused, but might be used for parallelization later
    num_steps: int = 128  # Unused, consider removing
    anneal_lr: bool = True  # Unused currently, but good for future experiments
    gamma: float = 1
    gae_lambda: float = 1
    num_minibatches: int = 10
    update_epochs: int = 4
    norm_adv: bool = True  # Keep for advantage normalization control
    clip_coef: float = 0.2
    clip_vloss: bool = True  # Unused, but part of PPO, consider removing
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None  # Unused, consider removing
    batch_size: int = 0  # Computed in runtime
    minibatch_size: int = 0  # Computed in runtime, consider removing
    num_iterations: int = 1000
    graphs_per_update: int = 2500
    devices: int = 4
    vcus: int = 1
    blocks: int = 2
    cuda: bool = True


args = Args()

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print(f"Using device: {device}")

run["hparams"] = vars(args)
#start_logger()

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic


def initialize_simulator():
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
    config = CholeskyConfig(blocks=args.blocks, task_config=task_config)
    tasks, data = make_graph(config, data_config=data_config)

    mem = 1600 * 1024 * 1024 * 1024
    bandwidth = (20 * 1024 * 1024 * 1024) / 10**4
    latency = 1
    devices = uniform_connected_devices(args.devices, mem, latency, bandwidth)

    # start_logger()  # Consider removing if not actively debugging

    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=TNoiseType.NONE,
        cmapper_type=CMapperType.EFT_DEQUEUE,
        pymapper=RoundRobinPythonMapper(args.devices),  # Use args.devices
        seed=100,
    )
    sim = H.create_simulator()
    sim.initialize(use_data=True)
    sim.initialize_data_manager()
    
    sim.randomize_durations()
    sim.enable_python_mapper()
    return H, sim


def logits_to_actions(logits, action=None):
    """Converts logits to actions, log probabilities, and entropy."""
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


class GreedyNetworkMapper(PythonMapper):
    """Mapper that selects actions greedily based on the model's output."""

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates)
        with torch.no_grad():
            p, d, v = self.model(data)
            p_per_task = torch.argmax(p, dim=1)
            dev_per_task = torch.argmax(d, dim=1)
            action_list = [
                Action(
                    candidates[i],
                    i,
                    dev_per_task[i].item(),
                    p_per_task[i].item(),
                    p_per_task[i].item(),
                )
                for i in range(len(candidates))
            ]
        return action_list


class RandomNetworkMapper(PythonMapper):
    """Mapper that samples actions from the model's output distribution."""

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates)
        data = data.to(device)
        with torch.no_grad():
            self.model.eval()  # Set to eval mode for inference

            p, d, v = self.model(data)
            self.model.train()  # Switch back to training mode

            p_per_task, plogprob, _ = logits_to_actions(p)
            dev_per_task, dlogprob, _ = logits_to_actions(d)

            if output is not None:
                output["candidates"] = candidates
                output["state"] = data
                output["plogprob"] = plogprob
                output["dlogprob"] = dlogprob
                output["value"] = v
                output["pactions"] = p_per_task
                output["dactions"] = dev_per_task

            action_list = [
                Action(
                    candidates[i],
                    i,
                    # (
                    #     dev_per_task[i].item()
                    #     if dev_per_task.dim() > 0
                    #     else dev_per_task.item()
                    # ),
                    3,
                    0,
                    0,
                    # p_per_task[i].item() if p_per_task.dim() > 0 else p_per_task.item(),
                    # p_per_task[i].item() if p_per_task.dim() > 0 else p_per_task.item(),
                )
                for i in range(len(candidates))
            ]
        return action_list

    def evaluate(self, obs, daction, paction):
        p, d, v = self.model(obs)
        _, plogprob, pentropy = logits_to_actions(p, paction)
        _, dlogprob, dentropy = logits_to_actions(d, daction)
        return (p, plogprob, pentropy), (d, dlogprob, dentropy), v


def collect_batch(episodes, sim, H, model, global_step=0):
    """Collects a batch of experience by running the simulator."""
    batch_info = []
    t = time.perf_counter()
    for _ in range(episodes):
        # sim.randomize_priorities()
        env = H.copy(sim)
        done = False

        # Run baseline (EFT) for comparison
        #baseline = H.copy(sim)
        #baseline.disable_python_mapper()
        #baseline.set_c_mapper(H.get_new_c_mapper())
        #baseline.run()
        baseline_time = 10 #baseline.get_current_time()

        # Run environment with policy
        obs, immediate_reward, done, terminated, info = env.step()
        episode_info = []

        while not done:
            env.enable_python_mapper()
            candidates = env.get_mapping_candidates()
            record = {}
            action_list = RandomNetworkMapper(model).map_tasks(candidates, env, record)
            obs, immediate_reward, done, terminated, info = env.step(action_list)
            record["done"] = done
            record["time"] = env.get_current_time()

            if done:
                percent_improvement = (
                    1 + (baseline_time - record["time"]) / baseline_time
                )
                record["reward"] = percent_improvement
                run.track(percent_improvement, name="percent_improvement")
                run.track(
                    record["time"],
                    name="makespan",
                )
            else:
                record["reward"] = 0

            episode_info.append(record)

        # Compute advantages and returns
        # with torch.no_grad():
        #     for t in range(len(episode_info)):
        #         episode_info[t]["returns"] = episode_info[-1]["reward"]
        #         episode_info[t]["advantage"] = (
        #             episode_info[-1]["reward"] - episode_info[t]["value"]
        #         )

        # print(
        #     f"Time: {env.get_current_time()}, Baseline: {baseline_time}, Length: {len(batch_info)}"
        # )
        batch_info.extend(episode_info)

    print(f"Batch collected in {time.perf_counter() - t} seconds")
    print(f"Number of episodes: {episodes}, batch size: {len(batch_info)}")
    return batch_info


def batch_update(batch_info, update_epoch, model, optimizer, global_step):
    """Updates the model using PPO with the collected batch data."""
    n_obs = len(batch_info)

    # Prepare data for training
    states = []
    for i in range(n_obs):
        item = batch_info[i]
        state = item["state"]
        state["plogprob"] = item["plogprob"]
        state["dlogprob"] = item["dlogprob"]
        state["value"] = item["value"]
        state["pactions"] = item["pactions"]
        state["dactions"] = item["dactions"]
        state["advantage"] = item["advantage"]
        state["returns"] = item["returns"]
        states.append(state)

    for _ in range(update_epoch):
        loader = DataLoader(
            states, batch_size=n_obs // args.num_minibatches, shuffle=True
        )

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            out = model(batch, batch["tasks"].batch)
            p, d, v = out

            pa, plogprob, pentropy = logits_to_actions(
                p, batch["pactions"].detach().view(-1)
            )
            da, dlogprob, dentropy = logits_to_actions(
                d, batch["dactions"].detach().view(-1)
            )

            plogratio = plogprob.view(-1) - batch["plogprob"].detach().view(-1)
            pratio = plogratio.exp()
            dlogratio = dlogprob.view(-1) - batch["dlogprob"].detach().view(-1)
            dratio = dlogratio.exp()

            mb_advantages = batch["advantage"].detach().view(-1)

            # Policy loss (PPO clipped surrogate objective)
            ppg_loss1 = mb_advantages * pratio.view(-1)
            ppg_loss2 = mb_advantages * torch.clamp(
                pratio.view(-1), 1 - args.clip_coef, 1 + args.clip_coef
            )
            ppg_loss = -torch.min(ppg_loss1, ppg_loss2).mean()

            dpg_loss1 = mb_advantages * dratio.view(-1)
            dpg_loss2 = mb_advantages * torch.clamp(
                dratio.view(-1), 1 - args.clip_coef, 1 + args.clip_coef
            )
            dpg_loss = -torch.min(dpg_loss1, dpg_loss2).mean()

            # Value loss (clipped)
            newvalue = v.view(-1)
            v_loss_unclipped = (newvalue - batch["returns"].detach().view(-1)) ** 2
            v_clipped = batch["value"].detach().view(-1) + torch.clamp(
                newvalue - batch["value"].detach().view(-1),
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - batch["returns"].detach().view(-1)) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            # Entropy loss
            entropy_loss = pentropy.mean() + dentropy.mean()

            # Total loss
            loss = (
                ppg_loss
                + dpg_loss
                - args.ent_coef * entropy_loss
                + v_loss * args.vf_coef
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            run.track(
                ppg_loss.item(),
                name="ppg_loss",
                context={"subset": "train"},
                epoch=global_step,
                step=epoch * len(loader) + batch_idx,
            )
            run.track(
                dpg_loss.item(),
                name="dpg_loss",
                context={"subset": "train"},
                epoch=global_step,
                step=epoch * len(loader) + batch_idx,
            )
            run.track(
                v_loss.item(),
                name="v_loss",
                context={"subset": "train"},
                epoch=global_step,
                step=epoch * len(loader) + batch_idx,
            )
            run.track(
                entropy_loss.item(),
                name="entropy_loss",
                context={"subset": "train"},
                epoch=global_step,
                step=epoch * len(loader) + batch_idx,
            )
            run.track(
                loss.item(),
                name="total_loss",
                context={"subset": "train"},
                epoch=global_step,
                step=epoch * len(loader) + batch_idx,
            )


# Main training loop
H, sim = initialize_simulator()
candidates = sim.get_mapping_candidates()
local_graph = sim.observer.local_graph_features(candidates)
model = TaskAssignmentNet(args.devices, 4, args.hidden_dim, local_graph, device).to(
    device
)
model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
netmap = GreedyNetworkMapper(model)  # For evaluation, not training
rnetmap = RandomNetworkMapper(model)  # Used during collection
H.set_python_mapper(netmap)  # Initial mapper
model.apply(init_weights)




for epoch in range(args.num_iterations):
    print(f"Epoch: {epoch}")

    start_t = time.perf_counter()
    batch_info = collect_batch(args.graphs_per_update, sim, H, model, global_step=epoch)
    end_t = time.perf_counter()

    print(f"Batch collection time: {end_t - start_t:.2f} seconds")
    
    start_t = time.perf_counter()
    batch_info = collect_batch(args.graphs_per_update, sim, H, model, global_step=epoch)
    end_t = time.perf_counter()

    print(f"Batch collection time: {end_t - start_t:.2f} seconds")

    start_t = time.perf_counter()
    batch_update(batch_info, args.update_epochs, model, optimizer, global_step=epoch)
    end_t = time.perf_counter()
    print(f"Batch update time: {end_t - start_t:.2f} seconds")

    start_t = time.perf_counter()
    # Gradient and parameter monitoring
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
    total_norm = total_norm**0.5
    run.track(total_norm, name="total_norm")
    track_params_dists(model, run)
    track_gradients_dists(model, run)

    end_t = time.perf_counter()
    print(f"Monitoring time: {end_t - start_t:.2f} seconds")
