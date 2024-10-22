from typing import Optional, Self
from task4feedback.types import *
from task4feedback.graphs import *
import argparse
import random
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
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import os


def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


@dataclass
class Args:
    hidden_dim = 64
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    replay_batches: int = 2
    """the number of batches to replay"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 1000
    """the number of iterations (computed in runtime)"""

    graphs_per_update: int = 50
    devices = 4
    vcus = 1
    blocks = 4


args = Args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic


def initialize_simulator():
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
    n_devices = args.devices
    devices = uniform_connected_devices(n_devices, mem, latency, bandwidth)
    # start_logger()

    H = SimulatorHandler(
        tasks,
        data,
        devices,
        noise_type=TNoiseType.NONE,
        cmapper_type=CMapperType.EFT_DEQUEUE,
        pymapper=RoundRobinPythonMapper(n_devices),
        seed=100,
    )
    sim = H.create_simulator()
    sim.initialize(use_data=True)
    sim.randomize_durations()
    sim.enable_python_mapper()

    # samples = 10
    # for i in range(samples):
    #     current_sim = H.copy(sim)
    #     candidates = current_sim.get_mapping_candidates()
    #     current_sim.run()
    #     print(current_sim.get_current_time())

    return H, sim


def logits_to_actions(logits, action=None):
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    return action, probs.log_prob(action), probs.entropy()


class GreedyNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator):
        data = simulator.observer.local_graph_features(candidates)
        with torch.no_grad():
            p, d, v = self.model.forward(data)

            # choose argmax of network output
            # This is e-greedy policy
            p_per_task = torch.argmax(p, dim=1)
            dev_per_task = torch.argmax(d, dim=1)
            action_list = []
            for i in range(len(candidates)):
                a = Action(
                    candidates[i],
                    i,
                    dev_per_task[i].item(),
                    p_per_task[i].item(),
                    p_per_task[i].item(),
                )
                action_list.append(a)
        return action_list


class RandomNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates)

        with torch.no_grad():
            p, d, v = self.model.forward(data)

            # sample from network output
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

            action_list = []
            for i in range(len(candidates)):

                if p_per_task.dim() == 0:
                    a = Action(
                        candidates[i],
                        i,
                        dev_per_task,
                        p_per_task,
                        p_per_task,
                    )
                else:
                    a = Action(
                        candidates[i],
                        i,
                        dev_per_task[i].item(),
                        p_per_task[i].item(),
                        p_per_task[i].item(),
                    )
                action_list.append(a)
        return action_list

    def evaluate(self, obs, daction, paction):
        p, d, v = self.model.forward(obs)
        _, plogprob, pentropy = logits_to_actions(p, paction)
        _, dlogprob, dentropy = logits_to_actions(d, daction)
        return (p, plogprob, pentropy), (d, dlogprob, dentropy), v


lr = args.learning_rate
epochs = args.num_iterations
graphs_per_epoch = args.graphs_per_update

H, sim = initialize_simulator()
candidates = sim.get_mapping_candidates()
local_graph = sim.observer.local_graph_features(candidates)
h = TaskAssignmentNet(args.devices, 4, args.hidden_dim, local_graph)
optimizer = optim.Adam(h.parameters(), lr=lr)
netmap = GreedyNetworkMapper(h)
rnetmap = RandomNetworkMapper(h)
H.set_python_mapper(netmap)
backup = H.copy(sim)
h.apply(init_weights)


# for i in range(10):
#     b = H.copy(backup)
#     b.run()
#     print(b.get_current_time())
run_name = f"a2c_4by4_ppo_compare"
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)


def collect_batch(episodes, sim, h, global_step=0):
    batch_info = []
    for e in range(0, episodes):
        # sim.randomize_priorities()
        env = H.copy(sim)
        done = False

        # Run baseline
        baseline = H.copy(sim)
        baseline.disable_python_mapper()
        a = H.get_new_c_mapper()
        baseline.set_c_mapper(a)
        baseline_done = baseline.run()
        baseline_time = baseline.get_current_time()

        # Run env to first mapping
        obs, immediate_reward, done, terminated, info = env.step()

        episode_info = []

        while not done:
            candidates = env.get_mapping_candidates()
            record = {}
            action_list = RandomNetworkMapper(h).map_tasks(candidates, env, record)

            obs, immediate_reward, done, terminated, info = env.step(action_list)
            record["done"] = done
            record["time"] = env.get_current_time()
            episode_info.append(record)

            # print(
            #     "e",
            #     e,
            #     "record",
            #     record["pactions"],
            #     record["plogprob"],
            #     record["value"],
            # )

            if done:
                percent_improvement = (
                    1 + (baseline_time - record["time"]) / baseline_time
                )
                percent_improvement = percent_improvement
                record["reward"] = percent_improvement

                # print("Terminal Reward: ", record["reward"])
                writer.add_scalar(
                    "charts/episode_reward",
                    record["reward"],
                    global_step * episodes + e,
                )

                writer.add_scalar(
                    "charts/time",
                    record["time"],
                    global_step * episodes + e,
                )

                break
            else:
                record["reward"] = 0

        # for r in episode_info:
        #     r["time"] = episode_info[-1]["time"]
        # r["reward"] = baseline_time - r["time"]

        # # Compute TD(0)
        # with torch.no_grad():
        #     lastgaelam = 0
        #     gamma = 1  # args.gamma
        #     gae_lambda = 1  # args.gae_lambda
        #     for t in reversed(range(len(episode_info))):
        #         episode_info[t]["returns"] = episode_info[-1]["reward"]

        #         if t == len(episode_info) - 1:
        #             res = episode_info[t]["value"]
        #         else:
        #             res = episode_info[t + 1]["value"] - episode_info[t]["value"]

        #         episode_info[t]["advantages"] = episode_info[t]["returns"] + res

        with torch.no_grad():
            for t in range(len(episode_info)):
                episode_info[t]["returns"] = episode_info[-1]["reward"]
                episode_info[t]["advantage"] = (
                    episode_info[-1]["reward"] - episode_info[t]["value"]
                )
        # print("Advantages: ", [e["advantages"] for e in episode_info])
        # print("Returns: ", [e["returns"] for e in episode_info])
        # print("Returns: ", [e["returns"] for e in episode_info])

        # print("Epoch: ", e)
        print(
            "Time: ",
            env.get_current_time(),
            "Baseline: ",
            baseline_time,
            "Length: ",
            len(batch_info),
        )
        # print("Epoch info: ", [e["reward"] for e in episode_info])

        batch_info.extend(episode_info)
    return batch_info


LI = 0


def batch_update(batch_info, update_epoch, h, optimizer, global_step):
    n_obs = len(batch_info)

    batch_size = args.batch_size

    pclipfracs = []
    dclipfracs = []

    state = []
    for i in range(n_obs):
        state.append(batch_info[i]["state"])
        state[i]["plogprob"] = batch_info[i]["plogprob"]
        state[i]["dlogprob"] = batch_info[i]["dlogprob"]
        state[i]["value"] = batch_info[i]["value"]
        state[i]["pactions"] = batch_info[i]["pactions"]
        state[i]["dactions"] = batch_info[i]["dactions"]
        state[i]["advantage"] = batch_info[i]["advantage"]
        state[i]["returns"] = batch_info[i]["returns"]

    global LI

    for k in range(update_epoch):
        # print("Update epoch: ", k)
        nbatches = args.num_minibatches
        batch_size = n_obs // nbatches
        loader = DataLoader(state, batch_size=batch_size, shuffle=True)

        for i, batch in enumerate(loader):
            # print("Mini-batch: ", i, "of", len(loader), "; size: ", len(batch))
            out = h(batch, batch["tasks"].batch)
            p, d, v = out

            pa, plogprob, pentropy = logits_to_actions(p, batch["pactions"])
            da, dlogprob, dentropy = logits_to_actions(d, batch["dactions"])

            mb_advantages = batch["advantage"]
            mb_advantages = mb_advantages.detach().view(-1)

            ppg_loss = -plogprob * mb_advantages
            ppg_loss = ppg_loss.mean()

            dpg_loss = -dlogprob * mb_advantages
            dpg_loss = dpg_loss.mean()

            # Value loss
            v_loss = nn.functional.mse_loss(v.view(-1), batch["returns"])

            entropy_loss = pentropy.mean() + dentropy.mean()
            loss = (
                (ppg_loss + dpg_loss)
                - args.ent_coef * entropy_loss
                + v_loss * args.vf_coef
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(h.parameters(), args.max_grad_norm)
            optimizer.step()

            writer.add_scalar(
                "charts/learning_rate",
                optimizer.param_groups[0]["lr"],
                global_step * update_epoch + k,
            )
            writer.add_scalar(
                "losses/value_loss", v_loss.item(), global_step * update_epoch + k
            )
            writer.add_scalar(
                "losses/ppolicy_loss", ppg_loss.item(), global_step * update_epoch + k
            )
            writer.add_scalar(
                "losses/entropy", entropy_loss.item(), global_step * update_epoch + k
            )
            writer.add_scalar(
                "losses/pentropy",
                pentropy.mean().item(),
                global_step * update_epoch + k,
            )
            writer.add_scalar(
                "losses/dentropy",
                dentropy.mean().item(),
                global_step * update_epoch + k,
            )
            writer.add_scalar("losses/dpolicy_loss", dpg_loss.item(), LI)
            LI = LI + 1


replay_buffer = []


def replay(batch_info, h, optimizer, global_step):
    M = 20000
    B = 1000

    global replay_buffer
    random.shuffle(replay_buffer)
    replay_buffer.extend(batch_info)

    if len(replay_buffer) > M:
        replay_buffer = replay_buffer[-M:]

    # Sample from replay buffer
    print("Replay buffer size: ", len(replay_buffer), B)

    for k in range(args.replay_batches):
        batch = random.sample(replay_buffer, min(B, len(replay_buffer) - 1))

        state = []
        for i in range(len(batch)):
            state.append(batch[i]["state"])
            state[i]["plogprob"] = batch[i]["plogprob"]
            state[i]["dlogprob"] = batch[i]["dlogprob"]
            state[i]["value"] = batch[i]["value"]
            state[i]["pactions"] = batch[i]["pactions"]
            state[i]["dactions"] = batch[i]["dactions"]
            state[i]["advantage"] = batch[i]["advantage"]
            state[i]["returns"] = batch[i]["returns"]

        loader = DataLoader(state, batch_size=len(state))

        for i, batchelem in enumerate(loader):
            out = h(batchelem, batchelem["tasks"].batch)
            p, d, v = out

            pa, plogprob, pentropy = logits_to_actions(
                p, batchelem["pactions"].detach().view(-1)
            )
            da, dlogprob, dentropy = logits_to_actions(
                d, batchelem["dactions"].detach().view(-1)
            )

            advantages = batchelem["returns"].detach().view(-1) - v.detach().view(-1)
            advantages = advantages.detach()

            with torch.no_grad():
                num_valid = (advantages > 0).sum().item()
                mask = (advantages > 0).float()
            ppg_loss = -plogprob * advantages * mask
            ppg_loss = ppg_loss.sum() / max(num_valid, 1)

            dpg_loss = -dlogprob * advantages * mask
            dpg_loss = dpg_loss.sum() / max(num_valid, 1)

            # Value loss
            v_loss = ((batchelem["returns"].detach().view(-1) - v.view(-1)) * mask).pow(
                2
            ).sum() / max(num_valid, 1)

            entropy_loss = (pentropy * mask).sum() + (dentropy * mask).sum()
            entropy_loss = entropy_loss / max(num_valid, 1)

            loss = (
                (ppg_loss + dpg_loss)
                - args.ent_coef * entropy_loss
                + v_loss * args.vf_coef
            )

            print("Num valid: ", num_valid)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(h.parameters(), args.max_grad_norm)
            optimizer.step()


for epoch in range(args.num_iterations):
    print("Epoch: ", epoch)
    batch_info = collect_batch(graphs_per_epoch, sim, h, global_step=epoch)
    batch_update(batch_info, args.update_epochs, h, optimizer, global_step=epoch)
    replay(batch_info, h, optimizer, global_step=epoch)

    # --- Gradient Monitoring ---
    total_norm = 0.0
    for name, param in h.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            writer.add_scalar(f"gradients/{name}_norm", param_norm, epoch)
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)
    total_norm = total_norm**0.5
    writer.add_scalar("gradients/total_norm", total_norm, epoch)

    for name, param in h.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            writer.add_scalar(f"grad_norms/{name}", grad_norm, epoch)

    for name, param in h.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"gradients_flow/{name}", param.grad.mean().item(), epoch)

    for name, param in h.named_parameters():
        writer.add_histogram(f"parameters/{name}", param, epoch)

    for name, param in h.named_parameters():
        writer.add_scalar(f"parameters_norm/{name}", param.data.norm(2).item(), epoch)
        writer.add_scalar(f"parameters_std/{name}", param.data.std().item(), 0)

    for i, param_group in enumerate(optimizer.param_groups):
        for j, param in enumerate(param_group["params"]):
            if param.grad is not None:
                state = optimizer.state[param]
                if "exp_avg" in state:
                    writer.add_scalar(
                        f"optimizer/group_{i}/param_{j}_exp_avg",
                        state["exp_avg"].mean().item(),
                        epoch,
                    )
                if "exp_avg_sq" in state:
                    writer.add_scalar(
                        f"optimizer/group_{i}/param_{j}_exp_avg_sq",
                        state["exp_avg_sq"].mean().item(),
                        epoch,
                    )

    # Save the model
    if (epoch + 1) % 500 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": h.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"runs/{run_name}/checkpoint_epoch_{epoch+1}.pth",
        )

    writer.flush()

writer.close()

# save pytorch model
torch.save(h.state_dict(), "model.pth")
torch.save(h, "model.torch")
