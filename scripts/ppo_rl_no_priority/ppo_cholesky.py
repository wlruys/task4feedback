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
)
from task4feedback.fastsim.models import (
    TaskAssignmentNetDeviceOnly,
    VectorTaskAssignmentNet,
)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import os
import wandb

run_name = f"ppo_cholesky_(4x4)_PreTrained_PriorRand"
# generate folder if "runs/{run_name}" does not exist
if not os.path.exists(f"runs/{run_name}"):
    os.makedirs(f"runs/{run_name}")


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
    gae_lambda: float = 1
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 1000
    """the number of iterations (computed in runtime)"""

    graphs_per_update: int = 50
    """the number of graphs to use for each update"""
    reward: str = "percent_improvement"

    devices = 4
    vcus = 1
    blocks = 4


args = Args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

wandb.init(
    project="cholesky_graph",
    name=run_name,
    config={
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "num_minibatches": args.num_minibatches,
        "update_epochs": args.update_epochs,
        "norm_adv": args.norm_adv,
        "clip_coef": args.clip_coef,
        "clip_vloss": args.clip_vloss,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "batch_size": args.batch_size,
        "minibatch_size": args.minibatch_size,
        "num_iterations": args.num_iterations,
        "graphs_per_update": args.graphs_per_update,
        "reward": args.reward,
        "devices": args.devices,
        "vcus": args.vcus,
        "blocks": args.blocks,
    },
)


def initialize_simulator(seed=0):
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
        seed=seed,
    )
    sim = H.create_simulator()
    sim.initialize(use_data=True)
    sim.randomize_durations()
    sim.enable_python_mapper()

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
        data = simulator.observer.local_graph_features(candidates, k_hop=1)
        with torch.no_grad():
            d, v = self.model.forward(data)
            # Choose argmax of network output for priority and device assignment
            dev_per_task = torch.argmax(d, dim=-1)
            action_list = []
            for i in range(len(candidates)):
                # Check if p_per_task and dev_per_task are scalars
                if dev_per_task.dim() == 0:
                    dev_task = dev_per_task.item()
                else:
                    dev_task = dev_per_task[i].item()
                a = Action(
                    candidates[i],
                    i,
                    dev_task,
                    0,
                    0,
                )
                action_list.append(a)
        return action_list


class RandomNetworkMapper(PythonMapper):

    def __init__(self, model):
        self.model = model

    def map_tasks(self, candidates: np.ndarray[np.int32], simulator, output=None):
        data = simulator.observer.local_graph_features(candidates, k_hop=1)

        with torch.no_grad():
            self.model.eval()
            d, v = self.model.forward(data)
            self.model.train()

            # sample from network output
            dev_per_task, dlogprob, _ = logits_to_actions(d)

            if output is not None:
                output["candidates"] = candidates
                output["state"] = data
                output["dlogprob"] = dlogprob
                output["value"] = v
                output["dactions"] = dev_per_task

            action_list = []
            for i in range(len(candidates)):

                a = Action(
                    candidates[i],
                    i,
                    dev_per_task,
                    0,
                    0,
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
h = TaskAssignmentNetDeviceOnly(args.devices, args.hidden_dim, local_graph)
optimizer = optim.Adam(h.parameters(), lr=lr)
netmap = GreedyNetworkMapper(h)
rnetmap = RandomNetworkMapper(h)
H.set_python_mapper(netmap)
backup = H.copy(sim)
# h.apply(init_weights)
h.load_state_dict(
    torch.load(
        "/Users/jaeyoung/work/task4feedback/scripts/ppo_rl_no_priority/runs/ppo_random_task15_50graphs_long_(5x10)per20/model.pth",
        map_location=torch.device("cpu"),
        weights_only=True,
    )
)


def collect_batch(episodes, sim, h, global_step=0):
    batch_info = []
    for e in range(0, episodes):
        sim.randomize_priorities()
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

            if done:
                if args.reward == "percent_improvement":
                    percent_improvement = (
                        1 + (baseline_time - record["time"]) / baseline_time
                    )
                    percent_improvement = percent_improvement
                    record["reward"] = percent_improvement
                elif args.reward == "better":
                    if record["time"] < baseline_time:
                        record["reward"] = 1
                    else:
                        record["reward"] = 0

                wandb.log(
                    {"episode_reward": record["reward"]},
                )

                break
            else:
                record["reward"] = 0

        with torch.no_grad():
            for t in range(len(episode_info)):
                episode_info[t]["returns"] = episode_info[-1]["reward"]
                episode_info[t]["advantage"] = (
                    episode_info[-1]["reward"] - episode_info[t]["value"]
                )

        batch_info.extend(episode_info)
    return batch_info


LI = 0


def batch_update(batch_info, update_epoch, h, optimizer, global_step):
    n_obs = len(batch_info)

    batch_size = args.batch_size

    dclipfracs = []

    state = []
    for i in range(n_obs):
        state.append(batch_info[i]["state"])
        state[i]["dlogprob"] = batch_info[i]["dlogprob"]
        state[i]["value"] = batch_info[i]["value"]
        state[i]["dactions"] = batch_info[i]["dactions"]
        state[i]["advantage"] = batch_info[i]["advantage"]
        state[i]["returns"] = batch_info[i]["returns"]

    global LI

    for k in range(update_epoch):
        nbatches = args.num_minibatches
        batch_size = n_obs // nbatches
        loader = DataLoader(state, batch_size=batch_size, shuffle=True)

        for i, batch in enumerate(loader):
            out = h(batch, batch["tasks"].batch)
            d, v = out

            da, dlogprob, dentropy = logits_to_actions(
                d, batch["dactions"].detach().view(-1)
            )

            dlogratio = dlogprob.view(-1) - batch["dlogprob"].detach().view(-1)
            dratio = dlogratio.exp()

            with torch.no_grad():
                dold_approx_kl = (-dlogratio).mean()
                dapprox_kl = ((dratio - 1) - dlogratio).mean()
                dclipfracs += [
                    ((dratio - 1.0).abs() > args.clip_coef).float().mean().item()
                ]

            mb_advantages = batch["advantage"].detach().view(-1)

            dpg_loss1 = mb_advantages * dratio.view(-1)
            dpg_loss2 = mb_advantages * torch.clamp(
                dratio.view(-1), 1 - args.clip_coef, 1 + args.clip_coef
            )
            dpg_loss = torch.min(dpg_loss1, dpg_loss2).mean()

            newvalue = v.view(-1)
            v_loss_unclipped = (newvalue - batch["returns"].detach().view(-1)) ** 2
            v_clipped = batch["value"].detach().view(-1) + torch.clamp(
                newvalue - batch["value"].detach().view(-1),
                -args.clip_coef,
                args.clip_coef,
            )
            v_loss_clipped = (v_clipped - batch["returns"].detach().view(-1)) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            entropy_loss = dentropy.mean()
            loss = (
                -1 * (dpg_loss) - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(h.parameters(), args.max_grad_norm)
            optimizer.step()

            wandb.log(
                {
                    "losses/value_loss": v_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/dentropy": dentropy.mean().item(),
                },
            )
            wandb.log(
                {
                    "losses/dratio": dratio.mean().item(),
                    "losses/dpolicy_loss": dpg_loss.item(),
                    "losses/dold_approx_kl": dold_approx_kl.item(),
                    "losses/dapprox_kl": dapprox_kl.item(),
                    "losses/dclipfrac": np.mean(dclipfracs),
                },
            )
            LI = LI + 1


for epoch in range(args.num_iterations):
    print("Epoch: ", epoch)

    batch_info = collect_batch(graphs_per_epoch, sim, h, global_step=epoch)
    batch_update(batch_info, args.update_epochs, h, optimizer, global_step=epoch)

    # --- Gradient Monitoring ---
    total_norm = 0.0
    for name, param in h.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
    total_norm = total_norm**0.5
    # Save the model
    if (epoch + 1) % 1000 == 0:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": h.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"runs/{run_name}/checkpoint_epoch_{epoch+1}.pth",
        )
        torch.save(h, f"runs/{run_name}/model_epoch_{epoch+1}.torch")

# save pytorch model
torch.save(h.state_dict(), f"runs/{run_name}/model.pth")
torch.save(h, f"runs/{run_name}/model.torch")
