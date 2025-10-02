from tensordict import TensorDict
import torch
from typing import Callable, Optional
from torchrl.envs import set_exploration_type, ExplorationType
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict
from task4feedback.graphs.mesh.plot import animate_mesh_graph, PlotConfig, ColorConfig
from dataclasses import dataclass, field
import wandb
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from task4feedback.logging import training
import os
import git
from task4feedback.ml.env import RuntimeEnv
import pickle
from torchrl.envs import step_mdp
import math


def compute_advantage(td: TensorDict):
    with torch.no_grad():
        state_values = td["state_value"].view(-1)
        traj_ids = td["collector", "traj_ids"].view(-1)
        rewards = td["next", "reward"].view(-1)
        # Sum the rewards along each trajectory
        # td["value_target"] = cumulative reward at each step
        cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
        for traj in traj_ids.unique():
            mask = traj_ids == traj
            traj_rewards = rewards[mask]
            # each element = sum of rewards from that step to the end of the trajectory
            traj_cum_rewards = torch.flip(torch.cumsum(torch.flip(traj_rewards, dims=[0]), dim=0), dims=[0])
            cumulative_rewards[mask] = traj_cum_rewards.to(torch.float32)

        td["value_target"] = cumulative_rewards.unsqueeze(1)
        td["advantage"] = cumulative_rewards - state_values
    return td


def redistribute_rewards_uniform(td: TensorDict) -> TensorDict:
    """
    Redistribute each non‐zero (bagged) reward uniformly across the
    preceding zero‐reward steps of the same trajectory.
    """
    # [workers, time_dim, 1]
    rewards = td.get(("next", "reward"))
    # [workers, time_dim]
    traj_ids = td.get(("collector", "traj_ids"))
    device = rewards.device

    W, T, _ = rewards.shape
    new_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)

    with torch.no_grad():
        for w in range(W):
            # shape: [time_dim]
            local_r = rewards[w, :, 0]
            local_traj = traj_ids[w]
            buffer = torch.zeros_like(local_r, dtype=torch.float32, device=device)

            for traj in local_traj.unique():
                mask = local_traj == traj
                if not mask.any():
                    continue

                # pull out just this trajectory’s rewards
                traj_r = local_r[mask]  # shape: [n_steps_for_traj]

                # reverse so that we can group “backwards from each nonzero”
                r_rev = traj_r.flip(0)
                # group‐id increases by 1 at each nonzero
                groups = torch.cumsum(r_rev != 0, dim=0)
                # number of distinct groups = max_group_id + 1
                n_groups = int(groups.max().item()) + 1

                # accumulate sums and counts per group
                sum_per_group = torch.zeros(n_groups, device=device)
                cnt_per_group = torch.zeros(n_groups, device=device)
                sum_per_group.scatter_add_(0, groups, r_rev)
                cnt_per_group.scatter_add_(0, groups, torch.ones_like(r_rev))

                # compute the uniform share for each reversed position
                avg_per_group = sum_per_group / cnt_per_group
                share_rev = avg_per_group[groups]  # shape: same as r_rev

                # flip back to original order
                share = share_rev.flip(0)
                # write into the buffer at the masked positions
                buffer[mask] = share

            # assign back into the [workers, time_dim, 1] tensor
            new_rewards[w, :, 0] = buffer

        # overwrite the tensordict’s reward field
        td.set(("next", "reward"), new_rewards)

    return td


def compute_gae(tensordict_data, gamma: float = 0.99, lam: float = 0.95):
    """
    Vectorized GAE computation that handles multiple trajectories efficiently.

    Args:
        tensordict_data: TensorDict containing:
            - "state_value": State values [batch_size, 1] or [batch_size]
            - ("next", "reward"): Rewards [batch_size, 1] or [batch_size]
            - ("next", "done"): Done flags [batch_size, 1] or [batch_size]
            - ("collector", "traj_ids"): Trajectory IDs [batch_size, 1] or [batch_size]
        gamma: Discount factor
        lam: GAE lambda parameter

    Returns:
        tensordict_data: Updated with "advantage" and "value_target"
    """
    with torch.no_grad():
        # Extract and flatten tensors
        values = tensordict_data["state_value"].squeeze(-1)  # [batch_size]
        rewards = tensordict_data["next", "reward"].squeeze(-1)  # [batch_size]
        dones = tensordict_data["next", "done"].squeeze(-1)  # [batch_size]
        traj_ids = tensordict_data["collector", "traj_ids"].squeeze(-1)  # [batch_size]

        batch_size = values.shape[0]
        device = values.device

        # Initialize outputs
        advantages = torch.zeros_like(values)
        value_targets = torch.zeros_like(values)

        # Get unique trajectory IDs and their positions
        unique_traj_ids = torch.unique(traj_ids)

        # Process each trajectory
        for traj_id in unique_traj_ids:
            # Get mask for current trajectory
            traj_mask = traj_ids == traj_id
            traj_indices = torch.where(traj_mask)[0]

            # Extract trajectory data
            traj_values = values[traj_mask]
            traj_rewards = rewards[traj_mask]
            traj_dones = dones[traj_mask]

            T = len(traj_values)

            # Compute next values for this trajectory
            next_values = torch.zeros_like(traj_values)
            # For all steps except the last, next value is the next state's value if not done
            next_values[:-1] = traj_values[1:] * (~traj_dones[:-1])
            # Last step always has next_value = 0 (episode boundary or terminal)

            # Compute TD residuals
            deltas = traj_rewards + gamma * next_values - traj_values

            # Compute GAE backwards through trajectory
            traj_advantages = torch.zeros_like(traj_values)
            gae = 0.0

            for t in reversed(range(T)):
                # GAE formula: A_t = δ_t + γλ(1-done_t)A_{t+1}
                gae = deltas[t] + gamma * lam * gae * (~traj_dones[t])
                traj_advantages[t] = gae

            # Store results back to full arrays
            advantages[traj_mask] = traj_advantages
            value_targets[traj_mask] = traj_advantages + traj_values

        # Update tensordict with computed advantages and value targets
        tensordict_data["advantage"] = advantages.unsqueeze(-1)
        tensordict_data["value_target"] = value_targets.unsqueeze(-1)

        return tensordict_data


def logits_to_action(logits: torch.Tensor, action):
    probs = torch.distributions.Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self, verbose: bool = True, name: Optional[str] = None):
        self.verbose = verbose
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.verbose:
            name_str = f" {self.name}" if self.name else ""
            print(f"Timer{name_str}: {self.interval:.4f} seconds")

    def get_elapsed(self) -> float:
        """Returns the elapsed time in seconds."""
        return self.interval


def log_parameter_and_gradient_norms(model):
    """Log parameter and gradient norms to wandb"""
    param_norms = {}
    grad_norms = {}

    total_param_norm = 0.0
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.detach().norm().item()
            param_norms[f"param_norm/{name}"] = param_norm
            total_param_norm += param_norm**2

            if param.grad is not None:
                grad_norm = param.grad.detach().norm().item()
                grad_norms[f"grad_norm/{name}"] = grad_norm
                total_grad_norm += grad_norm**2

    total_param_norm = total_param_norm**0.5
    total_grad_norm = total_grad_norm**0.5

    return {
        **param_norms,
        **grad_norms,
        "param_norm/total": total_param_norm,
        "grad_norm/total": total_grad_norm,
    }


def make_eval_envs(
    eval_env_fn: list[Callable],
):
    return [eval_env_fn(eval=True) for eval_env_fn in eval_env_fn]


@dataclass
class EvaluationConfig:
    eval_interval: int = 100
    animation_interval: int = 100
    max_frames: int = 100
    fig_size: tuple[int, int] = (4, 4)
    dpi: int = 50
    bitrate: int = 50
    exploration_types: list[str] = field(default_factory=lambda: ["RANDOM", "DETERMINISTIC"])
    samples: int = 10
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    video_seconds: int = 15
    pickle_path: Optional[str] = None
    pickled_states: Optional[Dict[str, Any]] = None


def eval_pickled_env(
    n_collections: int,
    policy,
    env,
    exploration_type: ExplorationType,
    pickled_states: Dict,
    samples: int = 1,
):
    env_rewards = []
    env_times = []
    env_vsEFT = []
    env_vsPolicy = []
    metrics = {}
    last_env = None
    for i in range(samples):
        env.reset_for_evaluation()
        env.disable_reward()
        with set_exploration_type(exploration_type), torch.no_grad():
            saved_loc = pickled_states["init_locs"][i % len(pickled_states["init_locs"])]
            workload = pickled_states["workloads"][i % len(pickled_states["workloads"])]
            env.reset_to_state(saved_loc, workload)
            tensordict = env.rollout(policy=policy, max_steps=100000)

        if "next" in tensordict and "reward" in tensordict["next"]:
            rewards = tensordict["next", "reward"]
            avg_reward = rewards.mean().item()
            env_rewards.append(avg_reward)

        if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
            completion_time = env.simulator.time
            env_times.append(completion_time)
            if n_collections == 0:
                if pickled_states["eft_times"][i] != env._get_baseline("EFT"):
                    training.warning(f"Environment {i} EFT time mismatch: {pickled_states['eft_times'][i]} " f"!= {env._get_baseline('EFT')}")
                else:
                    training.info(f"Environment {i} EFT time match: {pickled_states['eft_times'][i]} == {env._get_baseline('EFT')}")

        env_vsEFT.append(pickled_states["eft_times"][i] / completion_time if completion_time > 0 else 0.0)
        env_vsPolicy.append(pickled_states["policy_times"][i] / completion_time if completion_time > 0 else 0.0)
        env.enable_reward()

    # time metrics
    if samples > 1:
        mean_time = sum(env_times) / len(env_times) if env_times else 0
        std_time = torch.std(torch.tensor(env_times, dtype=torch.float64)).item() if env_times else 0.0
        metrics["std_time"] = std_time
    else:
        mean_time = env_times[0] if env_times else 0.0
        std_time = 0.0
    metrics["mean_time"] = mean_time

    # vsEFT / vsQuad metrics
    if env_vsEFT:
        metrics["mean_vsEFT"] = sum(env_vsEFT) / len(env_vsEFT)
        metrics["std_vsEFT"] = torch.std(torch.tensor(env_vsEFT, dtype=torch.float64)).item()
    else:
        metrics["mean_vsEFT"] = 0.0
        metrics["std_vsEFT"] = 0.0

    if env_vsPolicy:
        metrics["mean_vsPolicy"] = sum(env_vsPolicy) / len(env_vsPolicy)
        metrics["std_vsPolicy"] = torch.std(torch.tensor(env_vsPolicy, dtype=torch.float64)).item()
    else:
        metrics["mean_vsPolicy"] = 0.0
        metrics["std_vsPolicy"] = 0.0

    training.info(f"Evaluation results: mean_time={mean_time}, " f"mean_vsEFT={metrics['mean_vsEFT']}, mean_vsPolicy={metrics['mean_vsPolicy']}")

    last_env = env

    return metrics, last_env

def eval_env(n_collections: int, policy, env, exploration_type: ExplorationType, samples: int = 1, seed: int = 0):
    env_rewards = []
    env_times = []
    metrics = {}
    last_env = None

    for _ in range(samples):
        env.reset_for_evaluation(seed=seed)
        env.disable_reward()
        with set_exploration_type(exploration_type), torch.no_grad():
            # check_env_specs(env)
            # n_tasks = len(env.get_graph())
            # decision_per_epoch = 100
            # epochs = int(math.ceil(n_tasks / decision_per_epoch ))
            # input_td = None

            # for epoch in range(epochs):
            #     if input_td is None:
            #         tensordict = env.rollout(
            #             policy=policy,
            #             max_steps=100,
            #             return_contiguous=False,
            #         )
            #     else:
            #         tensordict = env.rollout(
            #             policy=policy,
            #             max_steps=100,
            #             return_contiguous=False,
            #             auto_reset=False,
            #             tensordict=input_td
            #         )
            #     input_td = step_mdp(tensordict[..., -1])
            #     print(f"Eval epoch {epoch+1}/{epochs} completed")
            start_t = time.perf_counter()
            tensordict = env.rollout(
                policy=policy,
                max_steps=100000,
            )

        if "next" in tensordict and "reward" in tensordict["next"]:
            rewards = tensordict["next", "reward"]
            avg_reward = rewards.mean().item()
            env_rewards.append(avg_reward)

        if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
            completion_time = env.simulator.time
            env_times.append(completion_time)
        env.enable_reward()

    if samples > 1:
        mean_time = sum(env_times) / len(env_times) if env_times else 0
        std_time = torch.std(torch.tensor(env_times, dtype=torch.float64)).item() if env_times else 0.0
        metrics["std_time"] = std_time
    else:
        mean_time = env_times[0] if env_times else 0.0
        std_time = 0.0

    metrics["mean_time"] = mean_time

    training.info(f"Evaluation results: mean_time={mean_time}")

    last_env = env

    return metrics, last_env



def evaluate_policy(n_collections: int, policy, eval_envs: list[RuntimeEnv], config: EvaluationConfig, exploration_type: str, metrics: dict) -> list[RuntimeEnv]:

    env = None
    metrics[f"eval/{str(exploration_type)}"] = {}

    for i, env in enumerate(eval_envs):

        if env is None:
            training.warning(f"Environment {i} is None, skipping evaluation.")
            continue

        if not hasattr(env, "reset_for_evaluation"):
            training.warning(f"Environment {i} does not have reset_for_evaluation method, skipping evaluation.")
            continue

        # if n_collections == 0 and hasattr(env, "get_graph") and hasattr(env.get_graph(), "get_workload"):
        #     training.info("Generating initial workload plot for environment {i}")
        #     env.reset_for_evaluation(seed=config.seeds[0])
        #     title = f"workload_env_{i}.mp4"
        #     workload = env.get_graph().get_workload()
        #     workload.animate_workload(title=title, show=False, bitrate=config.bitrate, video_seconds=config.video_seconds, figsize=config.fig_size, dpi=config.dpi)

        #     if wandb is None or wandb.run is None or wandb.run.dir is None:
        #         path = "."
        #     else:
        #         path = wandb.run.dir

        #     video_path = Path(path) / f"{title}"
        #     metrics[f"eval/env_{i}_workload"] = wandb.Video(
        #         video_path,
        #         caption=f"Env {i} Workload",
        #         fps=len(workload.levels) / (config.max_frames / 30),
        #         format="mp4",
        #     )

        if exploration_type == "RANDOM":
            exploration_type_enum = ExplorationType.RANDOM
        elif exploration_type == "DETERMINISTIC":
            exploration_type_enum = ExplorationType.DETERMINISTIC
        else:
            raise ValueError(f"Unknown exploration type: {exploration_type}")

        if config.pickle_path is not None:
            if config.pickled_states is None:
                try:
                    with open(config.pickle_path, "rb") as f:
                        config.pickled_states = pickle.load(f)
                except FileNotFoundError:
                    print(f"[ERROR] Pickle file not found: {config.pickle_path} - skipping pickled evaluation")
                    config.pickle_path = None # disable for future calls
                    config.pickled_states = None
            if config.pickled_states is not None:
                env_eval_metrics, output_env = eval_pickled_env(n_collections, policy, env, exploration_type_enum, samples=config.samples, pickled_states=config.pickled_states)
                metrics[f"eval/{str(exploration_type)}"] = env_eval_metrics
                return [output_env]

        for seed in config.seeds:
            metrics[f"eval/{str(exploration_type)}"][f"env_{i}_{seed}"] = {}
            training.info(f"Evaluating environment {i, seed} with {str(exploration_type)} policy")
            env_eval_metrics, output_env = eval_env(n_collections, policy, env, exploration_type_enum, samples=config.samples if exploration_type == "RANDOM" else 1, seed=seed)
            metrics[f"eval/{str(exploration_type)}"][f"env_{i}_{seed}"] = env_eval_metrics

    return [output_env]


def visualize_envs(n_collections: int, viz_envs: list[RuntimeEnv], config: EvaluationConfig, exploration_type: str, video_log: dict):
    for i, env in enumerate(viz_envs):
        assert env is not None
        training.info(f"Visualizing environment {i} with policy {exploration_type} at n_updates={n_collections}")
        title = f"network_eval_{exploration_type}_{n_collections}"

        plot_config = PlotConfig(
            use_labels=False,
            use_duration_shading=True,
            dpi=config.dpi,
            figsize=config.fig_size,
            bitrate=config.bitrate,
            video_seconds=config.video_seconds,
            n_frames=config.max_frames,
        )

        color_config = ColorConfig()

        if wandb is None or wandb.run is None or wandb.run.dir is None:
            path = "."
        else:
            path = wandb.run.dir

        animate_mesh_graph(
            env,
            plot_cfg=plot_config,
            color_cfg=color_config,
            folder=path,
            filename=f"{title}.mp4",
        )

        video_path = Path(path) / f"{title}.mp4"
        video_log[f"eval/video/{i}/{exploration_type}"] = wandb.Video(
            video_path,
            caption=f"Env {i}, {exploration_type} evaluation at n_collections={n_collections}",
            fps=config.max_frames,
            format="mp4",
        )


def run_evaluation(
    policy,
    eval_envs: list[Callable],
    config: EvaluationConfig,
    n_collections: int = 0,
    n_updates: int = 0,
    n_samples: int = 0,
):
    metrics = {}
    video_log = {}

    for exploration_type in config.exploration_types:
        viz_envs = evaluate_policy(n_collections, policy, eval_envs, config, exploration_type, metrics)

        if (config.animation_interval > 0) and (n_collections % config.animation_interval == 0):
            visualize_envs(n_collections, viz_envs, config, exploration_type, video_log)

    wandb.log(
        {
            **metrics,
            **video_log,
            "batch/n_updates": n_updates,
            "batch/n_collections": n_collections,
            "batch/n_samples": n_samples,
        }
    )

    return metrics


def save_checkpoint(
    step, policy_module, value_module, optimizer, lr_scheduler=None, extras: Optional[Dict[str, Any]] = None, checkpoint_dir: Optional[str] = None, filename: Optional[str] = None, wandb=None
) -> Path:
    try:
        state = dict(
            step=step,
            policy_module=policy_module.state_dict(),
            value_module=value_module.state_dict(),
            optimizer=optimizer.state_dict(),
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            extras=extras or {},
            commit_hash=git.Repo(search_parent_directories=True).head.object.hexsha,
            commit_dirty=git.Repo(search_parent_directories=True).is_dirty(),
        )
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
        elif wandb is not None and wandb.run is not None and wandb.run.dir is not None:
            checkpoint_dir = Path(wandb.run.dir)
        else:
            checkpoint_dir = Path(os.environ.get("HYDRA_RUNTIME_OUTPUT_DIR", "."))

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if filename is not None:
            checkpoint_file = checkpoint_dir / filename
        else:
            checkpoint_file = checkpoint_dir / f"checkpoint_{step}.pt"

        torch.save(state, checkpoint_file)
        training.info(f"Checkpoint saved to {checkpoint_file}")

        return checkpoint_file

    except Exception as e:
        training.error(f"Failed to save checkpoint at step {step}: {e}")
        raise
