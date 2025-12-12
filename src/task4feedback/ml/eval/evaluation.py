from __future__ import annotations

import math
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import wandb
from torchrl.envs import ExplorationType, set_exploration_type

from task4feedback.graphs.mesh.plot import ColorConfig, PlotConfig, animate_mesh_graph
from task4feedback.logging import training
from task4feedback.ml.env import RuntimeEnv


def make_eval_envs(eval_env_fn: List[Callable]) -> List[RuntimeEnv]:
    return [fn(eval=True) for fn in eval_env_fn]


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


def eval_pickled_env(
    n_collections: int,
    policy,
    env,
    exploration_type: ExplorationType,
    eval_location=None,
    samples: int = 1,
):
    env_rewards = []
    env_times = []
    metrics: Dict[str, Any] = {}

    if eval_location is None or not Path(eval_location.file_path).exists():
        raise FileNotFoundError(f"Pickled eval file not found at {getattr(eval_location, 'file_path', None)}")

    eval_state = pickle.load(open(eval_location.file_path, "rb"))

    eft_policy_str = "EFT"

    workloads = eval_state.workloads[eft_policy_str]
    locations = eval_state.init_locs[eft_policy_str]

    last_env = None
    vsPolicy = defaultdict(list)

    for i in range(samples):
        env.reset_for_evaluation()
        env.disable_reward()
        with set_exploration_type(exploration_type), torch.no_grad():
            saved_loc = locations[i % len(locations)]
            workload = workloads[i % len(workloads)]
            env.reset_to_state(saved_loc, workload)
            tensordict = env.rollout(policy=policy, max_steps=100000)

        if "next" in tensordict and "reward" in tensordict["next"]:
            rewards = tensordict["next", "reward"]
            avg_reward = rewards.mean().item()
            env_rewards.append(avg_reward)

        if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
            completion_time = env.simulator.time
            env_times.append(completion_time)

            saved_eft_time = eval_state.policy_times[eft_policy_str][i % len(workloads)]
            observed_eft_time = env._get_baseline("EFT")

            # Tight tolerance to detect non-determinism or eval file corruption
            EFT_REL_TOL = 1e-6  # 0.0001%
            EFT_ABS_TOL = 1e-9  # Absolute for near-zero times

            if not math.isclose(saved_eft_time, observed_eft_time,
                               rel_tol=EFT_REL_TOL, abs_tol=EFT_ABS_TOL):
                delta = abs(saved_eft_time - observed_eft_time)
                rel_error = delta / saved_eft_time if saved_eft_time > 0 else float('inf')

                training.error(
                    f"EFT time mismatch for environment {i}:\n"
                    f"  Expected: {saved_eft_time:.10f}\n"
                    f"  Observed: {observed_eft_time:.10f}\n"
                    f"  Delta:    {delta:.10f} ({rel_error*100:.6f}%)\n"
                    f"This indicates non-deterministic execution or eval file corruption."
                )
                raise ValueError("EFT validation failed - evaluation aborted")

        env.enable_reward()

    if samples > 1:
        mean_time = sum(env_times) / len(env_times) if env_times else 0
        std_time = torch.std(torch.tensor(env_times, dtype=torch.float64)).item() if env_times else 0.0
        metrics["std_time"] = std_time
    else:
        mean_time = env_times[0] if env_times else 0.0
        std_time = 0.0
    metrics["mean_time"] = mean_time

    for policy_str, policy_times in eval_state.policy_times.items():
        sum_env_times = sum(env_times)
        sum_policy_times = sum([policy_times[i % len(policy_times)] for i in range(len(env_times))])
        metrics[f"mean_vs_{policy_str}"] = sum_policy_times / sum_env_times if sum_env_times > 0 else 0.0

    training.info(
        f"Evaluation results: mean_time={mean_time} +/- {std_time}"
        + ", ".join([f"{k}={v:.2f}" for k, v in metrics.items() if k.startswith('mean_vs_')])
    )

    last_env = env
    return metrics, last_env


def eval_env(
    n_collections: int,
    policy,
    env,
    exploration_type: ExplorationType,
    samples: int = 1,
    seed: int = 0,
):
    env_rewards = []
    env_times = []
    metrics: Dict[str, Any] = {}
    last_env = None

    for _ in range(samples):
        env.reset_for_evaluation(seed=seed)
        env.disable_reward()
        with set_exploration_type(exploration_type), torch.inference_mode():
            tensordict = env.rollout(policy=policy, max_steps=100000)

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

    eft_baseline = env._get_baseline("EFT") if hasattr(env, "_get_baseline") else None
    if eft_baseline is not None and eft_baseline > 0:
        metrics["mean_vs_EFT"] = eft_baseline / mean_time if mean_time > 0 else 0.0

    training.info(
        f"Evaluation results: mean_time={mean_time} +/- {std_time}, "
        f"mean_vs_EFT={metrics.get('mean_vs_EFT', 'N/A')}"
    )

    last_env = env
    return metrics, last_env


def evaluate_policy(
    n_collections: int,
    policy,
    eval_envs: List[RuntimeEnv],
    config: EvaluationConfig,
    exploration_type: str,
    metrics: Dict[str, Any],
    eval_location=None,
) -> List[RuntimeEnv]:
    env = None
    metrics[f"eval/{str(exploration_type)}"] = {}

    for i, env in enumerate(eval_envs):
        if env is None:
            training.warning("Environment %s is None, skipping evaluation.", i)
            continue

        if not hasattr(env, "reset_for_evaluation"):
            training.warning("Environment %s does not have reset_for_evaluation method, skipping evaluation.", i)
            continue

        if exploration_type == "RANDOM":
            exploration_type_enum = ExplorationType.RANDOM
        elif exploration_type == "DETERMINISTIC":
            exploration_type_enum = ExplorationType.DETERMINISTIC
        else:
            raise ValueError(f"Unknown exploration type: {exploration_type}")

        if eval_location is not None:
            training.info("Evaluating pickled environment from %s", eval_location)
            env_eval_metrics, output_env = eval_pickled_env(n_collections, policy, env, exploration_type_enum, samples=config.samples, eval_location=eval_location)
            metrics[f"eval/{str(exploration_type)}"] = env_eval_metrics
            return [output_env]

        for seed in config.seeds:
            metrics[f"eval/{str(exploration_type)}"][f"env_{i}_{seed}"] = {}
            training.info("Evaluating environment %s with %s policy", (i, seed), str(exploration_type))
            env_eval_metrics, output_env = eval_env(n_collections, policy, env, exploration_type_enum, samples=config.samples if exploration_type == "RANDOM" else 1, seed=seed)
            metrics[f"eval/{str(exploration_type)}"][f"env_{i}_{seed}"] = env_eval_metrics

    return [output_env]


def visualize_envs(
    n_collections: int,
    viz_envs: List[RuntimeEnv],
    config: EvaluationConfig,
    exploration_type: str,
    video_log: Dict[str, Any],
):
    for i, env in enumerate(viz_envs):
        assert env is not None
        training.info("Visualizing environment %s with policy %s at n_updates=%s", i, exploration_type, n_collections)
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
    eval_envs: List[RuntimeEnv],
    config: EvaluationConfig,
    n_collections: int = 0,
    n_updates: int = 0,
    n_samples: int = 0,
    eval_location=None,
):
    metrics: Dict[str, Any] = {}
    video_log: Dict[str, Any] = {}

    for exploration_type in config.exploration_types:
        viz_envs = evaluate_policy(n_collections, policy, eval_envs, config, exploration_type, metrics, eval_location=eval_location)

        if (config.animation_interval > 0) and (n_collections % config.animation_interval == 0):
            visualize_envs(n_collections, viz_envs, config, exploration_type, video_log)

    if wandb.run:
        wandb.log(
            {
                **metrics,
                **video_log,
                "batch/n_updates": n_updates,
                "batch/n_collections": n_collections,
                "batch/n_samples": n_samples,
            }
        )
    else:
        training.debug("Skipping wandb.log for evaluation because no active run was found.")

    return metrics


__all__ = [
    "EvaluationConfig",
    "make_eval_envs",
    "eval_pickled_env",
    "eval_env",
    "evaluate_policy",
    "visualize_envs",
    "run_evaluation",
]
