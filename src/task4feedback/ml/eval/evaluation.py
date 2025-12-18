from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import wandb
from torchrl.envs import ExplorationType, set_exploration_type

from task4feedback.graphs.mesh.plot import ColorConfig, PlotConfig, animate_mesh_graph
from task4feedback.logging import training
from task4feedback.ml.env import RuntimeEnv
from task4feedback.ml.eval.metrics import (
    METRIC_REGISTRY,
    MetricContext,
    aggregate_all_metrics,
    compute_metrics,
    resolve_metric_ids,
)


def make_eval_envs(eval_env_fn: List[Callable]) -> List[RuntimeEnv]:
    return [fn(eval=True) for fn in eval_env_fn]


@dataclass
class EvaluationConfig:
    eval_interval: int = 100
    animation_interval: int = 100
    max_rollout_steps: int = 100000
    max_frames: int = 100
    fig_size: tuple[int, int] = (4, 4)
    dpi: int = 50
    bitrate: int = 50
    exploration_types: list[str] = field(default_factory=lambda: ["RANDOM", "DETERMINISTIC"])
    samples: int = 10
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    video_seconds: int = 15
    metrics: list[str] = field(default_factory=lambda: ["mean_time", "vs_baseline"])
    metric_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metric_intervals: Dict[str, int] = field(default_factory=dict)
    aggregations: Dict[str, List[str]] = field(default_factory=dict)
    log_raw_per_env: bool = False
    log_raw_per_seed: bool = False
    log_videos: bool = True
    best_metric: str | None = None
    best_metric_mode: str = "max"
    eval_timeout_s: int = 0
    device: str = "cpu"
    deterministic_eval: bool = True


def eval_pickled_env(
    n_collections: int,
    policy,
    env,
    exploration_type: ExplorationType,
    config: EvaluationConfig,
    eval_location=None,
    samples: int = 1,
    active_metrics: List[str] | None = None,
):
    env_rewards: List[float] = []
    env_times: List[float] = []
    metrics_by_id: Dict[str, Dict[str, float]] = {}

    if eval_location is None or not Path(eval_location.file_path).exists():
        raise FileNotFoundError(f"Pickled eval file not found at {getattr(eval_location, 'file_path', None)}")

    eval_state = pickle.load(open(eval_location.file_path, "rb"))

    eft_policy_str = "EFT"

    workloads = eval_state.workloads[eft_policy_str]
    locations = eval_state.init_locs[eft_policy_str]

    last_env = None

    for i in range(samples):
        env.reset_for_evaluation()
        env.disable_reward()
        with set_exploration_type(exploration_type), torch.no_grad():
            saved_loc = locations[i % len(locations)]
            workload = workloads[i % len(workloads)]
            env.reset_to_state(saved_loc, workload)
            tensordict = env.rollout(policy=policy, max_steps=config.max_rollout_steps)

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

    context = MetricContext(env_times=env_times, rewards=env_rewards, rollout=tensordict)
    active_metrics = active_metrics or resolve_metric_ids(config.metrics)
    metrics_by_id = compute_metrics(active_metrics, env, context, metric_params=config.metric_params)

    if "vs_baseline" in active_metrics:
        extra = metrics_by_id.setdefault("vs_baseline", {})
        for policy_str, policy_times in eval_state.policy_times.items():
            sum_env_times = sum(env_times)
            sum_policy_times = sum([policy_times[i % len(policy_times)] for i in range(len(env_times))])
            extra[f"mean_vs_{policy_str}"] = sum_policy_times / sum_env_times if sum_env_times > 0 else 0.0

    training.info(
        "Evaluation results from pickled env: "
        + ", ".join([f"{k}={v}" for d in metrics_by_id.values() for k, v in d.items()])
    )

    last_env = env
    return metrics_by_id, last_env


def eval_env(
    n_collections: int,
    policy,
    env,
    exploration_type: ExplorationType,
    config: EvaluationConfig,
    samples: int = 1,
    seed: int = 0,
    active_metrics: List[str] | None = None,
):
    env_rewards: List[float] = []
    env_times: List[float] = []
    metrics_by_id: Dict[str, Dict[str, float]] = {}
    last_env = None

    for _ in range(samples):
        env.reset_for_evaluation(seed=seed)
        env.disable_reward()
        with set_exploration_type(exploration_type), torch.inference_mode():
            tensordict = env.rollout(policy=policy, max_steps=config.max_rollout_steps)

        if "next" in tensordict and "reward" in tensordict["next"]:
            rewards = tensordict["next", "reward"]
            avg_reward = rewards.mean().item()
            env_rewards.append(avg_reward)

        if hasattr(env, "simulator") and hasattr(env.simulator, "time"):
            completion_time = env.simulator.time
            env_times.append(completion_time)
        env.enable_reward()

    context = MetricContext(env_times=env_times, rewards=env_rewards, rollout=tensordict)
    active_metrics = active_metrics or resolve_metric_ids(config.metrics)
    metrics_by_id = compute_metrics(active_metrics, env, context, metric_params=config.metric_params)

    training.info(
        "Evaluation results: "
        + ", ".join([f"{k}={v}" for d in metrics_by_id.values() for k, v in d.items()])
    )

    last_env = env
    return metrics_by_id, last_env


def _filter_metric_ids(config: EvaluationConfig, n_collections: int) -> List[str]:
    """Select active metric ids based on config and per-metric intervals."""
    candidates = resolve_metric_ids(config.metrics)
    active: List[str] = []
    for metric_id in candidates:
        interval = config.metric_intervals.get(metric_id)
        if interval and interval > 0 and n_collections % interval != 0:
            continue
        active.append(metric_id)
    return active


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
    exploration_key = str(exploration_type)

    if exploration_type == "RANDOM":
        exploration_type_enum = ExplorationType.RANDOM
    elif exploration_type == "DETERMINISTIC":
        exploration_type_enum = ExplorationType.DETERMINISTIC
    else:
        raise ValueError(f"Unknown exploration type: {exploration_type}")

    active_metrics = _filter_metric_ids(config, n_collections)
    per_run: List[Dict[str, Dict[str, float]]] = []
    output_envs: List[RuntimeEnv] = []

    for i, env in enumerate(eval_envs):
        if env is None:
            training.warning("Environment %s is None, skipping evaluation.", i)
            continue

        if not hasattr(env, "reset_for_evaluation"):
            training.warning("Environment %s does not have reset_for_evaluation method, skipping evaluation.", i)
            continue

        sample_count = config.samples if exploration_type_enum == ExplorationType.RANDOM else 1

        if eval_location is not None:
            training.info("Evaluating pickled environment from %s", eval_location)
            env_eval_metrics, output_env = eval_pickled_env(
                n_collections,
                policy,
                env,
                exploration_type_enum,
                config=config,
                samples=sample_count,
                eval_location=eval_location,
                active_metrics=active_metrics,
            )
            per_run.append(env_eval_metrics)
            output_envs.append(output_env)
            break

        for seed in config.seeds:
            training.info("Evaluating environment %s with %s policy", (i, seed), str(exploration_type))
            env_eval_metrics, output_env = eval_env(
                n_collections,
                policy,
                env,
                exploration_type_enum,
                config=config,
                samples=sample_count,
                seed=seed,
                active_metrics=active_metrics,
            )
            per_run.append(env_eval_metrics)
            output_envs.append(output_env)

            if config.log_raw_per_env or config.log_raw_per_seed:
                raw_prefix = f"eval/{exploration_key}/env_{i}"
                if config.log_raw_per_seed:
                    raw_prefix = f"{raw_prefix}_seed_{seed}"
                for metric_id, values in env_eval_metrics.items():
                    for key, value in values.items():
                        name = f"{raw_prefix}/{metric_id}"
                        if key != metric_id:
                            name = f"{name}/{key}"
                        metrics[name] = value

    aggregated = aggregate_all_metrics(per_run, active_metrics, config.aggregations)
    for metric_id, agg_values in aggregated.items():
        for key, value in agg_values.items():
            full_key = f"eval/{exploration_key}/{metric_id}/{key}"
            metrics[full_key] = value

    return output_envs or [env]


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
        viz_envs = evaluate_policy(
            n_collections,
            policy,
            eval_envs,
            config,
            exploration_type,
            metrics,
            eval_location=eval_location,
        )

        if (
            config.log_videos
            and (config.animation_interval > 0)
            and (n_collections % config.animation_interval == 0)
        ):
            visualize_envs(n_collections, viz_envs, config, exploration_type, video_log)

    return  {
                **metrics,
                **video_log,
                "batch/n_updates": n_updates,
                "batch/n_collections": n_collections,
                "batch/n_samples": n_samples,
            }


__all__ = [
    "EvaluationConfig",
    "make_eval_envs",
    "eval_pickled_env",
    "eval_env",
    "evaluate_policy",
    "visualize_envs",
    "run_evaluation",
]
