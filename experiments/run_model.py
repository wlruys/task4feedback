import fcntl
import os
import pickle
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchrl.envs import ExplorationType, set_exploration_type

from task4feedback.experiment_helper.env import RuntimeEnv, make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.model import (
    create_td_actor_critic_models,
    load_policy_from_checkpoint,
)
from task4feedback.experiment_helper.run_name import make_folder_name
from task4feedback.graphs.mesh.plot import _build_state, animate_mesh_graph
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import FeatureDimConfig

# =============================================================================
# Constants
# =============================================================================


MAX_ROLLOUT_STEPS = 1_000_000
EVAL_GRAPH_STEPS = 256
PHASE_LENGTH = 128
SYSTEM_MEMORY = 96e9
INFINITE_MEMORY = 9999e9

# =============================================================================
# File Utilities
# =============================================================================


def write_results_atomic(path: str, lines: Iterable[str]) -> None:
    """
    Append lines to a file using an exclusive file lock.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def csv_entry_exists(path: str, key: tuple[str, ...]) -> bool:
    """
    Check whether a CSV file already contains an entry starting with `key`.
    """
    if not os.path.exists(path):
        return False

    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= len(key) and tuple(parts[: len(key)]) == tuple(
                map(str, key)
            ):
                return True
    return False


# =============================================================================
# Model / Environment Setup
# =============================================================================


def load_normalization(folder_name: str, observer_version: str):
    norm_path = f"./norms/{folder_name}/{observer_version}_norm.pkl"
    with open(norm_path, "rb") as f:
        return pickle.load(f)


def prepare_eval_cfg(cfg: DictConfig) -> int:
    """
    Mutate cfg for evaluation and return number of evaluation runs.
    """
    cfg.system.mem = SYSTEM_MEMORY
    cfg.graph.config.steps = EVAL_GRAPH_STEPS
    cfg.graph.config.workload_args.traj_specifics.phase_length = PHASE_LENGTH

    return (
        20
        if cfg.graph.env.change_duration
        else 12
        if cfg.graph.env.change_workload
        else 1
    )


def build_env_and_model(cfg: DictConfig, norm, model_path: Path):
    graph_builder = make_graph_builder(cfg)
    env = make_env(
        graph_builder=graph_builder,
        cfg=cfg,
        normalization=norm,
        eval=True,
    )

    # cfg.system.mem = INFINITE_MEMORY

    # infenv = make_env(
    #     graph_builder=graph_builder,
    #     cfg=cfg,
    #     normalization=norm,
    #     eval=True,
    # )

    # cfg.system.mem = SYSTEM_MEMORY

    feature_config = FeatureDimConfig.from_observer(env.get_observer())
    model, _, _ = create_td_actor_critic_models(cfg, feature_config)

    if not load_policy_from_checkpoint(model, model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")

    return env, None, model


# =============================================================================
# Evaluation Logic
# =============================================================================


class ReplayMapper:
    def __init__(self, history):
        self.history = history

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        candidates = torch.zeros(
            (simulator.observer.graph_spec.max_candidates), dtype=torch.int64
        )
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            device = self.history[global_task_id]
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(
                global_task_id
            )
            mapping_result.append(
                fastsim.Action(i, device, mapping_priority, mapping_priority)
            )
        return mapping_result


# ------------------------------------------------------------
# Interval helpers
# ------------------------------------------------------------
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def interval_length(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in intervals)


def interval_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = 0
    overlap = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if s < e:
            overlap += e - s
        if e1 < e2:
            i += 1
        else:
            j += 1
    return overlap


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
def analyze_policy_run(env) -> dict[str, Any]:
    static_state, dynamic_state = _build_state(env)
    graph = env.get_graph()

    # ------------------------------------------------------------
    # Collect intervals
    # ------------------------------------------------------------
    compute_intervals_per_device = defaultdict(list)
    all_compute_intervals = []

    for i in range(static_state.n_compute_tasks):
        s = static_state.ct_launch_time[i]
        e = static_state.ct_complete_time[i]
        if s < 0 or e < 0:
            continue
        dev = int(static_state.ct_device[i]) - 1
        interval = (int(s), int(e))
        compute_intervals_per_device[dev].append(interval)
        all_compute_intervals.append(interval)

    comm_intervals = []
    comm_intervals_per_link = defaultdict(list)

    for i in range(static_state.n_data_tasks):
        if static_state.dt_virtual[i]:
            continue
        s = static_state.dt_launch_time[i]
        e = static_state.dt_complete_time[i]
        if s < 0 or e < 0:
            continue
        src = int(static_state.dt_source[i]) - 1
        dst = int(static_state.dt_device[i]) - 1
        interval = (int(s), int(e))
        comm_intervals.append(interval)
        comm_intervals_per_link[(src, dst)].append(interval)

    # ------------------------------------------------------------
    # Merge intervals
    # ------------------------------------------------------------
    all_compute_merged = merge_intervals(all_compute_intervals)
    all_comm_merged = merge_intervals(comm_intervals)

    per_device_compute = {
        d: merge_intervals(v) for d, v in compute_intervals_per_device.items()
    }

    # ------------------------------------------------------------
    # Makespan
    # ------------------------------------------------------------
    t0 = min(
        [s for s, _ in all_compute_intervals + comm_intervals],
        default=0,
    )
    t1 = max(
        [e for _, e in all_compute_intervals + comm_intervals],
        default=0,
    )
    makespan = t1 - t0

    # ------------------------------------------------------------
    # Utilization metrics
    # ------------------------------------------------------------
    total_compute_time = interval_length(all_compute_merged)
    total_comm_time = interval_length(all_comm_merged)
    overlap_time = interval_overlap(all_compute_merged, all_comm_merged)

    exposed_comm_time = total_comm_time - overlap_time
    gpu_idle_time = makespan - total_compute_time
    gpu_idle_frac = gpu_idle_time / makespan if makespan > 0 else 0.0

    # ------------------------------------------------------------
    # Critical path (approximate but effective)
    # ------------------------------------------------------------
    # Assumption: anything exposed (not overlapped) is on the critical path
    critical_path_length = total_compute_time + exposed_comm_time

    # ------------------------------------------------------------
    # Per-device utilization
    # ------------------------------------------------------------
    per_device_stats = {}
    for dev, intervals in per_device_compute.items():
        busy = interval_length(intervals)
        per_device_stats[dev] = {
            "busy_time": busy,
            "utilization": busy / makespan if makespan > 0 else 0.0,
        }

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    return {
        "makespan": makespan,
        "total_compute_time": total_compute_time,
        "total_comm_time": total_comm_time,
        "overlap_time": overlap_time,
        "overlap_ratio": overlap_time / total_comm_time if total_comm_time > 0 else 0.0,
        "exposed_comm_time": exposed_comm_time,
        "gpu_idle_time": gpu_idle_time,
        "gpu_idle_fraction": gpu_idle_frac,
        "critical_path_length": critical_path_length,
        "per_device": per_device_stats,
    }


def evaluate_model(
    env: RuntimeEnv, infenv: RuntimeEnv, model, num_runs: int
) -> tuple[float, float]:
    """
    Run evaluation rollouts and return (avg_time, avg_evictions).
    """
    model.eval()
    results: list[tuple[float, float]] = []
    # eft_results: List[Tuple[float, float]] = []
    # inf_results: List[Tuple[float, float]] = []

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for run_idx in range(num_runs):
            td = env.reset()
            # infenv.reset()

            # copy_sim = env.simulator.copy()
            # copy_sim.disable_external_mapper()
            # copy_sim.run()
            # print(copy_sim.time, flush=True)
            # eft_results.append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))

            env.rollout(
                max_steps=MAX_ROLLOUT_STEPS,
                policy=model.actor,
                auto_reset=False,
                tensordict=td,
            )

            # infsim = infenv.simulator
            # runtime = env.simulator.state.get_task_runtime()
            # history = {}
            # for task_id in range(8 * 8 * EVAL_GRAPH_STEPS):
            # history[task_id] = runtime.get_compute_task_mapped_device(task_id)
            # infsim.external_mapper = ReplayMapper(history)
            # infsim.enable_external_mapper()
            # infsim.run()
            # inf_results.append((infsim.time, sum(list(infsim.total_eviction_movement())[1:]), sum(infsim.total_data_movement())))

            sim = env.simulator
            eviction_cost = sum(list(sim.total_eviction_movement())[1:])
            data_movement = sum(sim.total_data_movement())
            results.append((sim.time, eviction_cost, data_movement))

    avg_time = sum(r[0] for r in results) / len(results)
    avg_eviction = sum(r[1] for r in results) / len(results)
    avg_data_movement = sum(r[2] for r in results) / len(results)

    # eft_avg_time = sum(r[0] for r in eft_results) / len(eft_results)
    # eft_avg_eviction = sum(r[1] for r in eft_results) / len(eft_results)
    # eft_avg_data_movement = sum(r[2] for r in eft_results) / len(eft_results)

    # inf_avg_time = sum(r[0] for r in inf_results) / len(inf_results)
    # inf_avg_eviction = sum(r[1] for r in inf_results) / len(inf_results)
    # inf_avg_data_movement = sum(r[2] for r in inf_results) / len(inf_results)

    return {
        "rl": {
            "time": avg_time,
            "eviction": avg_eviction,
            "data_movement": avg_data_movement,
        },
        # "eft": {"time": eft_avg_time, "eviction": eft_avg_eviction, "data_movement": eft_avg_data_movement},
        # "inf": {"time": inf_avg_time, "eviction": inf_avg_eviction, "data_movement": inf_avg_data_movement},
    }


# =============================================================================
# Main Evaluation Driver
# =============================================================================


def configure_training(cfg: DictConfig) -> None:
    folder_name, _, _, _ = make_folder_name(cfg, change_name=False)
    model_dir = Path(f"./models/{cfg.system.n_devices - 1}gpus") / folder_name
    default_mem = cfg.graph.config.level_memory
    while not model_dir.exists():
        cfg.graph.config.level_memory += 1e9
        folder_name, _, _, _ = make_folder_name(cfg, change_name=False)
        model_dir = Path(f"./models/{cfg.system.n_devices - 1}gpus") / folder_name
    cfg.graph.config.level_memory = default_mem
    print(f"Using models from {model_dir}", flush=True)

    graph_name = cfg.graph.config.workload_args.traj_type
    if cfg.graph.env.change_duration:
        RESULTS_CSV = f"./results/{cfg.system.n_devices - 1}gpus/noise_results_rl_{EVAL_GRAPH_STEPS}.csv"
    else:
        RESULTS_CSV = f"./results/{cfg.system.n_devices - 1}gpus/results_rl_{EVAL_GRAPH_STEPS}.csv"

    norm = load_normalization(
        folder_name,
        cfg.feature.observer.version,
    )

    num_runs = prepare_eval_cfg(cfg)
    output_lines: list[str] = []
    output_lines_with_eft: list[str] = []

    for model_path in model_dir.glob("*.pt"):
        key = (
            graph_name,
            cfg.graph.config.level_memory,
            cfg.graph.config.r_interior,
            cfg.graph.config.r_boundary,
            model_path.stem,
        )

        if csv_entry_exists(RESULTS_CSV, key):
            print(f"[SKIP] CSV entry already exists for {model_path.name}", flush=True)
            # vid_path = model_path.with_suffix(".mp4")
            # if vid_path.exists():
            #     print(f"[SKIP] Video already exists for {model_path.name}", flush=True)
            # else:
            #     env, infenv, model = build_env_and_model(cfg, norm, model_path)
            #     env.rollout(policy=model.actor, max_steps=MAX_ROLLOUT_STEPS)
            #     result = analyze_policy_run(env)
            #     animate_mesh_graph(env=env, folder=model_path.parent, filename=vid_path.name)
            #     print(f"[SAVE] Animation saved to {vid_path}", flush=True)
            #     # save analysis result as json
            #     result_path = model_path.with_suffix(".json")
            #     with open(result_path, "w") as f:
            #         import json

            #         json.dump(result, f, indent=4)
            #         print(f"[SAVE] Analysis result saved to {result_path}", flush=True)
            continue

        try:
            env, infenv, model = build_env_and_model(cfg, norm, model_path)
            results = evaluate_model(env, infenv, model, num_runs)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            continue

        line = (
            f"{graph_name},"
            f"{cfg.graph.config.level_memory},"
            f"{cfg.graph.config.r_interior},"
            f"{cfg.graph.config.r_boundary},"
            f"{model_path.stem},"
            f"{results['rl']['time']:.0f},"
            f"{results['rl']['eviction']:.0f},"
            f"{results['rl']['data_movement']:.0f}"
        )

        # line_inf = (
        #     f"{graph_name},"
        #     f"{cfg.graph.config.level_memory},"
        #     f"{cfg.graph.config.r_interior},"
        #     f"{cfg.graph.config.r_boundary},"
        #     f"{model_path.stem}_inf,"
        #     f"{results['inf']['time']:.0f},"
        #     f"{results['inf']['eviction']:.0f},"
        #     f"{results['inf']['data_movement']:.0f}"
        # )

        # line_eft = (
        #     f"{graph_name},"
        #     f"{cfg.graph.config.level_memory},"
        #     f"{cfg.graph.config.r_interior},"
        #     f"{cfg.graph.config.r_boundary},"
        #     f"{model_path.stem},"
        #     f"{results['rl']['time']:.0f},"
        #     f"{results['rl']['eviction']:.0f},"
        #     f"{results['eft']['time']:.0f},"
        # )
        write_results_atomic(RESULTS_CSV, [line])
        # output_lines.append(line_inf)
        # output_lines_with_eft.append(line_eft)
        # write_results_atomic(RESULTS_SANITY, output_lines_with_eft)


# =============================================================================
# Entrypoint
# =============================================================================


@hydra.main(
    config_path="conf",
    config_name="dynamic_batch.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.use_deterministic_algorithms(cfg.deterministic_torch)
    configure_training(cfg)


if __name__ == "__main__":
    main()
