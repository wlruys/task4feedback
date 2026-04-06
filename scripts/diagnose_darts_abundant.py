"""
diagnose_darts_abundant.py
Compare DARTS vs EFT under the EXACT parameters used by plot_bench_eviction_jacobi.py
to understand why DARTS underperforms in abundant-memory / transfer-dominated regimes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import random
import numpy as np
import torch

from bench_eviction_jacobi import build_graph, build_system, N_DEVICES
from bench_mapper_support import (
    DARTSConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    make_darts_mapper,
    make_internal_mapper,
    make_transition_conditions,
)
import task4feedback.fastsim2 as fastsim
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput

# ── parameters matching plot_bench_eviction_jacobi.py defaults ──────────────
PLOT_SCRIPT_KWARGS = dict(
    domain_ratio=1.0,
    arithmetic_intensity=595.5555555,
    arithmetic_complexity=1.0,
    boundary_complexity=1.0,
    memory_intensity=0.0,
    boundary_width=0.25,
    r_interior=10.0,
    r_boundary=0.1,
    workload_scale=0.3,
    workload_lower_bound=1.0,
    workload_upper_bound=5.0,
    workload_phase_length=128,
    randomize_initial_placement=True,
)

N_GPU = N_DEVICES - 1  # 4 GPUs, 1 CPU


def _bytes_hr(v: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1024.0:
            return f"{v:.1f} {u}"
        v /= 1024.0
    return f"{v:.1f} PB"


def run_one(name: str, mapper, transition_name: str, tc: TransitionConfig,
            *, level_memory: int, gpu_mem: int, grid_n: int, steps: int, seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sys_obj = build_system(mem=gpu_mem)
    graph = build_graph(sys_obj, grid_n=grid_n, steps=steps, level_memory=level_memory,
                        **PLOT_SCRIPT_KWARGS)
    blocks = graph.get_blocks()
    task_noise = fastsim.TaskNoise(graph.static_graph)
    tc_obj = make_transition_conditions(transition_name, top_k_candidates=256, config=tc)
    sim_input = SimulatorInput(
        graph, blocks, build_system(mem=gpu_mem), task_noise=task_noise,
        transition_conditions=tc_obj, top_k_candidates=256,
        eviction_policy=fastsim.EvictionPolicy.LRU,
    )
    driver = SimulatorDriver(sim_input, internal_mapper=mapper)
    driver.initialize(); driver.initialize_data(); driver.disable_external_mapper()

    t0 = time.perf_counter()
    status = driver.run()
    wall = time.perf_counter() - t0

    if status != fastsim.ExecutionState.COMPLETE:
        print(f"  {name:<55} FAILED: {status}")
        return None

    sim_s = driver.time / 1e6
    total_mv = sum(driver.total_data_movement())
    evict_mv = sum(driver.total_eviction_movement())
    rstate = driver.state.get_task_runtime()
    n_compute = rstate.get_n_compute_tasks()

    tasks_per_dev = [0] * N_DEVICES
    compute_time_per_dev = [0] * N_DEVICES
    idle_time_per_dev = [0] * N_DEVICES
    xfer_time_per_dev = [0] * N_DEVICES
    first_launch = [driver.time] * N_DEVICES
    last_complete = [0] * N_DEVICES

    for i in range(n_compute):
        dev = rstate.get_compute_task_mapped_device(i)
        if dev < 0 or dev >= N_DEVICES:
            continue
        tasks_per_dev[dev] += 1
        dur = rstate.get_compute_task_duration(i)
        compute_time_per_dev[dev] += dur
        launched = rstate.get_compute_task_launched_time(i)
        completed = rstate.get_compute_task_completed_time(i)
        if launched < first_launch[dev]:
            first_launch[dev] = launched
        if completed > last_complete[dev]:
            last_complete[dev] = completed

    n_data = rstate.get_n_data_tasks()
    for i in range(n_data):
        if rstate.is_data_task_virtual(i):
            continue
        dev = rstate.get_data_task_mapped_device(i)
        if dev < 0 or dev >= N_DEVICES:
            continue
        xfer_time_per_dev[dev] += rstate.get_data_task_duration(i)

    for dev in range(1, N_DEVICES):
        span = last_complete[dev] - first_launch[dev]
        if span > 0:
            idle_time_per_dev[dev] = max(0, span - compute_time_per_dev[dev])

    gpu_tasks = tasks_per_dev[1:]
    gpu_idle = [t / 1e6 for t in idle_time_per_dev[1:]]
    gpu_xfer = [t / 1e6 for t in xfer_time_per_dev[1:]]
    gpu_compute = [t / 1e6 for t in compute_time_per_dev[1:]]

    # Load balance coefficient of variation
    mean_t = sum(gpu_tasks) / len(gpu_tasks) if gpu_tasks else 1
    lb_cv = (sum((t - mean_t) ** 2 for t in gpu_tasks) / len(gpu_tasks)) ** 0.5 / mean_t if mean_t else 0

    print(f"  {name:<55} sim={sim_s:7.3f}s  mv={_bytes_hr(total_mv):>10}  ev={_bytes_hr(evict_mv):>8}  lb_cv={lb_cv:.3f}")
    print(f"    tasks/gpu:   {gpu_tasks}  total={sum(gpu_tasks)}")
    print(f"    compute(s):  {[f'{t:.3f}' for t in gpu_compute]}  sum={sum(gpu_compute):.3f}")
    print(f"    idle(s):     {[f'{t:.3f}' for t in gpu_idle]}  sum={sum(gpu_idle):.3f}")
    print(f"    data_xfr(s): {[f'{t:.3f}' for t in gpu_xfer]}  sum={sum(gpu_xfer):.3f}")
    return sim_s


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--level-memory-gb", type=float, default=200.0)
    parser.add_argument("--gpu-mem-gb", type=float, default=96.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gpu_mem = int(args.gpu_mem_gb * 1e9)
    level_memory = int(args.level_memory_gb * 1e9)
    grid_n = args.grid_n
    steps = args.steps

    compute_time_approx = level_memory / (grid_n * grid_n) / 129e9 * 1e6 / 10  # µs
    xfer_time_approx = level_memory / (grid_n * grid_n) / 129e9 * 1e6  # µs

    print(f"\n{'='*80}")
    print(f"  Jacobi {grid_n}x{grid_n} {steps} steps")
    print(f"  level_memory={args.level_memory_gb:.0f} GB  gpu_mem={args.gpu_mem_gb:.0f} GB")
    print(f"  per-cell data size = {level_memory/(grid_n*grid_n)/1e9:.3f} GB")
    print(f"  Approx compute time per task = {compute_time_approx:.0f} µs")
    print(f"  Approx transfer time per cell = {xfer_time_approx:.0f} µs")
    print(f"  Transfer:Compute ratio = 10:1 (r_interior=10.0)")
    print(f"{'='*80}")

    # ── Transition conditions ─────────────────────────────────────────────────
    # EFT uses hysteresis: maps tasks in bulk (open at 16, close at 36 in-flight).
    tc_hysteresis = TransitionConfig(
        kind="hysteresis", hysteresis_open=16, hysteresis_close=36, hysteresis_starvation=2)

    # PlannedThreshold is the new DARTS pairing.
    tc_darts_planned_1 = TransitionConfig(kind="planned", planned_threshold=1)
    tc_darts_planned_2 = TransitionConfig(kind="planned", planned_threshold=2)
    tc_darts_planned_4 = TransitionConfig(kind="planned", planned_threshold=4)

    configs = [
        # ── Baselines ────────────────────────────────────────────────────────
        ("EFT+hysteresis [baseline]",
         fastsim.DequeueEFTMapper(), "dequeue_eft", tc_hysteresis),

        # EFT with planned transition for apples-to-apples trigger behavior.
        ("EFT+planned(th=1)",
         fastsim.DequeueEFTMapper(), "dequeue_eft", tc_darts_planned_1),

        ("MemAwareEFT+hysteresis",
         fastsim.MemoryAwareEFTMapper(), "memory_aware_eft", tc_hysteresis),

        # ── New DARTS mapper: short/medium horizon thresholds ─────────────────
        ("DARTS(sh=2,mh=4) [DEFAULT]",
         make_darts_mapper(DARTSConfig(short_horizon_threshold=2, medium_horizon_threshold=4)),
         "darts", tc_darts_planned_1),
        ("DARTS(sh=1,mh=2)",
         make_darts_mapper(DARTSConfig(short_horizon_threshold=1, medium_horizon_threshold=2)),
         "darts", tc_darts_planned_2),
        ("DARTS(sh=2,mh=6)",
         make_darts_mapper(DARTSConfig(short_horizon_threshold=2, medium_horizon_threshold=6)),
         "darts", tc_darts_planned_2),
        ("DARTS(sh=4,mh=8)",
         make_darts_mapper(DARTSConfig(short_horizon_threshold=4, medium_horizon_threshold=8)),
         "darts", tc_darts_planned_4),
    ]

    results = {}
    for name, mapper, transition_name, tc in configs:
        print(f"\n  {name}")
        sim_s = run_one(name, mapper, transition_name, tc,
                        level_memory=level_memory, gpu_mem=gpu_mem,
                        grid_n=grid_n, steps=steps, seed=args.seed)
        if sim_s is not None:
            results[name] = sim_s

    if results:
        best = min(results.values())
        eft_name = "EFT+hysteresis"
        eft_s = results.get(eft_name, 0)
        print(f"\n{'='*80}")
        print("  Summary (sorted by sim time):")
        for name, sim_s in sorted(results.items(), key=lambda kv: kv[1]):
            ratio = sim_s / eft_s if eft_s else 1
            print(f"    {name:<55} {sim_s:7.3f}s  ({ratio:.2f}× EFT)")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
