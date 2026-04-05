"""
bench_darts_diagnostic.py
Diagnostic benchmark: compare EFT, MemoryAwareEFT, and DARTS mappers
across Jacobi and Cholesky workloads.

Reports per-device load balance, idle time, data movement, and transfer counts
to help identify why DARTS underperforms EFT in memory-abundant regimes.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from bench_mapper_support import (
    DARTSConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    make_darts_mapper,
    make_internal_mapper,
    make_transition_conditions,
)
from bench_eviction_jacobi import (
    N_DEVICES as JACOBI_N_DEVICES,
    build_graph as build_jacobi_graph,
    build_system as build_jacobi_system,
    set_seed,
    _bytes_hr,
)
from bench_eviction_cholesky import (
    N_DEVICES as CHOL_N_DEVICES,
    build_graph as build_cholesky_graph,
    build_system as build_cholesky_system,
    derive_block_bytes_for_pressure,
    lower_triangular_block_count,
)
from task4feedback.fastsim2 import EvictionPolicy, ExecutionState, TaskNoise
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput


N_GPU_DEVICES = JACOBI_N_DEVICES - 1

JACOBI_DEFAULTS = dict(
    domain_ratio=1.0,
    arithmetic_intensity=1.0,
    arithmetic_complexity=2.0,
    boundary_complexity=1.5,
    memory_intensity=1.0,
    boundary_width=0.1,
    r_interior=1.0,
    r_boundary=1.0,
    workload_scale=1.0,
    workload_lower_bound=0.5,
    workload_upper_bound=1.5,
    workload_phase_length=64,
)


@dataclass
class DetailedResult:
    label: str
    workload: str
    regime: str
    status: str
    sim_s: float = 0.0
    wall_s: float = 0.0
    total_mv: int = 0
    evict_mv: int = 0
    n_data_moves: int = 0
    n_eviction_moves: int = 0
    per_device_completed: list[int] = field(default_factory=list)
    per_device_compute_time: list[float] = field(default_factory=list)
    per_device_data_time: list[float] = field(default_factory=list)
    per_device_idle_time: list[float] = field(default_factory=list)
    error: Optional[str] = None


def extract_per_device_stats(driver, n_devices: int) -> dict:
    """Extract per-device statistics from a completed simulation."""
    task_runtime = driver.state.get_task_runtime()
    n_compute = task_runtime.get_n_compute_tasks()
    n_data = task_runtime.get_n_data_tasks()
    n_eviction = task_runtime.get_n_eviction_tasks()

    sim_end_time = driver.time  # total simulation time in microseconds

    # Per-device compute task counts and total compute time
    device_task_count = [0] * n_devices
    device_compute_time = [0] * n_devices  # sum of task durations
    device_first_launch = [sim_end_time] * n_devices
    device_last_complete = [0] * n_devices

    for i in range(n_compute):
        dev = task_runtime.get_compute_task_mapped_device(i)
        if dev < 0 or dev >= n_devices:
            continue
        device_task_count[dev] += 1
        duration = task_runtime.get_compute_task_duration(i)
        device_compute_time[dev] += duration
        launched = task_runtime.get_compute_task_launched_time(i)
        completed = task_runtime.get_compute_task_completed_time(i)
        if launched < device_first_launch[dev]:
            device_first_launch[dev] = launched
        if completed > device_last_complete[dev]:
            device_last_complete[dev] = completed

    # Per-device data movement time
    device_data_time = [0] * n_devices
    for i in range(n_data):
        if task_runtime.is_data_task_virtual(i):
            continue
        dev = task_runtime.get_data_task_mapped_device(i)
        if dev < 0 or dev >= n_devices:
            continue
        duration = task_runtime.get_data_task_duration(i)
        device_data_time[dev] += duration

    # Idle time: span from first launch to last complete minus compute time
    device_idle_time = [0] * n_devices
    for dev in range(n_devices):
        span = device_last_complete[dev] - device_first_launch[dev]
        if span > 0:
            device_idle_time[dev] = max(0, span - device_compute_time[dev])

    # Data movement counts
    n_data_moves = sum(1 for i in range(n_data) if not task_runtime.is_data_task_virtual(i))
    n_eviction_moves = sum(
        1 for i in range(n_eviction) if not task_runtime.is_eviction_task_virtual(i)
    )

    return {
        "per_device_completed": device_task_count,
        "per_device_compute_time": device_compute_time,
        "per_device_data_time": device_data_time,
        "per_device_idle_time": device_idle_time,
        "n_data_moves": n_data_moves,
        "n_eviction_moves": n_eviction_moves,
    }


def run_mapper(
    mapper_name: str,
    *,
    workload: str,
    regime: str,
    gpu_mem: int,
    seed: int,
    top_k: int,
    # Jacobi params
    grid_n: int = 16,
    steps: int = 256,
    level_memory: int = 0,
    # Cholesky params
    n_blocks: int = 16,
    block_bytes: int = 0,
    # Mapper config
    darts_config: Optional[DARTSConfig] = None,
    memory_aware_eft_config: Optional[MemoryAwareEFTConfig] = None,
    transition_kind: str = "auto",
    max_mapped: Optional[int] = None,
    transfer_aware: bool = False,
) -> DetailedResult:
    set_seed(seed)

    label = mapper_name
    if darts_config is not None:
        mt = darts_config.mapped_threshold
        rt = darts_config.reserved_threshold
        ext = "ext" if darts_config.extended_frontier else ""
        label = f"DARTS(mt={mt},rt={rt}{','+ext if ext else ''})"

    n_devices = JACOBI_N_DEVICES

    if workload == "jacobi":
        graph_system = build_jacobi_system(mem=gpu_mem)
        graph = build_jacobi_graph(
            graph_system,
            grid_n=grid_n,
            steps=steps,
            level_memory=level_memory,
            randomize_initial_placement=True,
            **JACOBI_DEFAULTS,
        )
        sim_system = build_jacobi_system(mem=gpu_mem)
    elif workload == "cholesky":
        n_devices = CHOL_N_DEVICES
        graph_system = build_cholesky_system(mem=gpu_mem)
        graph = build_cholesky_graph(
            graph_system,
            n_blocks=n_blocks,
            block_bytes=block_bytes,
            randomize_initial_placement=True,
            verbose=False,
        )
        sim_system = build_cholesky_system(mem=gpu_mem)
    else:
        raise ValueError(f"Unknown workload: {workload}")

    # Resolve transition config
    if mapper_name.startswith("darts"):
        dc = darts_config or DARTSConfig()
        tc = TransitionConfig(
            kind=transition_kind if transition_kind != "auto" else "device_threshold",
            mapped_threshold=dc.mapped_threshold,
            reserved_threshold=dc.reserved_threshold,
            max_in_flight=max_mapped,
        )
    else:
        tc = TransitionConfig(kind=transition_kind if transition_kind != "auto" else "auto")

    blocks = graph.get_blocks()
    task_noise = TaskNoise(graph.static_graph)
    sim_input = SimulatorInput(
        graph,
        blocks,
        sim_system,
        task_noise=task_noise,
        transition_conditions=make_transition_conditions(
            mapper_name.split("(")[0] if "(" not in mapper_name else "darts",
            top_k_candidates=top_k,
            config=tc,
        ),
        top_k_candidates=top_k,
        eviction_policy=EvictionPolicy.LRU,
        transfer_aware_data_launch_order=transfer_aware,
    )

    internal_mapper = make_internal_mapper(
        mapper_name.split("(")[0] if "(" not in mapper_name else "darts",
        darts_config=darts_config,
        memory_aware_eft_config=memory_aware_eft_config,
    )
    driver = SimulatorDriver(sim_input, internal_mapper=internal_mapper)
    driver.initialize()
    driver.initialize_data()
    driver.disable_external_mapper()

    t0 = time.perf_counter()
    status = driver.run()
    wall_s = time.perf_counter() - t0

    if status != ExecutionState.COMPLETE:
        return DetailedResult(label, workload, regime, "FAILED", error=str(status))

    total_mv = sum(driver.total_data_movement())
    evict_mv = sum(driver.total_eviction_movement())
    stats = extract_per_device_stats(driver, n_devices)

    return DetailedResult(
        label=label,
        workload=workload,
        regime=regime,
        status="OK",
        sim_s=driver.time / 1e6,
        wall_s=wall_s,
        total_mv=total_mv,
        evict_mv=evict_mv,
        n_data_moves=stats["n_data_moves"],
        n_eviction_moves=stats["n_eviction_moves"],
        per_device_completed=stats["per_device_completed"],
        per_device_compute_time=stats["per_device_compute_time"],
        per_device_data_time=stats["per_device_data_time"],
        per_device_idle_time=stats["per_device_idle_time"],
    )


def load_balance_metric(counts: list[int]) -> float:
    """Coefficient of variation of per-GPU task counts (lower = better balanced).
    Excludes device 0 (CPU)."""
    gpu_counts = [c for c in counts[1:] if c > 0]
    if not gpu_counts:
        return 0.0
    mean = sum(gpu_counts) / len(gpu_counts)
    if mean == 0:
        return 0.0
    variance = sum((c - mean) ** 2 for c in gpu_counts) / len(gpu_counts)
    return (variance ** 0.5) / mean


def print_detailed(results: list[DetailedResult], title: str) -> None:
    print()
    print(f"### {title}")
    print()

    # Summary table
    hdr = (
        f"{'Mapper':<40}  {'sim(s)':>8}  {'wall(s)':>8}  "
        f"{'total_mv':>10}  {'evict_mv':>10}  {'#moves':>6}  {'#evict':>6}  {'LB(CV)':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.status != "OK":
            print(f"{r.label:<40}  {'FAILED':>8}  {r.error}")
            continue
        lb = load_balance_metric(r.per_device_completed)
        print(
            f"{r.label:<40}  {r.sim_s:8.3f}  {r.wall_s:8.3f}  "
            f"{_bytes_hr(r.total_mv):>10}  {_bytes_hr(r.evict_mv):>10}  "
            f"{r.n_data_moves:>6}  {r.n_eviction_moves:>6}  {lb:7.3f}"
        )

    # Per-device detail
    print()
    print("  Per-device detail (GPU only, device 1..N):")
    for r in results:
        if r.status != "OK":
            continue
        gpu_completed = r.per_device_completed[1:]
        gpu_compute = [t / 1e6 for t in r.per_device_compute_time[1:]]
        gpu_idle = [t / 1e6 for t in r.per_device_idle_time[1:]]
        gpu_data = [t / 1e6 for t in r.per_device_data_time[1:]]
        total_tasks = sum(gpu_completed)
        total_compute = sum(gpu_compute)
        total_idle = sum(gpu_idle)
        print(f"  {r.label}:")
        print(f"    tasks/GPU:     {gpu_completed}  (total={total_tasks})")
        print(f"    compute(s):    {[f'{t:.3f}' for t in gpu_compute]}  (sum={total_compute:.3f})")
        print(f"    idle(s):       {[f'{t:.3f}' for t in gpu_idle]}  (sum={total_idle:.3f})")
        print(f"    data_xfer(s):  {[f'{t:.3f}' for t in gpu_data]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS vs EFT diagnostic")
    parser.add_argument("--workload", choices=["jacobi", "cholesky", "both"], default="both")
    parser.add_argument("--regime", choices=["abundant", "tight", "both"], default="both")
    parser.add_argument("--gpu-mem-gb", type=float, default=4.0)
    parser.add_argument("--grid-n", type=int, default=16, help="Jacobi grid size")
    parser.add_argument("--steps", type=int, default=256, help="Jacobi steps")
    parser.add_argument("--n-blocks", type=int, default=16, help="Cholesky block dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()

    gpu_mem = int(args.gpu_mem_gb * 1e9)
    n_gpu = N_GPU_DEVICES

    # Define mapper configurations to compare
    mapper_configs = [
        ("dequeue_eft", None, None, "auto", False),
        ("memory_aware_eft", None, MemoryAwareEFTConfig(), "auto", False),
        ("darts", DARTSConfig(mapped_threshold=0, reserved_threshold=-1), None, "device_threshold", False),
        ("darts", DARTSConfig(mapped_threshold=4, reserved_threshold=-1), None, "device_threshold", False),
        ("darts", DARTSConfig(mapped_threshold=8, reserved_threshold=-1), None, "device_threshold", False),
        ("darts", DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                              extended_frontier=True, extended_batch=True, extended_batch_cap=2),
         None, "device_threshold", False),
        ("darts", DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                              extended_frontier=True, extended_batch=True, extended_batch_cap=4),
         None, "device_threshold", False),
        # DARTS with transfer_aware data launch ordering
        ("darts", DARTSConfig(mapped_threshold=0, reserved_threshold=-1), None, "device_threshold", True),
        ("darts", DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                              extended_frontier=True, extended_batch=True, extended_batch_cap=2),
         None, "device_threshold", True),
    ]

    workloads = []
    if args.workload in {"jacobi", "both"}:
        workloads.append("jacobi")
    if args.workload in {"cholesky", "both"}:
        workloads.append("cholesky")

    regimes_to_run = []
    if args.regime in {"abundant", "both"}:
        regimes_to_run.append("abundant")
    if args.regime in {"tight", "both"}:
        regimes_to_run.append("tight")

    for wl in workloads:
        for regime in regimes_to_run:
            if wl == "jacobi":
                aggregate = n_gpu * gpu_mem
                if regime == "abundant":
                    level_memory = int(aggregate * 0.6)
                else:
                    level_memory = int(aggregate * 1.4)
                wl_kwargs = dict(
                    grid_n=args.grid_n, steps=args.steps, level_memory=level_memory,
                )
                title = (f"Jacobi {args.grid_n}x{args.grid_n} {args.steps}step "
                         f"{regime} (lm={_bytes_hr(level_memory)})")
            else:
                if regime == "abundant":
                    block_bytes = derive_block_bytes_for_pressure(
                        n_blocks=args.n_blocks, gpu_mem=gpu_mem,
                        n_gpu_devices=n_gpu, target_overcommit_ratio=0.6,
                    )
                else:
                    block_bytes = derive_block_bytes_for_pressure(
                        n_blocks=args.n_blocks, gpu_mem=gpu_mem,
                        n_gpu_devices=n_gpu, target_overcommit_ratio=1.4,
                    )
                wl_kwargs = dict(n_blocks=args.n_blocks, block_bytes=block_bytes)
                title = (f"Cholesky {args.n_blocks}x{args.n_blocks} "
                         f"{regime} (bb={_bytes_hr(block_bytes)})")

            print(f"\n{'='*80}")
            print(f"  {title}")
            print(f"{'='*80}")

            results = []
            for mapper_name, darts_cfg, maeft_cfg, tk, ta in mapper_configs:
                try:
                    r = run_mapper(
                        mapper_name,
                        workload=wl,
                        regime=regime,
                        gpu_mem=gpu_mem,
                        seed=args.seed,
                        top_k=args.top_k,
                        darts_config=darts_cfg,
                        memory_aware_eft_config=maeft_cfg,
                        transition_kind=tk,
                        transfer_aware=ta,
                        **wl_kwargs,
                    )
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    r = DetailedResult(mapper_name, wl, regime, "ERROR", error=str(exc))
                results.append(r)
                if r.status == "OK":
                    print(f"  {r.label:<40}  sim={r.sim_s:.3f}s  mv={_bytes_hr(r.total_mv)}")
                else:
                    print(f"  {r.label:<40}  FAILED: {r.error}")

            print_detailed(results, title)


if __name__ == "__main__":
    main()
