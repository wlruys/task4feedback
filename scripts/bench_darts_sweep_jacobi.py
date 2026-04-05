"""
bench_darts_sweep_jacobi.py
Sweep DARTS configuration axes on the dynamic Jacobi stencil workload.

Mirrors the structure of bench_darts_sweep_cholesky.py but for Jacobi,
allowing direct comparison of DARTS tuning across two qualitatively different
workloads.

Problem structure:
  - grid_n×grid_n dynamic Jacobi (circle trajectory, 256 steps)
  - Two memory regimes:
      abundant: level_memory sized so total data fits in aggregate GPU mem
      tight:    level_memory sized so eviction is forced
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

from bench_mapper_support import (
    DARTSConfig,
    TransitionConfig,
    make_darts_mapper,
    make_transition_conditions,
)
from bench_eviction_jacobi import (
    N_DEVICES,
    build_graph,
    build_system,
    set_seed,
    _bytes_hr,
)
from task4feedback.fastsim2 import EvictionPolicy, ExecutionState
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput
from task4feedback.fastsim2 import TaskNoise

# ---------------------------------------------------------------------------
# Default benchmark parameters
# ---------------------------------------------------------------------------
GRID_N = 16
STEPS = 256
GPU_MEM_GB = 4.0
SEED = 0
TOP_K = 256

N_GPU_DEVICES = N_DEVICES - 1

# Jacobi graph parameters (matching bench_eviction_jacobi defaults)
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


def derive_level_memory_for_pressure(
    *,
    grid_n: int,
    gpu_mem: int,
    n_gpu_devices: int,
    target_overcommit_ratio: float,
) -> int:
    """Compute level_memory so that total_data / aggregate_gpu_mem ≈ target_ratio.

    For Jacobi, total_data ≈ level_memory * (steps + 1), but the key metric
    is level_memory vs aggregate_gpu_mem since the scheduler sees at most a few
    levels at a time.  We size level_memory as a fraction of aggregate GPU mem.
    """
    aggregate_gpu_mem = n_gpu_devices * gpu_mem
    # level_memory * ratio ≈ aggregate_gpu_mem → level_memory = agg / ratio
    # But we want: level_memory / aggregate_gpu_mem = target_ratio
    level_memory = int(aggregate_gpu_mem * target_overcommit_ratio)
    return level_memory


@dataclass
class SweepResult:
    label: str
    regime: str
    status: str
    sim_s: Optional[float] = None
    total_mv: Optional[int] = None
    evict_mv: Optional[int] = None
    n_data_moves: Optional[int] = None
    error: Optional[str] = None


def run_darts(
    cfg: DARTSConfig,
    *,
    gpu_mem: int,
    level_memory: int,
    grid_n: int,
    steps: int,
    seed: int,
    top_k: int,
    transition_kind: str = "device_threshold",
    max_mapped: Optional[int] = None,
    transfer_aware: bool = False,
    pipeline_depth: Optional[int] = None,
) -> dict:
    set_seed(seed)
    mapper = make_darts_mapper(cfg)
    tc_kwargs = dict(
        kind=transition_kind,
        mapped_threshold=cfg.mapped_threshold,
        reserved_threshold=cfg.reserved_threshold,
        max_in_flight=max_mapped,
    )
    if pipeline_depth is not None:
        tc_kwargs["pipeline_depth"] = pipeline_depth
    transition_config = TransitionConfig(**tc_kwargs)
    graph_system = build_system(mem=gpu_mem)
    graph = build_graph(
        graph_system,
        grid_n=grid_n,
        steps=steps,
        level_memory=level_memory,
        randomize_initial_placement=True,
        **JACOBI_DEFAULTS,
    )
    sim_system = build_system(mem=gpu_mem)
    blocks = graph.get_blocks()
    task_noise = TaskNoise(graph.static_graph)
    sim_input = SimulatorInput(
        graph,
        blocks,
        sim_system,
        task_noise=task_noise,
        transition_conditions=make_transition_conditions(
            "darts",
            top_k_candidates=top_k,
            config=transition_config,
        ),
        top_k_candidates=top_k,
        eviction_policy=EvictionPolicy.LRU,
        transfer_aware_data_launch_order=transfer_aware,
    )
    driver = SimulatorDriver(sim_input, internal_mapper=mapper)
    driver.initialize()
    driver.initialize_data()
    driver.disable_external_mapper()
    status = driver.run()

    task_runtime = driver.state.get_task_runtime()
    n_data_events = task_runtime.get_n_data_tasks()
    n_data_moves = sum(
        1 for i in range(n_data_events) if not task_runtime.is_data_task_virtual(i)
    )

    return {
        "status": status,
        "sim_us": driver.time,
        "total_mv": sum(driver.total_data_movement()),
        "evict_mv": sum(driver.total_eviction_movement()),
        "n_data_moves": n_data_moves,
    }


def safe_run(
    label: str,
    regime: str,
    *,
    transition_kind: str = "device_threshold",
    max_mapped: Optional[int] = None,
    transfer_aware: bool = False,
    pipeline_depth: Optional[int] = None,
    **kwargs,
) -> SweepResult:
    try:
        result = run_darts(
            transition_kind=transition_kind,
            max_mapped=max_mapped,
            transfer_aware=transfer_aware,
            pipeline_depth=pipeline_depth,
            **kwargs,
        )
        if result["status"] != ExecutionState.COMPLETE:
            return SweepResult(label, regime, "FAILED", error=str(result["status"]))
        return SweepResult(
            label=label,
            regime=regime,
            status="OK",
            sim_s=result["sim_us"] / 1e6,
            total_mv=result["total_mv"],
            evict_mv=result["evict_mv"],
            n_data_moves=result["n_data_moves"],
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return SweepResult(label, regime, "ERROR", error=str(exc))


def print_table(rows: list[SweepResult], title: str) -> None:
    print()
    print(f"### {title}")
    hdr = f"{'Config':<56}  {'Status':<6}  {'sim(s)':>8}  {'total_mv':>10}  {'evict_mv':>10}  {'#moves':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.status == "OK":
            print(
                f"{r.label:<56}  {r.status:<6}  {r.sim_s:8.3f}  "
                f"{_bytes_hr(r.total_mv):>10}  {_bytes_hr(r.evict_mv):>10}  "
                f"{r.n_data_moves:>6}"
            )
        else:
            print(
                f"{r.label:<56}  {r.status:<6}  {'--':>8}  {'--':>10}  {'--':>10}  {'--':>6}  {r.error}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS config sweep — Jacobi")
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--gpu-mem-gb", type=float, default=GPU_MEM_GB)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--regime", choices=["abundant", "tight", "both"], default="both",
    )
    args = parser.parse_args()

    gpu_mem = int(args.gpu_mem_gb * 1e9)

    # Abundant: level_memory = 60% of aggregate GPU mem → no eviction
    level_memory_abundant = derive_level_memory_for_pressure(
        grid_n=args.grid_n, gpu_mem=gpu_mem,
        n_gpu_devices=N_GPU_DEVICES, target_overcommit_ratio=0.6,
    )
    # Tight: level_memory = 140% of aggregate GPU mem → eviction
    level_memory_tight = derive_level_memory_for_pressure(
        grid_n=args.grid_n, gpu_mem=gpu_mem,
        n_gpu_devices=N_GPU_DEVICES, target_overcommit_ratio=1.4,
    )

    regimes: list[tuple[str, int]] = []
    if args.regime in {"abundant", "both"}:
        regimes.append(("abundant", level_memory_abundant))
    if args.regime in {"tight", "both"}:
        regimes.append(("tight", level_memory_tight))

    print("=" * 80)
    print("DARTS Configuration Sweep — Jacobi")
    print(f"  grid={args.grid_n}x{args.grid_n}  steps={args.steps}  "
          f"gpu_mem/device={_bytes_hr(gpu_mem)}  n_gpus={N_GPU_DEVICES}")
    print(f"  abundant level_memory={_bytes_hr(level_memory_abundant)}  "
          f"tight level_memory={_bytes_hr(level_memory_tight)}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Axis 1: mapped_threshold sweep
    # -----------------------------------------------------------------------
    threshold_configs = [
        (f"DARTS  mt={t}",
         DARTSConfig(mapped_threshold=t, reserved_threshold=-1))
        for t in [0, 1, 2, 4, 8, 16]
    ]

    # -----------------------------------------------------------------------
    # Axis 2: extended_batch_cap sweep (mt=0)
    # -----------------------------------------------------------------------
    cap_configs = [
        (f"DARTS  ext_batch cap={c}  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=c))
        for c in [1, 2, 4, 8, 16, 32]
    ]

    # -----------------------------------------------------------------------
    # Axis 3: combined mapped_threshold + extended batch cap=4
    # -----------------------------------------------------------------------
    combined_configs = [
        (f"DARTS  ext_batch cap=4  mt={t}",
         DARTSConfig(mapped_threshold=t, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=4))
        for t in [0, 1, 2, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 4: reserved_threshold sweep
    # -----------------------------------------------------------------------
    reserved_threshold_configs = [
        (f"DARTS  rt={t}",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=t))
        for t in [0, 1, 2, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 5: reserved_threshold + extended_batch cap=2
    # -----------------------------------------------------------------------
    reserved_ext_configs = [
        (f"DARTS  ext_batch cap=2  rt={t}",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=t,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2))
        for t in [0, 1, 2, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 6: IWC — cascade_passes sweep, mt=0
    # -----------------------------------------------------------------------
    iwc_configs = [
        (f"DARTS  iwc passes={p}  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     intra_window_coordination=True, cascade_passes=p))
        for p in [1, 2, 3, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 7: IWC + extended_batch cap=2
    # -----------------------------------------------------------------------
    iwc_ext_configs = [
        (f"DARTS  iwc passes={p}  ext_batch cap=2  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2,
                     intra_window_coordination=True, cascade_passes=p))
        for p in [1, 2, 3, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 8: DARTSAdaptive — max_mapped sweep
    # -----------------------------------------------------------------------
    adaptive_max_mapped_values = [4, 8, 16, 32, 64]
    adaptive_configs = [
        (f"DARTS  adaptive rt=0  max={m}  ext_batch cap=2",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=0,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2))
        for m in adaptive_max_mapped_values
    ]

    # -----------------------------------------------------------------------
    # Axis 9: transfer_aware_data_launch_order (with best DARTS config)
    # -----------------------------------------------------------------------
    transfer_aware_configs = [
        ("DARTS  mt=0  transfer_aware=False",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1), False),
        ("DARTS  mt=0  transfer_aware=True",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1), True),
        ("DARTS  ext_batch cap=2  mt=0  transfer_aware=False",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2), False),
        ("DARTS  ext_batch cap=2  mt=0  transfer_aware=True",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2), True),
    ]

    # -----------------------------------------------------------------------
    # Axis 10: Finish-time-aware (FTA)
    # -----------------------------------------------------------------------
    fta_configs = [
        (f"DARTS  FTA  mt={t}",
         DARTSConfig(mapped_threshold=t, reserved_threshold=-1, finish_time_aware=True))
        for t in [0, 2, 4, 8]
    ] + [
        (f"DARTS  FTA  ext_batch cap={c}  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=c,
                     finish_time_aware=True))
        for c in [1, 2, 4]
    ]

    # -----------------------------------------------------------------------
    # Axis 11: DARTSPipeline transition — pipeline_depth sweep
    # -----------------------------------------------------------------------
    pipeline_depth_values = [1, 2, 4, 8, 16]
    pipeline_configs = [
        (f"DARTS  pipeline depth={d}  ext_batch cap=2",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2,
                     pipeline_depth=d))
        for d in pipeline_depth_values
    ]

    # -----------------------------------------------------------------------
    # Axis 12: Pipeline + FTA combined
    # -----------------------------------------------------------------------
    pipeline_fta_configs = [
        (f"DARTS  pipeline depth={d}  FTA  ext_batch cap=2",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2,
                     finish_time_aware=True, pipeline_depth=d))
        for d in [2, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Per-regime runs
    # -----------------------------------------------------------------------
    for regime_name, level_memory in regimes:
        print(f"\n{'='*80}")
        print(f"Regime: {regime_name}  level_memory={_bytes_hr(level_memory)}")
        print(f"{'='*80}")

        def run_axis(label_cfg_pairs, transition_kind="device_threshold", max_mapped=None, pipeline_depth=None):
            rows = []
            for label, cfg in label_cfg_pairs:
                print(f"  {label}  [{regime_name}]", end="", flush=True)
                r = safe_run(
                    label, regime_name, cfg=cfg,
                    gpu_mem=gpu_mem, level_memory=level_memory,
                    grid_n=args.grid_n, steps=args.steps,
                    seed=args.seed, top_k=args.top_k,
                    transition_kind=transition_kind, max_mapped=max_mapped,
                    pipeline_depth=pipeline_depth,
                )
                rows.append(r)
                if r.status == "OK":
                    print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}  #moves={r.n_data_moves}")
                else:
                    print(f"  FAILED: {r.error}")
            return rows

        rows1 = run_axis(threshold_configs)
        print_table(rows1, f"Axis 1 — mapped_threshold sweep [{regime_name}]")

        rows2 = run_axis(cap_configs)
        print_table(rows2, f"Axis 2 — extended_batch_cap sweep (mt=0) [{regime_name}]")

        rows3 = run_axis(combined_configs)
        print_table(rows3, f"Axis 3 — combined threshold + ext_batch cap=4 [{regime_name}]")

        rows4 = run_axis(reserved_threshold_configs)
        print_table(rows4, f"Axis 4 — reserved_threshold sweep [{regime_name}]")

        rows5 = run_axis(reserved_ext_configs)
        print_table(rows5, f"Axis 5 — reserved_threshold + ext_batch cap=2 [{regime_name}]")

        rows6 = run_axis(iwc_configs)
        print_table(rows6, f"Axis 6 — IWC cascade_passes sweep (mt=0) [{regime_name}]")

        rows7 = run_axis(iwc_ext_configs)
        print_table(rows7, f"Axis 7 — IWC + ext_batch cap=2 (mt=0) [{regime_name}]")

        rows8 = []
        for (label, cfg), max_m in zip(adaptive_configs, adaptive_max_mapped_values):
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(
                label, regime_name, cfg=cfg,
                gpu_mem=gpu_mem, level_memory=level_memory,
                grid_n=args.grid_n, steps=args.steps,
                seed=args.seed, top_k=args.top_k,
                transition_kind="darts_adaptive", max_mapped=max_m,
            )
            rows8.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}  #moves={r.n_data_moves}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows8, f"Axis 8 — DARTSAdaptive rt=0 + ext_batch cap=2 max_mapped sweep [{regime_name}]")

        # Axis 9: transfer_aware
        rows9 = []
        for label, cfg, ta in transfer_aware_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(
                label, regime_name, cfg=cfg,
                gpu_mem=gpu_mem, level_memory=level_memory,
                grid_n=args.grid_n, steps=args.steps,
                seed=args.seed, top_k=args.top_k,
                transfer_aware=ta,
            )
            rows9.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}  #moves={r.n_data_moves}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows9, f"Axis 9 — transfer_aware_data_launch_order [{regime_name}]")

        rows10 = run_axis(fta_configs)
        print_table(rows10, f"Axis 10 — Finish-time-aware (FTA) [{regime_name}]")

        rows11 = []
        for (label, cfg), depth in zip(pipeline_configs, pipeline_depth_values):
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(
                label, regime_name, cfg=cfg,
                gpu_mem=gpu_mem, level_memory=level_memory,
                grid_n=args.grid_n, steps=args.steps,
                seed=args.seed, top_k=args.top_k,
                transition_kind="darts_pipeline", pipeline_depth=depth,
            )
            rows11.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}  #moves={r.n_data_moves}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows11, f"Axis 11 — DARTSPipeline depth sweep [{regime_name}]")

        rows12 = []
        for label, cfg in pipeline_fta_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(
                label, regime_name, cfg=cfg,
                gpu_mem=gpu_mem, level_memory=level_memory,
                grid_n=args.grid_n, steps=args.steps,
                seed=args.seed, top_k=args.top_k,
                transition_kind="darts_pipeline", pipeline_depth=cfg.pipeline_depth,
            )
            rows12.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}  #moves={r.n_data_moves}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows12, f"Axis 12 — Pipeline + FTA combined [{regime_name}]")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
