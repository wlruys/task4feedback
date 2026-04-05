"""
bench_darts_sweep.py
Sweep DARTS configuration axes to diagnose the abundant-memory performance gap.

Axes tested:
  1. mapped_threshold (0, 1, 2, 4, 8) – in-flight pipeline depth per device
  2. extended_batch_emission_cap (1, 2, 4, 8, 16) – S1/S2/S3 tasks per window per device
  3. extended_frontier on/off with threshold=0 baseline

Runs both memory regimes:
  - abundant (gpu_mem = level_mem = no eviction)
  - tight    (gpu_mem = 0.3 × level_mem, forces eviction)

Reports sim_s, total_mv, evict_mv so the data-movement / time trade-off is visible.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
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
    make_sim_input,
    run_once,
    set_seed,
    _bytes_hr,
)
from task4feedback.fastsim2 import EvictionPolicy, ExecutionState

# ---------------------------------------------------------------------------
# Default benchmark parameters
# ---------------------------------------------------------------------------
GRID_N = 8
STEPS = 64
LEVEL_MEMORY_GB = 1.5
GPU_MEM_ABUNDANT_GB = 4.0    # no eviction
GPU_MEM_TIGHT_GB = 0.3       # heavy eviction
SEED = 0
REPS = 1
TOP_K = 64

GRAPH_KWARGS = dict(
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


@dataclass
class SweepResult:
    label: str
    regime: str
    status: str
    sim_s: Optional[float] = None
    total_mv: Optional[int] = None
    evict_mv: Optional[int] = None
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
) -> dict:
    set_seed(seed)
    mapper = make_darts_mapper(cfg)
    transition_config = TransitionConfig(
        kind=transition_kind,
        mapped_threshold=cfg.mapped_threshold,
        reserved_threshold=cfg.reserved_threshold,
        max_in_flight=max_mapped,
    )
    graph_system = build_system(mem=gpu_mem)
    graph = build_graph(
        graph_system,
        grid_n=grid_n,
        steps=steps,
        level_memory=level_memory,
        **GRAPH_KWARGS,
    )
    sim_system = build_system(mem=gpu_mem)
    sim_input = make_sim_input(
        graph,
        sim_system,
        mapper_name="darts",
        eviction_policy=EvictionPolicy.LRU,
        top_k_candidates=top_k,
        transition_config=transition_config,
    )
    return run_once(sim_input, mapper_name="darts", darts_config=cfg)


def safe_run(label: str, regime: str, transition_kind: str = "device_threshold",
             max_mapped: Optional[int] = None, **kwargs) -> SweepResult:
    try:
        result = run_darts(transition_kind=transition_kind, max_mapped=max_mapped, **kwargs)
        if result["status"] != ExecutionState.COMPLETE:
            return SweepResult(label, regime, "FAILED", error=str(result["status"]))
        return SweepResult(
            label=label,
            regime=regime,
            status="OK",
            sim_s=result["sim_us"] / 1e6,
            total_mv=result["total_mv"],
            evict_mv=result["evict_mv"],
        )
    except Exception as exc:
        return SweepResult(label, regime, "ERROR", error=str(exc))


def print_table(rows: list[SweepResult], title: str) -> None:
    print()
    print(f"### {title}")
    hdr = f"{'Config':<44}  {'Status':<6}  {'sim(s)':>8}  {'total_mv':>10}  {'evict_mv':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.status == "OK":
            print(
                f"{r.label:<44}  {r.status:<6}  {r.sim_s:8.3f}  "
                f"{_bytes_hr(r.total_mv):>10}  {_bytes_hr(r.evict_mv):>10}"
            )
        else:
            print(f"{r.label:<44}  {r.status:<6}  {'--':>8}  {'--':>10}  {'--':>10}  {r.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS config sweep")
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--level-memory-gb", type=float, default=LEVEL_MEMORY_GB)
    parser.add_argument("--gpu-mem-abundant-gb", type=float, default=GPU_MEM_ABUNDANT_GB)
    parser.add_argument("--gpu-mem-tight-gb", type=float, default=GPU_MEM_TIGHT_GB)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--regime", choices=["abundant", "tight", "both"], default="both",
    )
    args = parser.parse_args()

    level_memory = int(args.level_memory_gb * 1e9)
    gpu_mem_abundant = int(args.gpu_mem_abundant_gb * 1e9)
    gpu_mem_tight = int(args.gpu_mem_tight_gb * 1e9)

    regimes: list[tuple[str, int]] = []
    if args.regime in {"abundant", "both"}:
        regimes.append(("abundant", gpu_mem_abundant))
    if args.regime in {"tight", "both"}:
        regimes.append(("tight", gpu_mem_tight))

    common = dict(
        level_memory=level_memory,
        grid_n=args.grid_n,
        steps=args.steps,
        seed=args.seed,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("DARTS Configuration Sweep")
    print(f"  grid={args.grid_n}×{args.grid_n}  steps={args.steps}  "
          f"level_mem={_bytes_hr(level_memory)}")
    print(f"  abundant gpu_mem={_bytes_hr(gpu_mem_abundant)}  "
          f"tight gpu_mem={_bytes_hr(gpu_mem_tight)}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Axis 1: mapped_threshold sweep (base DARTS, no extended)
    # -----------------------------------------------------------------------
    threshold_configs = [
        (f"DARTS  mt={t}",
         DARTSConfig(mapped_threshold=t, reserved_threshold=-1))
        for t in [0, 1, 2, 4, 8, 16]
    ]

    # -----------------------------------------------------------------------
    # Axis 2: extended_batch_cap sweep (extended frontier + batch, mt=0)
    # -----------------------------------------------------------------------
    cap_configs = [
        (f"DARTS  ext_batch cap={c}  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=c))
        for c in [1, 2, 4, 8, 16, 32]
    ]

    # -----------------------------------------------------------------------
    # Axis 3: combined threshold + extended batch
    # -----------------------------------------------------------------------
    combined_configs = [
        (f"DARTS  ext_batch cap=4  mt={t}",
         DARTSConfig(mapped_threshold=t, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=4))
        for t in [0, 1, 2, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 4: reserved_threshold sweep (base DARTS, no extended)
    # reserved_threshold fires when n_reserved(device) <= threshold
    # (fires earlier than mapped_threshold — closer to StarPU reactive semantics)
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
    # Axis 6: intra-window coordination (IWC) — cascade_passes sweep, mt=0
    # Items 4+5: cross-device claim + S1→S0 cascade within one plan_tasks call
    # -----------------------------------------------------------------------
    iwc_configs = [
        (f"DARTS  iwc passes={p}  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     intra_window_coordination=True, cascade_passes=p))
        for p in [1, 2, 3, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 7: IWC + extended_batch cap=2 (best combo from axes 2 and 6)
    # -----------------------------------------------------------------------
    iwc_ext_configs = [
        (f"DARTS  iwc passes={p}  ext_batch cap=2  mt=0",
         DARTSConfig(mapped_threshold=0, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2,
                     intra_window_coordination=True, cascade_passes=p))
        for p in [1, 2, 3, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 8: DARTSAdaptiveTransitionConditions — max_mapped sweep, rt=0, ext_batch cap=2
    # Tests whether the global cap prevents tight-memory thrashing while preserving
    # the cascade behavior that makes rt=0 fast in abundant memory.
    # N_DEVICES=4, so multiples of 4 are natural pipeline depths.
    # -----------------------------------------------------------------------
    adaptive_max_mapped_values = [4, 8, 16, 32, 64]
    adaptive_configs = [
        (f"DARTS  adaptive rt=0  max={m}  ext_batch cap=2",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=0,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2))
        for m in adaptive_max_mapped_values
    ]

    # -----------------------------------------------------------------------
    # Reference baselines (added by importing EFT mapper directly)
    # -----------------------------------------------------------------------
    # We don't re-run EFT here — instead we note the 0.020s / 1.6 GB baseline from the report.

    for regime_name, gpu_mem in regimes:
        print(f"\n{'='*80}")
        print(f"Regime: {regime_name}  (gpu_mem={_bytes_hr(gpu_mem)})")
        print(f"{'='*80}")

        # Axis 1
        rows1 = []
        for label, cfg in threshold_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem, **common)
            rows1.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows1, f"Axis 1 — mapped_threshold sweep [{regime_name}]")

        # Axis 2
        rows2 = []
        for label, cfg in cap_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem, **common)
            rows2.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows2, f"Axis 2 — extended_batch_cap sweep (mt=0) [{regime_name}]")

        # Axis 3
        rows3 = []
        for label, cfg in combined_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem, **common)
            rows3.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows3, f"Axis 3 — combined threshold + ext_batch cap=4 [{regime_name}]")

        # Axis 4: reserved_threshold
        rows4 = []
        for label, cfg in reserved_threshold_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem,
                         transition_kind="device_threshold", **common)
            rows4.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows4, f"Axis 4 — reserved_threshold sweep [{regime_name}]")

        # Axis 5: reserved_threshold + ext_batch cap=2
        rows5 = []
        for label, cfg in reserved_ext_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem,
                         transition_kind="device_threshold", **common)
            rows5.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows5, f"Axis 5 — reserved_threshold + ext_batch cap=2 [{regime_name}]")

        # Axis 6: IWC cascade_passes sweep
        rows6 = []
        for label, cfg in iwc_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem, **common)
            rows6.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows6, f"Axis 6 — IWC cascade_passes sweep (mt=0) [{regime_name}]")

        # Axis 7: IWC + ext_batch cap=2
        rows7 = []
        for label, cfg in iwc_ext_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem, **common)
            rows7.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows7, f"Axis 7 — IWC + ext_batch cap=2 (mt=0) [{regime_name}]")

        # Axis 8: DARTSAdaptiveTransitionConditions — max_mapped sweep
        rows8 = []
        for (label, cfg), max_m in zip(adaptive_configs, adaptive_max_mapped_values):
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(label, regime_name, cfg=cfg, gpu_mem=gpu_mem,
                         transition_kind="darts_adaptive", max_mapped=max_m, **common)
            rows8.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows8, f"Axis 8 — DARTSAdaptive rt=0 + ext_batch cap=2 max_mapped sweep [{regime_name}]")

    print()
    print("=" * 80)
    print("Reference (from bench_compare_report.md, 8×8 64-step, abundant memory):")
    print("  DequeueEFT: sim=0.020s  total_mv=1.6 GB")
    print("  DARTS(mt=0): sim=0.288s  total_mv=94 GB")
    print("  ExtendedDARS(mt=0,cap=4): sim=0.147s  total_mv=44 GB")
    print("=" * 80)


if __name__ == "__main__":
    main()
