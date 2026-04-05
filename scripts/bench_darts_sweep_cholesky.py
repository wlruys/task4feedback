"""
bench_darts_sweep_cholesky.py
Sweep DARTS configuration axes on the block Cholesky factorization workload.

Replicates the same 8 axes as bench_darts_sweep.py (Jacobi) but for Cholesky,
allowing direct comparison of DARTS tuning across two qualitatively different
workloads.

Problem structure:
  - n×n lower-triangular block matrix; n_blocks controls problem size
  - Data footprint: n*(n+1)/2 blocks × block_bytes
  - Two memory regimes derived from target overcommit ratios:
      abundant: total_data / aggregate_gpu_mem = 0.6  (no eviction)
      tight:    total_data / aggregate_gpu_mem = 1.4  (moderate eviction)

Default: n_blocks=16, GPU_MEM=4 GB/GPU, 4 GPU devices.
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
from bench_eviction_cholesky import (
    N_DEVICES,
    build_graph,
    build_system,
    derive_block_bytes_for_pressure,
    lower_triangular_block_count,
    set_seed,
    _bytes_hr,
    estimate_footprint,
)
from task4feedback.fastsim2 import EvictionPolicy, ExecutionState
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput
from task4feedback.fastsim2 import TaskNoise

# ---------------------------------------------------------------------------
# Default benchmark parameters
# ---------------------------------------------------------------------------
N_BLOCKS = 16
GPU_MEM_GB = 4.0          # per-GPU capacity used for block-size derivation
ABUNDANT_RATIO = 0.6      # total data / aggregate GPU mem
TIGHT_RATIO = 1.4
SEED = 0
TOP_K = 256

N_GPU_DEVICES = N_DEVICES - 1   # number of non-CPU devices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    block_bytes: int,
    n_blocks: int,
    seed: int,
    top_k: int,
    transition_kind: str = "device_threshold",
    max_mapped: Optional[int] = None,
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
        n_blocks=n_blocks,
        block_bytes=block_bytes,
        randomize_initial_placement=True,
        verbose=False,
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
    )
    driver = SimulatorDriver(sim_input, internal_mapper=mapper)
    driver.initialize()
    driver.initialize_data()
    driver.disable_external_mapper()
    status = driver.run()
    return {
        "status": status,
        "sim_us": driver.time,
        "total_mv": sum(driver.total_data_movement()),
        "evict_mv": sum(driver.total_eviction_movement()),
    }


def safe_run(
    label: str,
    regime: str,
    *,
    transition_kind: str = "device_threshold",
    max_mapped: Optional[int] = None,
    pipeline_depth: Optional[int] = None,
    **kwargs,
) -> SweepResult:
    try:
        result = run_darts(
            transition_kind=transition_kind,
            max_mapped=max_mapped,
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
        )
    except Exception as exc:
        import traceback
        return SweepResult(label, regime, "ERROR", error=str(exc))


def print_table(rows: list[SweepResult], title: str) -> None:
    print()
    print(f"### {title}")
    hdr = f"{'Config':<52}  {'Status':<6}  {'sim(s)':>8}  {'total_mv':>10}  {'evict_mv':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.status == "OK":
            print(
                f"{r.label:<52}  {r.status:<6}  {r.sim_s:8.3f}  "
                f"{_bytes_hr(r.total_mv):>10}  {_bytes_hr(r.evict_mv):>10}"
            )
        else:
            print(
                f"{r.label:<52}  {r.status:<6}  {'--':>8}  {'--':>10}  {'--':>10}  {r.error}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS config sweep — Cholesky")
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--gpu-mem-gb", type=float, default=GPU_MEM_GB,
                        help="Per-GPU capacity used to derive block sizes")
    parser.add_argument("--abundant-ratio", type=float, default=ABUNDANT_RATIO,
                        help="Target overcommit ratio for abundant regime")
    parser.add_argument("--tight-ratio", type=float, default=TIGHT_RATIO,
                        help="Target overcommit ratio for tight regime")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--regime", choices=["abundant", "tight", "both"], default="both",
    )
    args = parser.parse_args()

    gpu_mem = int(args.gpu_mem_gb * 1e9)

    block_bytes_abundant = derive_block_bytes_for_pressure(
        n_blocks=args.n_blocks,
        gpu_mem=gpu_mem,
        n_gpu_devices=N_GPU_DEVICES,
        target_overcommit_ratio=args.abundant_ratio,
    )
    block_bytes_tight = derive_block_bytes_for_pressure(
        n_blocks=args.n_blocks,
        gpu_mem=gpu_mem,
        n_gpu_devices=N_GPU_DEVICES,
        target_overcommit_ratio=args.tight_ratio,
    )

    regimes: list[tuple[str, int]] = []
    if args.regime in {"abundant", "both"}:
        regimes.append(("abundant", block_bytes_abundant))
    if args.regime in {"tight", "both"}:
        regimes.append(("tight", block_bytes_tight))

    def _print_problem(block_bytes: int) -> None:
        fp = estimate_footprint(
            n_blocks=args.n_blocks,
            block_bytes=block_bytes,
            gpu_mem=gpu_mem,
            n_gpu_devices=N_GPU_DEVICES,
        )
        print(f"  n_blocks={args.n_blocks}  lower_tri_blocks={fp['total_blocks']}  "
              f"block_size={_bytes_hr(block_bytes)}")
        print(f"  total_data={_bytes_hr(fp['total_data_bytes'])}  "
              f"overcommit={fp['overcommit_ratio']:.2f}x  "
              f"aggregate_gpu_mem={_bytes_hr(fp['aggregate_gpu_mem'])}")

    print("=" * 80)
    print("DARTS Configuration Sweep — Cholesky")
    print(f"  n_blocks={args.n_blocks}  gpu_mem/device={_bytes_hr(gpu_mem)}  "
          f"n_gpus={N_GPU_DEVICES}")
    print(f"  abundant block_bytes={_bytes_hr(block_bytes_abundant)}  "
          f"tight block_bytes={_bytes_hr(block_bytes_tight)}")
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
    # Axis 3: combined mapped_threshold + extended batch cap=4
    # -----------------------------------------------------------------------
    combined_configs = [
        (f"DARTS  ext_batch cap=4  mt={t}",
         DARTSConfig(mapped_threshold=t, reserved_threshold=-1,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=4))
        for t in [0, 1, 2, 4, 8]
    ]

    # -----------------------------------------------------------------------
    # Axis 4: reserved_threshold sweep (base DARTS, no extended)
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
    # Axis 8: DARTSAdaptiveTransitionConditions — max_mapped sweep
    # rt=0, ext_batch cap=2; N_GPU_DEVICES=4 so multiples of 4 are natural
    # -----------------------------------------------------------------------
    adaptive_max_mapped_values = [4, 8, 16, 32, 64]
    adaptive_configs = [
        (f"DARTS  adaptive rt=0  max={m}  ext_batch cap=2",
         DARTSConfig(mapped_threshold=-1, reserved_threshold=0,
                     extended_frontier=True, extended_batch=True, extended_batch_cap=2))
        for m in adaptive_max_mapped_values
    ]

    # -----------------------------------------------------------------------
    # Axis 9: Finish-time-aware (FTA) — blends EFT load balancing with DARTS
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
    # Axis 10: DARTSPipeline transition — pipeline_depth sweep
    # Uses n_mapped(device) < pipeline_depth instead of broken reserved_threshold
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
    # Axis 11: Pipeline + FTA combined
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
    for regime_name, block_bytes in regimes:
        print(f"\n{'='*80}")
        print(f"Regime: {regime_name}")
        _print_problem(block_bytes)
        print(f"{'='*80}")

        common = dict(
            gpu_mem=gpu_mem,
            block_bytes=block_bytes,
            n_blocks=args.n_blocks,
            seed=args.seed,
            top_k=args.top_k,
            cfg=None,  # overridden per entry
        )

        def run_axis(label_cfg_pairs, transition_kind="device_threshold", max_mapped=None, pipeline_depth=None):
            rows = []
            for label, cfg in label_cfg_pairs:
                print(f"  {label}  [{regime_name}]", end="", flush=True)
                r = safe_run(
                    label, regime_name, cfg=cfg,
                    gpu_mem=gpu_mem, block_bytes=block_bytes,
                    n_blocks=args.n_blocks, seed=args.seed, top_k=args.top_k,
                    transition_kind=transition_kind, max_mapped=max_mapped,
                    pipeline_depth=pipeline_depth,
                )
                rows.append(r)
                if r.status == "OK":
                    print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
                else:
                    print(f"  FAILED: {r.error}")
            return rows

        rows1 = run_axis(threshold_configs)
        print_table(rows1, f"Axis 1 — mapped_threshold sweep [{regime_name}]")

        rows2 = run_axis(cap_configs)
        print_table(rows2, f"Axis 2 — extended_batch_cap sweep (mt=0) [{regime_name}]")

        rows3 = run_axis(combined_configs)
        print_table(rows3, f"Axis 3 — combined threshold + ext_batch cap=4 [{regime_name}]")

        rows4 = run_axis(reserved_threshold_configs, transition_kind="device_threshold")
        print_table(rows4, f"Axis 4 — reserved_threshold sweep [{regime_name}]")

        rows5 = run_axis(reserved_ext_configs, transition_kind="device_threshold")
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
                gpu_mem=gpu_mem, block_bytes=block_bytes,
                n_blocks=args.n_blocks, seed=args.seed, top_k=args.top_k,
                transition_kind="darts_adaptive", max_mapped=max_m,
            )
            rows8.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows8, f"Axis 8 — DARTSAdaptive rt=0 + ext_batch cap=2 max_mapped sweep [{regime_name}]")

        rows9 = run_axis(fta_configs)
        print_table(rows9, f"Axis 9 — Finish-time-aware (FTA) [{regime_name}]")

        rows10 = []
        for (label, cfg), depth in zip(pipeline_configs, pipeline_depth_values):
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(
                label, regime_name, cfg=cfg,
                gpu_mem=gpu_mem, block_bytes=block_bytes,
                n_blocks=args.n_blocks, seed=args.seed, top_k=args.top_k,
                transition_kind="darts_pipeline", pipeline_depth=depth,
            )
            rows10.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows10, f"Axis 10 — DARTSPipeline depth sweep [{regime_name}]")

        rows11 = []
        for label, cfg in pipeline_fta_configs:
            print(f"  {label}  [{regime_name}]", end="", flush=True)
            r = safe_run(
                label, regime_name, cfg=cfg,
                gpu_mem=gpu_mem, block_bytes=block_bytes,
                n_blocks=args.n_blocks, seed=args.seed, top_k=args.top_k,
                transition_kind="darts_pipeline", pipeline_depth=cfg.pipeline_depth,
            )
            rows11.append(r)
            if r.status == "OK":
                print(f"  sim={r.sim_s:.3f}s  total_mv={_bytes_hr(r.total_mv)}")
            else:
                print(f"  FAILED: {r.error}")
        print_table(rows11, f"Axis 11 — Pipeline + FTA combined [{regime_name}]")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
