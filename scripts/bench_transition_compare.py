"""
bench_transition_compare.py

Compares DARTS transition condition families against EFT baselines:
  Baselines:
    - DequeueEFT         (batch transition, LRU eviction)
    - MemoryAwareEFT     (batch transition, LRU eviction; alpha sweep)
  DARTS variants (mapper fixed to DARTSMapper ext_batch cap=2):
    1. DeviceThreshold   — mapped_threshold sweep
    2. DARTSAdaptive     — max_mapped sweep (reserved_threshold=0, starvation=1)
    3. DARTSPipeline     — pipeline_depth sweep (max_in_flight=8..64, starvation=1)

Run on Cholesky and Jacobi in abundant and tight memory regimes.

Usage:
    python bench_transition_compare.py [--workload cholesky|jacobi|both]
                                       [--regime abundant|tight|both]
                                       [--n-blocks N] [--grid-n N]
                                       [--gpu-mem-gb F]
                                       [--seed N] [--top-k N]
"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from typing import Optional

from bench_mapper_support import (
    DARTSConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    ExternalMapperConfig,
    make_darts_mapper,
    make_transition_conditions,
    make_internal_mapper,
)
from bench_eviction_cholesky import (
    N_DEVICES as CHOL_N_DEVICES,
    TOP_K_CANDIDATES as CHOL_TOP_K,
    build_graph as chol_build_graph,
    build_system as chol_build_system,
    derive_block_bytes_for_pressure,
    estimate_footprint,
    lower_triangular_block_count,
    make_sim_input as chol_make_sim_input,
    run_once as chol_run_once,
    set_seed as chol_set_seed,
    _bytes_hr,
)
from bench_eviction_jacobi import (
    N_DEVICES as JAC_N_DEVICES,
    TOP_K_CANDIDATES as JAC_TOP_K,
    build_graph as jac_build_graph,
    build_system as jac_build_system,
    make_sim_input as jac_make_sim_input,
    run_once as jac_run_once,
    set_seed as jac_set_seed,
)
from task4feedback.fastsim2 import EvictionPolicy, ExecutionState, TaskNoise
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_BLOCKS = 16          # Cholesky: blocks per side
GRID_N = 16            # Jacobi: grid cells per side
JAC_STEPS = 256        # Jacobi: time steps
GPU_MEM_GB = 4.0       # per-GPU capacity
ABUNDANT_RATIO = 0.6   # total_data / aggregate_gpu_mem (no eviction)
TIGHT_RATIO = 1.4      # total_data / aggregate_gpu_mem (moderate eviction)
SEED = 0
TOP_K = 256

# Base DARTS mapper options shared across all configurations.
# Device-selection fields (mapped_threshold, reserved_threshold, pipeline_depth,
# starvation_threshold, max_in_flight) are filled per-variant so the mapper's
# device-selection logic is coherent with the transition condition being tested.
_MAPPER_COMMON = dict(
    extended_frontier=True,
    extended_batch=True,
    extended_batch_cap=2,
)


def _make_mapper_cfg(**overrides) -> DARTSConfig:
    """Merge _MAPPER_COMMON with per-variant overrides into a DARTSConfig."""
    return DARTSConfig(**{**_MAPPER_COMMON, **overrides})

CHOL_N_GPU = CHOL_N_DEVICES - 1
JAC_N_GPU = JAC_N_DEVICES - 1

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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class Row:
    tc_family: str
    param_str: str
    regime: str
    workload: str
    status: str
    sim_s: Optional[float] = None
    total_mv: Optional[int] = None
    evict_mv: Optional[int] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers: derive level_memory for Jacobi
# ---------------------------------------------------------------------------
def derive_jac_level_memory(gpu_mem: int, ratio: float) -> int:
    return int(JAC_N_GPU * gpu_mem * ratio)


# ---------------------------------------------------------------------------
# Run a single DARTS simulation (Cholesky)
# ---------------------------------------------------------------------------
def _run_chol(
    cfg: DARTSConfig,
    *,
    gpu_mem: int,
    block_bytes: int,
    n_blocks: int,
    seed: int,
    top_k: int,
    tc_kind: str,
    tc_config: TransitionConfig,
) -> dict:
    chol_set_seed(seed)
    mapper = make_darts_mapper(cfg)
    graph_system = chol_build_system(mem=gpu_mem)
    graph = chol_build_graph(
        graph_system,
        n_blocks=n_blocks,
        block_bytes=block_bytes,
        randomize_initial_placement=True,
        verbose=False,
    )
    sim_system = chol_build_system(mem=gpu_mem)
    blocks = graph.get_blocks()
    noise = TaskNoise(graph.static_graph)
    sim_input = SimulatorInput(
        graph, blocks, sim_system,
        task_noise=noise,
        transition_conditions=make_transition_conditions(
            "darts", top_k_candidates=top_k, config=tc_config,
        ),
        top_k_candidates=top_k,
        eviction_policy=EvictionPolicy.LRU,
    )
    drv = SimulatorDriver(sim_input, internal_mapper=mapper)
    drv.initialize()
    drv.initialize_data()
    drv.disable_external_mapper()
    status = drv.run()
    return {
        "status": status,
        "sim_us": drv.time,
        "total_mv": sum(drv.total_data_movement()),
        "evict_mv": sum(drv.total_eviction_movement()),
    }


# ---------------------------------------------------------------------------
# Run a single DARTS simulation (Jacobi)
# ---------------------------------------------------------------------------
def _run_jac(
    cfg: DARTSConfig,
    *,
    gpu_mem: int,
    level_memory: int,
    grid_n: int,
    steps: int,
    seed: int,
    top_k: int,
    tc_kind: str,
    tc_config: TransitionConfig,
) -> dict:
    jac_set_seed(seed)
    mapper = make_darts_mapper(cfg)
    graph_system = jac_build_system(mem=gpu_mem)
    graph = jac_build_graph(
        graph_system,
        grid_n=grid_n,
        steps=steps,
        level_memory=level_memory,
        randomize_initial_placement=True,
        **JACOBI_DEFAULTS,
    )
    sim_system = jac_build_system(mem=gpu_mem)
    blocks = graph.get_blocks()
    noise = TaskNoise(graph.static_graph)
    sim_input = SimulatorInput(
        graph, blocks, sim_system,
        task_noise=noise,
        transition_conditions=make_transition_conditions(
            "darts", top_k_candidates=top_k, config=tc_config,
        ),
        top_k_candidates=top_k,
        eviction_policy=EvictionPolicy.LRU,
    )
    drv = SimulatorDriver(sim_input, internal_mapper=mapper)
    drv.initialize()
    drv.initialize_data()
    drv.disable_external_mapper()
    status = drv.run()
    return {
        "status": status,
        "sim_us": drv.time,
        "total_mv": sum(drv.total_data_movement()),
        "evict_mv": sum(drv.total_eviction_movement()),
    }


# ---------------------------------------------------------------------------
# EFT baseline runners (use existing make_sim_input / run_once from bench files)
# ---------------------------------------------------------------------------
def _run_eft_chol(
    *,
    mapper_name: str,
    gpu_mem: int,
    block_bytes: int,
    n_blocks: int,
    seed: int,
    meft_config: Optional[MemoryAwareEFTConfig] = None,
) -> dict:
    chol_set_seed(seed)
    graph_system = chol_build_system(mem=gpu_mem)
    graph = chol_build_graph(
        graph_system,
        n_blocks=n_blocks,
        block_bytes=block_bytes,
        randomize_initial_placement=True,
        verbose=False,
    )
    sim_system = chol_build_system(mem=gpu_mem)
    sim_input = chol_make_sim_input(
        graph, sim_system, mapper_name,
        EvictionPolicy.LRU,
        TransitionConfig(),   # default (batch) transition for EFT
    )
    result = chol_run_once(
        sim_input,
        mapper_name=mapper_name,
        memory_aware_eft_alpha=meft_config.alpha if meft_config else 1.0,
        external_mapper_config=ExternalMapperConfig(),
    )
    return result


def _run_eft_jac(
    *,
    mapper_name: str,
    gpu_mem: int,
    level_memory: int,
    grid_n: int,
    steps: int,
    seed: int,
    meft_config: Optional[MemoryAwareEFTConfig] = None,
) -> dict:
    jac_set_seed(seed)
    graph_system = jac_build_system(mem=gpu_mem)
    graph = jac_build_graph(
        graph_system,
        grid_n=grid_n,
        steps=steps,
        level_memory=level_memory,
        randomize_initial_placement=True,
        **JACOBI_DEFAULTS,
    )
    sim_system = jac_build_system(mem=gpu_mem)
    sim_input = jac_make_sim_input(
        graph, sim_system,
        mapper_name=mapper_name,
        eviction_policy=EvictionPolicy.LRU,
        top_k_candidates=JAC_TOP_K,
        transition_config=TransitionConfig(),
    )
    result = jac_run_once(
        sim_input,
        mapper_name=mapper_name,
        memory_aware_eft_config=meft_config,
        external_mapper_config=ExternalMapperConfig(),
    )
    return result


def build_eft_variants() -> list[tuple[str, str, dict]]:
    """Return (tc_family, param_str, run_kwargs) for EFT baselines."""
    variants = []
    variants.append(("EFT-baseline", "DequeueEFT", {"mapper_name": "dequeue_eft"}))
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        cfg = MemoryAwareEFTConfig(alpha=alpha)
        variants.append((
            "EFT-baseline",
            f"MemoryAwareEFT alpha={alpha:g}",
            {"mapper_name": "memory_aware_eft", "meft_config": cfg},
        ))
    return variants


# ---------------------------------------------------------------------------
# Safe wrapper
# ---------------------------------------------------------------------------
def safe_run(run_fn, label_prefix: str, tc_family: str, param_str: str,
             regime: str, workload: str, **kwargs) -> Row:
    label = f"{tc_family}  {param_str}"
    print(f"  {workload:10s} [{regime:9s}]  {label:<48}", end="", flush=True)
    try:
        r = run_fn(**kwargs)
        if r["status"] != ExecutionState.COMPLETE:
            print(f"  FAILED ({r['status']})")
            return Row(tc_family, param_str, regime, workload, "FAILED",
                       error=str(r["status"]))
        sim_s = r["sim_us"] / 1e6
        print(f"  sim={sim_s:.3f}s  mv={_bytes_hr(r['total_mv'])}  evict={_bytes_hr(r['evict_mv'])}")
        return Row(tc_family, param_str, regime, workload, "OK",
                   sim_s=sim_s, total_mv=r["total_mv"], evict_mv=r["evict_mv"])
    except Exception as exc:
        traceback.print_exc()
        return Row(tc_family, param_str, regime, workload, "ERROR", error=str(exc))


# ---------------------------------------------------------------------------
# Build all TC variants to run
# ---------------------------------------------------------------------------
def build_tc_variants(top_k: int, max_in_flight: int = 64) -> list[tuple[str, str, TransitionConfig, DARTSConfig]]:
    """Return list of (tc_family, param_str, TransitionConfig, DARTSConfig).

    The DARTSConfig's device-selection fields are set to match the TC semantics:
      - DeviceThreshold → legacy threshold mode (pipeline_depth=0)
      - DARTSAdaptive   → reserved_threshold mode (pipeline_depth=0)
      - DARTSPipeline   → pipeline_depth mode (pipeline_depth=D)
    """
    def mapper_for_dt(mt: int, rt: int) -> DARTSConfig:
        """DeviceThreshold: mapper uses same mapped/reserved threshold."""
        return _make_mapper_cfg(mapped_threshold=mt, reserved_threshold=rt, pipeline_depth=0)

    def mapper_for_adaptive(rt: int, stv: int, mif: int) -> DARTSConfig:
        """DARTSAdaptive: mapper uses reserved_threshold in legacy mode."""
        return _make_mapper_cfg(mapped_threshold=-1, reserved_threshold=rt,
                                pipeline_depth=0, starvation_threshold=stv,
                                max_in_flight=mif)

    def mapper_for_pipeline(depth: int, stv: int, mif: int) -> DARTSConfig:
        """DARTSPipeline: mapper uses pipeline_depth mode."""
        return _make_mapper_cfg(mapped_threshold=-1, reserved_threshold=-1,
                                pipeline_depth=depth, starvation_threshold=stv,
                                max_in_flight=mif)

    variants = []

    # ── DeviceThreshold: mapped_threshold sweep ──────────────────────────────
    for mt in [0, 1, 2, 4, 8]:
        variants.append((
            "DeviceThreshold",
            f"mt={mt}",
            TransitionConfig(kind="device_threshold",
                             mapped_threshold=mt, reserved_threshold=-1),
            mapper_for_dt(mt, -1),
        ))

    # ── DeviceThreshold: reserved_threshold sweep ────────────────────────────
    for rt in [0, 1, 2, 4]:
        variants.append((
            "DeviceThreshold",
            f"rt={rt}",
            TransitionConfig(kind="device_threshold",
                             mapped_threshold=-1, reserved_threshold=rt),
            mapper_for_dt(-1, rt),
        ))

    # ── DARTSAdaptive: max_mapped sweep (rt=0, starvation=1) ─────────────────
    for max_m in [8, 16, 32, 64]:
        variants.append((
            "DARTSAdaptive",
            f"max={max_m} rt=0 stv=1",
            TransitionConfig(kind="darts_adaptive",
                             reserved_threshold=0,
                             max_in_flight=max_m,
                             pipeline_starvation=1),
            mapper_for_adaptive(0, 1, max_m),
        ))

    # ── DARTSAdaptive: starvation_threshold sweep (rt=0, max=32, stv > 1) ─────
    # stv=1 is already covered above; only sweep values > 1 here.
    for stv in [2, 4, 8]:
        variants.append((
            "DARTSAdaptive",
            f"max=32 rt=0 stv={stv}",
            TransitionConfig(kind="darts_adaptive",
                             reserved_threshold=0,
                             max_in_flight=32,
                             pipeline_starvation=stv),
            mapper_for_adaptive(0, stv, 32),
        ))

    # ── DARTSPipeline: pipeline_depth sweep (max_in_flight, starvation=1) ────
    for depth in [1, 2, 4, 8, 16]:
        variants.append((
            "DARTSPipeline",
            f"depth={depth} max={max_in_flight} stv=1",
            TransitionConfig(kind="darts_pipeline",
                             pipeline_depth=depth,
                             max_in_flight=max_in_flight,
                             pipeline_starvation=1),
            mapper_for_pipeline(depth, 1, max_in_flight),
        ))

    # ── DARTSPipeline: starvation_threshold sweep (depth=4, max=64, stv > 1) ─
    # stv=1 is covered in the depth and max sweeps; only sweep values > 1.
    for stv in [2, 4, 8]:
        variants.append((
            "DARTSPipeline",
            f"depth=4 max={max_in_flight} stv={stv}",
            TransitionConfig(kind="darts_pipeline",
                             pipeline_depth=4,
                             max_in_flight=max_in_flight,
                             pipeline_starvation=stv),
            mapper_for_pipeline(4, stv, max_in_flight),
        ))

    # ── DARTSPipeline: max_in_flight sweep (depth=4, starvation=1) ───────────
    for mif in [8, 16, 32, 64, 128]:
        variants.append((
            "DARTSPipeline",
            f"depth=4 max={mif} stv=1",
            TransitionConfig(kind="darts_pipeline",
                             pipeline_depth=4,
                             max_in_flight=mif,
                             pipeline_starvation=1),
            mapper_for_pipeline(4, 1, mif),
        ))

    return variants


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
def print_table(rows: list[Row], title: str) -> None:
    print()
    print(f"### {title}")
    hdr = f"  {'TC Family':<20}  {'Param':<32}  {'Status':<6}  {'sim(s)':>8}  {'total_mv':>10}  {'evict_mv':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    last_family = None
    for r in rows:
        # separator between TC families
        if last_family is not None and r.tc_family != last_family:
            print()
        last_family = r.tc_family
        if r.status == "OK":
            print(
                f"  {r.tc_family:<20}  {r.param_str:<32}  {r.status:<6}  "
                f"{r.sim_s:8.3f}  {_bytes_hr(r.total_mv):>10}  {_bytes_hr(r.evict_mv):>10}"
            )
        else:
            print(
                f"  {r.tc_family:<20}  {r.param_str:<32}  {r.status:<6}  "
                f"{'--':>8}  {'--':>10}  {'--':>10}  {r.error}"
            )


def print_summary(all_rows: list[Row]) -> None:
    """Print one-line best-per-family summary across all workload/regime combos."""
    print()
    print("=" * 80)
    print("SUMMARY — best sim time per (workload, regime, TC family)")
    print("=" * 80)
    hdr = f"  {'Workload':<10}  {'Regime':<9}  {'TC Family':<20}  {'Best param':<32}  {'sim(s)':>8}  {'evict_mv':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    from itertools import groupby
    ok_rows = [r for r in all_rows if r.status == "OK"]
    key_fn = lambda r: (r.workload, r.regime, r.tc_family)
    for key, grp in groupby(sorted(ok_rows, key=key_fn), key=key_fn):
        workload, regime, family = key
        best = min(grp, key=lambda r: r.sim_s)
        print(
            f"  {workload:<10}  {regime:<9}  {family:<20}  {best.param_str:<32}  "
            f"{best.sim_s:8.3f}  {_bytes_hr(best.evict_mv):>10}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="DARTS TC comparison — Cholesky + Jacobi")
    parser.add_argument("--workload", choices=["cholesky", "jacobi", "both"], default="both")
    parser.add_argument("--regime", choices=["abundant", "tight", "both"], default="both")
    parser.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--steps", type=int, default=JAC_STEPS)
    parser.add_argument("--gpu-mem-gb", type=float, default=GPU_MEM_GB)
    parser.add_argument("--abundant-ratio", type=float, default=ABUNDANT_RATIO)
    parser.add_argument("--tight-ratio", type=float, default=TIGHT_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-in-flight", type=int, default=64,
                        help="Max in-flight cap for pipeline/adaptive TCs (default 64)")
    args = parser.parse_args()

    gpu_mem = int(args.gpu_mem_gb * 1e9)
    tc_variants = build_tc_variants(args.top_k, args.max_in_flight)

    workloads = []
    if args.workload in {"cholesky", "both"}:
        workloads.append("cholesky")
    if args.workload in {"jacobi", "both"}:
        workloads.append("jacobi")

    regimes = []
    if args.regime in {"abundant", "both"}:
        regimes.append("abundant")
    if args.regime in {"tight", "both"}:
        regimes.append("tight")

    eft_variants = build_eft_variants()

    print("=" * 80)
    print("DARTS TC + EFT Baseline Comparison")
    print(f"  Workloads   : {workloads}")
    print(f"  Regimes     : {regimes}")
    print(f"  gpu_mem/dev : {_bytes_hr(gpu_mem)}  n_gpus_chol={CHOL_N_GPU}  n_gpus_jac={JAC_N_GPU}")
    print(f"  DARTS variants : {len(tc_variants)}  (DeviceThreshold, DARTSAdaptive, DARTSPipeline)")
    print(f"  EFT baselines  : {len(eft_variants)}  (DequeueEFT, MemoryAwareEFT alpha sweep)")
    print("=" * 80)

    all_rows: list[Row] = []

    for regime in regimes:
        ratio = args.abundant_ratio if regime == "abundant" else args.tight_ratio

        # Cholesky regime params
        block_bytes = derive_block_bytes_for_pressure(
            n_blocks=args.n_blocks,
            gpu_mem=gpu_mem,
            n_gpu_devices=CHOL_N_GPU,
            target_overcommit_ratio=ratio,
        )
        fp = estimate_footprint(
            n_blocks=args.n_blocks,
            block_bytes=block_bytes,
            gpu_mem=gpu_mem,
            n_gpu_devices=CHOL_N_GPU,
        )

        # Jacobi regime params
        jac_level_mem = derive_jac_level_memory(gpu_mem, ratio)

        print(f"\n{'='*80}")
        print(f"Regime: {regime}  (ratio={ratio:.1f}x)")
        if "cholesky" in workloads:
            print(f"  Cholesky: n_blocks={args.n_blocks}  block={_bytes_hr(block_bytes)}  "
                  f"total={_bytes_hr(fp['total_data_bytes'])}  overcommit={fp['overcommit_ratio']:.2f}x")
        if "jacobi" in workloads:
            print(f"  Jacobi:   grid={args.grid_n}x{args.grid_n}  steps={args.steps}  "
                  f"level_mem={_bytes_hr(jac_level_mem)}")
        print(f"{'='*80}")

        regime_rows: list[Row] = []

        # ── EFT baselines ────────────────────────────────────────────────────
        for tc_family, param_str, run_kwargs in eft_variants:
            if "cholesky" in workloads:
                row = safe_run(
                    _run_eft_chol, tc_family, tc_family, param_str,
                    regime, "cholesky",
                    gpu_mem=gpu_mem,
                    block_bytes=block_bytes,
                    n_blocks=args.n_blocks,
                    seed=args.seed,
                    **run_kwargs,
                )
                regime_rows.append(row)
                all_rows.append(row)

            if "jacobi" in workloads:
                row = safe_run(
                    _run_eft_jac, tc_family, tc_family, param_str,
                    regime, "jacobi",
                    gpu_mem=gpu_mem,
                    level_memory=jac_level_mem,
                    grid_n=args.grid_n,
                    steps=args.steps,
                    seed=args.seed,
                    **run_kwargs,
                )
                regime_rows.append(row)
                all_rows.append(row)

        # ── DARTS TC variants ────────────────────────────────────────────────
        for tc_family, param_str, tc_config, mapper_cfg in tc_variants:
            if "cholesky" in workloads:
                row = safe_run(
                    _run_chol, tc_family, tc_family, param_str,
                    regime, "cholesky",
                    cfg=mapper_cfg,
                    gpu_mem=gpu_mem,
                    block_bytes=block_bytes,
                    n_blocks=args.n_blocks,
                    seed=args.seed,
                    top_k=args.top_k,
                    tc_kind=tc_config.kind,
                    tc_config=tc_config,
                )
                regime_rows.append(row)
                all_rows.append(row)

            if "jacobi" in workloads:
                row = safe_run(
                    _run_jac, tc_family, tc_family, param_str,
                    regime, "jacobi",
                    cfg=mapper_cfg,
                    gpu_mem=gpu_mem,
                    level_memory=jac_level_mem,
                    grid_n=args.grid_n,
                    steps=args.steps,
                    seed=args.seed,
                    top_k=args.top_k,
                    tc_kind=tc_config.kind,
                    tc_config=tc_config,
                )
                regime_rows.append(row)
                all_rows.append(row)

        # Per-workload tables within this regime
        for wl in workloads:
            wl_rows = [r for r in regime_rows if r.workload == wl]
            if wl_rows:
                print_table(wl_rows, f"{wl.capitalize()} [{regime}]")

    print_summary(all_rows)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
