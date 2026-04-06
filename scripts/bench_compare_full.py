"""
bench_compare_full.py
Full comparison suite: all Mappers × EvictionPolicies × TransitionConditions.

Runs three phases:
  1. All mappers × both eviction policies (auto transition)
  2. Key mappers × all transition kinds (lru eviction)
  3. Key mappers × all transition kinds (least_used_mapped eviction)

Produces a formatted summary report at the end.
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
    ALL_MAPPER_NAMES,
    EXTERNAL_MAPPER_NAMES,
    INTERNAL_MAPPER_NAMES,
    ExternalMapperConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    is_external_mapper,
    make_external_mapper,
    make_internal_mapper,
    make_transition_conditions,
    mapper_label,
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
# Benchmark parameters
# ---------------------------------------------------------------------------
GRID_N = 8
STEPS = 64
LEVEL_MEMORY_GB = 1.5
GPU_MEM_GB = 1.5       # tight memory to force evictions
SEED = 0
REPS = 1
TOP_K = 64

# Graph workload parameters (defaults from bench_eviction_jacobi)
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

ALL_EVICTION_POLICIES = ["lru", "least_used_mapped"]

# Mappers to test in phase 1 (reference mapper skipped: requires special graph attribute)
PHASE1_MAPPERS = [
    "dequeue_eft",
    "memory_aware_eft",
    "kahypar",
    "metis",           # skipped gracefully if not in build
    "darts",
    "block_cyclic",
    "row_cyclic",
    "col_cyclic",
    "checkerboard",
]

# Transition kinds to sweep in phases 2 & 3
ALL_TRANSITION_KINDS = ["planned", "default", "batch", "range", "hysteresis"]

# Mappers for transition-condition sweep (internal mappers only; external mappers ignore it)
PHASE2_MAPPERS = ["dequeue_eft", "memory_aware_eft", "darts"]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    phase: str
    mapper: str
    eviction_policy: str
    transition_kind: str
    status: str
    sim_s: Optional[float] = None
    wall_s: Optional[float] = None
    max_mem: Optional[int] = None
    total_mv: Optional[int] = None
    evict_mv: Optional[int] = None
    data_events: Optional[int] = None
    eviction_events: Optional[int] = None
    error: Optional[str] = None

    def label(self) -> str:
        return mapper_label(self.mapper)


def resolve_eviction_policy(name: str) -> EvictionPolicy:
    if name == "lru":
        return EvictionPolicy.LRU
    if name in {"least_used_mapped", "lum"}:
        return EvictionPolicy.LEAST_USED_MAPPED
    raise ValueError(f"Unknown eviction policy: {name!r}")


def run_single(
    *,
    mapper_name: str,
    eviction_policy_name: str,
    transition_kind: str,
    gpu_mem: int,
    level_memory: int,
    grid_n: int,
    steps: int,
    seed: int,
    reps: int,
    top_k: int,
    memory_aware_eft_config: MemoryAwareEFTConfig,
    external_mapper_config: ExternalMapperConfig,
) -> RunResult:
    eviction_policy = resolve_eviction_policy(eviction_policy_name)
    transition_config = TransitionConfig(kind=transition_kind)

    results_list = []
    for _ in range(reps):
        set_seed(seed)
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
            mapper_name=mapper_name,
            eviction_policy=eviction_policy,
            top_k_candidates=top_k,
            transition_config=transition_config,
        )
        result = run_once(
            sim_input,
            mapper_name=mapper_name,
            memory_aware_eft_alpha=memory_aware_eft_config.alpha,
            memory_aware_eft_config=memory_aware_eft_config,
            external_mapper_config=external_mapper_config,
        )
        results_list.append(result)

    last = results_list[-1]
    if last["status"] != ExecutionState.COMPLETE:
        return RunResult(
            phase="",
            mapper=mapper_name,
            eviction_policy=eviction_policy_name,
            transition_kind=transition_kind,
            status="FAILED",
            error=f"ExecutionState={last['status']}",
        )

    sim_times = [r["sim_us"] for r in results_list]
    return RunResult(
        phase="",
        mapper=mapper_name,
        eviction_policy=eviction_policy_name,
        transition_kind=transition_kind,
        status="OK",
        sim_s=statistics.mean(sim_times) / 1e6,
        wall_s=statistics.mean([r["wall_s"] for r in results_list]),
        max_mem=last["max_mem"],
        total_mv=last["total_mv"],
        evict_mv=last["evict_mv"],
        data_events=last["data_events"],
        eviction_events=last["eviction_events"],
    )


def safe_run(phase: str, **kwargs) -> RunResult:
    mapper_name = kwargs["mapper_name"]
    eviction_policy_name = kwargs["eviction_policy_name"]
    transition_kind = kwargs["transition_kind"]
    try:
        result = run_single(**kwargs)
        result.phase = phase
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  ERROR: {exc}", file=sys.stderr)
        return RunResult(
            phase=phase,
            mapper=mapper_name,
            eviction_policy=eviction_policy_name,
            transition_kind=transition_kind,
            status="ERROR",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def print_table(rows: list[RunResult], title: str) -> None:
    print()
    print(f"### {title}")
    header = (
        f"{'Mapper':<34}  {'Eviction':<18}  {'Transition':<16}  "
        f"{'Status':<6}  {'sim(s)':>8}  {'wall(s)':>8}  "
        f"{'max_mem':>10}  {'total_mv':>10}  {'evict_mv':>10}  "
        f"{'data_ev':>8}  {'evict_ev':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.status == "OK":
            sim_str = f"{r.sim_s:8.3f}"
            wall_str = f"{r.wall_s:8.3f}"
            mem_str = f"{_bytes_hr(r.max_mem):>10}"
            tmv_str = f"{_bytes_hr(r.total_mv):>10}"
            emv_str = f"{_bytes_hr(r.evict_mv):>10}"
            dev_str = f"{r.data_events:8d}"
            eev_str = f"{r.eviction_events:8d}"
        else:
            sim_str = f"{'--':>8}"
            wall_str = f"{'--':>8}"
            mem_str = f"{'--':>10}"
            tmv_str = f"{'--':>10}"
            emv_str = f"{'--':>10}"
            dev_str = f"{'--':>8}"
            eev_str = f"{'--':>8}"
        lbl = mapper_label(r.mapper)
        print(
            f"{lbl:<34}  {r.eviction_policy:<18}  {r.transition_kind:<16}  "
            f"{r.status:<6}  {sim_str}  {wall_str}  "
            f"{mem_str}  {tmv_str}  {emv_str}  "
            f"{dev_str}  {eev_str}"
        )


def print_markdown_report(
    phase1_results: list[RunResult],
    phase2_results: list[RunResult],
    phase3_results: list[RunResult],
    *,
    grid_n: int,
    steps: int,
    level_memory: int,
    gpu_mem: int,
) -> None:
    print()
    print("=" * 80)
    print("FULL COMPARISON REPORT")
    print("=" * 80)
    print(f"Grid       : {grid_n}×{grid_n}  ({grid_n*grid_n} cells)")
    print(f"Steps      : {steps}")
    print(f"Level mem  : {_bytes_hr(level_memory)}")
    print(f"GPU mem    : {_bytes_hr(gpu_mem)} per device  ({N_DEVICES-1} GPUs)")
    print(f"Seed       : {SEED}")
    print()

    print_table(phase1_results, "Phase 1 — All Mappers × Eviction Policies (auto transition)")
    print()
    ok1 = [r for r in phase1_results if r.status == "OK"]
    if ok1:
        best = min(ok1, key=lambda r: r.sim_s)
        worst = max(ok1, key=lambda r: r.sim_s)
        print(f"  Best  sim: {mapper_label(best.mapper)} / {best.eviction_policy}  → {best.sim_s:.3f}s")
        print(f"  Worst sim: {mapper_label(worst.mapper)} / {worst.eviction_policy}  → {worst.sim_s:.3f}s")

    print_table(phase2_results, "Phase 2 — Key Mappers × Transition Kinds (eviction=lru)")
    print_table(phase3_results, "Phase 3 — Key Mappers × Transition Kinds (eviction=least_used_mapped)")
    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Full mapper/eviction/transition comparison")
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--level-memory-gb", type=float, default=LEVEL_MEMORY_GB)
    parser.add_argument("--gpu-mem-gb", type=float, default=GPU_MEM_GB)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Run only the specified phase (default: all phases)"
    )
    parser.add_argument(
        "--mappers", default=None,
        help="Comma-separated override for Phase 1 mapper list"
    )
    parser.add_argument(
        "--eviction-policies", default=None,
        help="Comma-separated override for eviction policies"
    )
    args = parser.parse_args()

    grid_n = args.grid_n
    steps = args.steps
    level_memory = int(args.level_memory_gb * 1e9)
    gpu_mem = int(args.gpu_mem_gb * 1e9)
    seed = args.seed
    reps = args.reps
    top_k = args.top_k

    phase1_mappers = (
        [m.strip() for m in args.mappers.split(",") if m.strip()]
        if args.mappers
        else PHASE1_MAPPERS
    )
    eviction_policies = (
        [p.strip() for p in args.eviction_policies.split(",") if p.strip()]
        if args.eviction_policies
        else ALL_EVICTION_POLICIES
    )

    memory_aware_eft_config = MemoryAwareEFTConfig()
    external_mapper_config = ExternalMapperConfig()

    common_kwargs = dict(
        gpu_mem=gpu_mem,
        level_memory=level_memory,
        grid_n=grid_n,
        steps=steps,
        seed=seed,
        reps=reps,
        top_k=top_k,
        memory_aware_eft_config=memory_aware_eft_config,
        external_mapper_config=external_mapper_config,
    )

    run_phases = [1, 2, 3] if args.phase is None else [args.phase]

    print("=" * 80)
    print("Full Comparison Suite")
    print(f"  grid={grid_n}×{grid_n}  steps={steps}  level_mem={_bytes_hr(level_memory)}  gpu_mem={_bytes_hr(gpu_mem)}")
    print(f"  seed={seed}  reps={reps}  top_k={top_k}")
    print("=" * 80)

    phase1_results: list[RunResult] = []
    phase2_results: list[RunResult] = []
    phase3_results: list[RunResult] = []

    # -----------------------------------------------------------------------
    # Phase 1: All mappers × eviction policies (auto transition)
    # -----------------------------------------------------------------------
    if 1 in run_phases:
        print()
        print("--- Phase 1: All Mappers × Eviction Policies (auto transition) ---")
        for mapper_name in phase1_mappers:
            for eviction_policy_name in eviction_policies:
                lbl = mapper_label(mapper_name)
                print(f"\n  [{lbl} / {eviction_policy_name}]  transition=auto", flush=True)
                result = safe_run(
                    "phase1",
                    mapper_name=mapper_name,
                    eviction_policy_name=eviction_policy_name,
                    transition_kind="auto",
                    **common_kwargs,
                )
                if result.status == "OK":
                    print(
                        f"    sim={result.sim_s:.3f}s  wall={result.wall_s:.3f}s  "
                        f"max_mem={_bytes_hr(result.max_mem)}  "
                        f"total_mv={_bytes_hr(result.total_mv)}  "
                        f"evict_mv={_bytes_hr(result.evict_mv)}"
                    )
                else:
                    print(f"    FAILED: {result.error}")
                phase1_results.append(result)

    # -----------------------------------------------------------------------
    # Phase 2: Key mappers × all transition kinds (lru)
    # -----------------------------------------------------------------------
    if 2 in run_phases:
        print()
        print("--- Phase 2: Key Mappers × Transition Kinds (eviction=lru) ---")
        for mapper_name in PHASE2_MAPPERS:
            for transition_kind in ALL_TRANSITION_KINDS:
                lbl = mapper_label(mapper_name)
                print(f"\n  [{lbl}]  transition={transition_kind}  eviction=lru", flush=True)
                result = safe_run(
                    "phase2",
                    mapper_name=mapper_name,
                    eviction_policy_name="lru",
                    transition_kind=transition_kind,
                    **common_kwargs,
                )
                if result.status == "OK":
                    print(
                        f"    sim={result.sim_s:.3f}s  wall={result.wall_s:.3f}s  "
                        f"evict_mv={_bytes_hr(result.evict_mv)}"
                    )
                else:
                    print(f"    FAILED: {result.error}")
                phase2_results.append(result)

    # -----------------------------------------------------------------------
    # Phase 3: Key mappers × all transition kinds (least_used_mapped)
    # -----------------------------------------------------------------------
    if 3 in run_phases:
        print()
        print("--- Phase 3: Key Mappers × Transition Kinds (eviction=least_used_mapped) ---")
        for mapper_name in PHASE2_MAPPERS:
            for transition_kind in ALL_TRANSITION_KINDS:
                lbl = mapper_label(mapper_name)
                print(f"\n  [{lbl}]  transition={transition_kind}  eviction=least_used_mapped", flush=True)
                result = safe_run(
                    "phase3",
                    mapper_name=mapper_name,
                    eviction_policy_name="least_used_mapped",
                    transition_kind=transition_kind,
                    **common_kwargs,
                )
                if result.status == "OK":
                    print(
                        f"    sim={result.sim_s:.3f}s  wall={result.wall_s:.3f}s  "
                        f"evict_mv={_bytes_hr(result.evict_mv)}"
                    )
                else:
                    print(f"    FAILED: {result.error}")
                phase3_results.append(result)

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print_markdown_report(
        phase1_results,
        phase2_results,
        phase3_results,
        grid_n=grid_n,
        steps=steps,
        level_memory=level_memory,
        gpu_mem=gpu_mem,
    )


if __name__ == "__main__":
    main()
