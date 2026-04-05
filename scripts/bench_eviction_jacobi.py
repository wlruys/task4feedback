"""
bench_eviction_jacobi.py
Benchmark: configurable dynamic Jacobi stencil under fixed GPU memory.

Supports both internal C++ mappers and Python external mappers such as
`block_cyclic`, `row_cyclic`, `col_cyclic`, and `checkerboard`.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np
import torch

from typing import TYPE_CHECKING, Optional

from bench_mapper_support import (
    ALL_MAPPER_NAMES,
    EXTERNAL_MAPPER_NAMES,
    DARTSConfig,
    ExternalMapperConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    is_external_mapper,
    make_external_mapper,
    make_internal_mapper,
    make_transition_conditions,
    mapper_label,
)
from task4feedback.fastsim2 import (
    EvictionPolicy,
    ExecutionState,
    TaskNoise,
)
from task4feedback.graphs.base import TrajectoryWorkload
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiConfig, DynamicJacobiGraph
from task4feedback.graphs.mesh import build_geometry, generate_quad_mesh
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput, uniform_connected_devices

N_DEVICES = 5
GPU_MEM = 96_000_000_000
H2D_BW = 129_000_000_000
D2D_BW = 54_000_000_000
LATENCY = 1
TOP_K_CANDIDATES = 256

SYSTEM_SPECS = dict(
    fastest_flops=67_000_000_000_000,
    slowest_flops=67_000_000_000_000,
    gpu_flop=67_000_000_000_000,
    fastest_gmbw=3_350_000_000_000,
    slowest_gmbw=3_350_000_000_000,
)

DEFAULT_GRID_N = "16"
DEFAULT_STEPS = "256"
DEFAULT_LEVEL_MEMORY_GB = "120"
DEFAULT_SEED = 0
DEFAULT_N_SEEDS = 1
DEFAULT_REPS = 1
DEFAULT_MAPPER = "dequeue_eft"
DEFAULT_EVICTION_POLICY = "lru"
DEFAULT_COMPARE_MAPPERS = "dequeue_eft,memory_aware_eft,kahypar,metis,block_cyclic,row_cyclic"
DEFAULT_COMPARE_EVICTION_POLICIES = "lru,least_used_mapped"


def _bytes_hr(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_eviction_policy(name: str) -> EvictionPolicy:
    policy_name = name.lower()
    if policy_name == "lru":
        return EvictionPolicy.LRU
    if policy_name in {"least_used_mapped", "lum"}:
        return EvictionPolicy.LEAST_USED_MAPPED
    raise ValueError(f"Unsupported eviction policy '{name}'")


def _assign_fair_initial_locations(graph, gpu_locations: list[int]) -> None:
    cell_ids = list(range(len(graph.data.geometry.cells)))
    random.shuffle(cell_ids)

    locations = [0] * len(cell_ids)
    for index, cell_id in enumerate(cell_ids):
        locations[cell_id] = gpu_locations[index % len(gpu_locations)]

    graph.set_cell_locations(locations, step=0)


def build_system(mem: int = GPU_MEM):
    return uniform_connected_devices(
        n_devices=N_DEVICES,
        mem=mem,
        latency=LATENCY,
        h2d_bw=H2D_BW,
        d2d_bw=D2D_BW,
        h2d_links=1,
        d2d_links=1,
        cpu_copyengines=2,
        device_copyengines=2,
        system_specs=SYSTEM_SPECS,
    )


def build_graph(
    system,
    *,
    grid_n: int,
    steps: int,
    level_memory: int,
    domain_ratio: float,
    arithmetic_intensity: float,
    arithmetic_complexity: float,
    boundary_complexity: float,
    memory_intensity: float,
    boundary_width: float,
    r_interior: float,
    r_boundary: float,
    workload_scale: float,
    workload_lower_bound: float,
    workload_upper_bound: float,
    workload_phase_length: int,
    randomize_initial_placement: bool,
) -> DynamicJacobiGraph:
    mesh = generate_quad_mesh(L=1, n=grid_n)
    geometry = build_geometry(mesh)
    config = DynamicJacobiConfig(
        n=grid_n,
        steps=steps,
        level_memory=level_memory,
        domain_ratio=domain_ratio,
        arithmetic_intensity=arithmetic_intensity,
        arithmetic_complexity=arithmetic_complexity,
        boundary_complexity=boundary_complexity,
        memory_intensity=memory_intensity,
        boundary_width=boundary_width,
        r_interior=r_interior,
        r_boundary=r_boundary,
        vcu_usage=1.0,
        task_internal_memory=0,
        workload=TrajectoryWorkload(),
        workload_args=dict(
            traj_type="circle",
            lower_bound=workload_lower_bound,
            upper_bound=workload_upper_bound,
            scale=workload_scale,
            traj_specifics=dict(radius=0.3, phase_length=workload_phase_length),
        ),
    )

    graph = DynamicJacobiGraph(geometry, config, system=system)
    if randomize_initial_placement:
        _assign_fair_initial_locations(graph, list(range(1, N_DEVICES)))
    return graph


def make_sim_input(
    graph,
    system,
    *,
    mapper_name: str,
    eviction_policy: EvictionPolicy,
    top_k_candidates: int,
    transition_config: TransitionConfig,
):
    return SimulatorInput(
        graph,
        graph.get_blocks(),
        system,
        task_noise=TaskNoise(graph.static_graph),
        transition_conditions=make_transition_conditions(
            mapper_name,
            top_k_candidates=top_k_candidates,
            config=transition_config,
        ),
        top_k_candidates=top_k_candidates,
        eviction_policy=eviction_policy,
    )


def run_once(
    sim_input: SimulatorInput,
    *,
    mapper_name: str,
    memory_aware_eft_alpha: float = 1.0,
    memory_aware_eft_config: Optional["MemoryAwareEFTConfig"] = None,
    external_mapper_config: Optional["ExternalMapperConfig"] = None,
    darts_config: Optional["DARTSConfig"] = None,
):
    if is_external_mapper(mapper_name):
        from bench_mapper_support import ExternalMapperConfig as _EMC
        driver = SimulatorDriver(sim_input)
        external_mapper = make_external_mapper(
            mapper_name,
            sim_input.graph,
            n_gpu_devices=N_DEVICES - 1,
            config=external_mapper_config or _EMC(),
        )
        driver.initialize()
        driver.initialize_data()
        driver.enable_external_mapper(external_mapper)
    else:
        internal_mapper = make_internal_mapper(
            mapper_name,
            memory_aware_eft_alpha=memory_aware_eft_alpha,
            memory_aware_eft_config=memory_aware_eft_config,
            darts_config=darts_config,
        )
        driver = SimulatorDriver(sim_input, internal_mapper=internal_mapper)
        driver.initialize()
        driver.initialize_data()
        driver.disable_external_mapper()

    t0 = time.perf_counter()
    status = driver.run()
    wall_s = time.perf_counter() - t0

    total_mv = sum(driver.total_data_movement())
    evict_mv = sum(driver.total_eviction_movement())
    task_runtime = driver.state.get_task_runtime()
    n_data_events = task_runtime.get_n_data_tasks()
    n_data_movement_events = sum(
        1 for i in range(n_data_events) if not task_runtime.is_data_task_virtual(i)
    )
    n_eviction_events = task_runtime.get_n_eviction_tasks()
    n_eviction_movement_events = sum(
        1 for i in range(n_eviction_events) if not task_runtime.is_eviction_task_virtual(i)
    )
    return {
        "status": status,
        "wall_s": wall_s,
        "sim_us": driver.time,
        "max_mem": driver.max_mem_usage,
        "total_mv": total_mv,
        "evict_mv": evict_mv,
        "data_events": n_data_events,
        "data_movement_events": n_data_movement_events,
        "eviction_events": n_eviction_events,
        "eviction_movement_events": n_eviction_movement_events,
    }


def print_problem_summary(*, grid_n: int, steps: int, level_memory: int, gpu_mem: int) -> None:
    cells = grid_n * grid_n
    total_data_bytes = int(level_memory * (steps + 1))
    aggregate_gpu_mem = gpu_mem * (N_DEVICES - 1)
    print(
        f"  problem: grid={grid_n}x{grid_n}  cells={cells}  steps={steps}  "
        f"level_memory={_bytes_hr(level_memory)}"
    )
    print(
        f"  pressure: total_levels={steps + 1}  total_level_data~={_bytes_hr(total_data_bytes)}  "
        f"aggregate_gpu_mem={_bytes_hr(aggregate_gpu_mem)}"
    )


def run_scenario(
    *,
    seed: int,
    grid_n: int,
    steps: int,
    level_memory: int,
    gpu_mem: int,
    mapper_name: str,
    eviction_policy_name: str,
    reps: int,
    top_k_candidates: int,
    randomize_initial_placement: bool,
    memory_aware_eft_alpha: float,
    memory_aware_eft_config: MemoryAwareEFTConfig,
    external_mapper_config: ExternalMapperConfig,
    transition_config: TransitionConfig,
    domain_ratio: float,
    arithmetic_intensity: float,
    arithmetic_complexity: float,
    boundary_complexity: float,
    memory_intensity: float,
    boundary_width: float,
    r_interior: float,
    r_boundary: float,
    workload_scale: float,
    workload_lower_bound: float,
    workload_upper_bound: float,
    workload_phase_length: int,
):
    print()
    print(
        f"[seed={seed}] mapper={mapper_label(mapper_name, memory_aware_eft_alpha=memory_aware_eft_alpha, memory_aware_eft_config=memory_aware_eft_config)} "
        f"eviction={eviction_policy_name} gpu_mem={_bytes_hr(gpu_mem)}"
    )
    print_problem_summary(
        grid_n=grid_n,
        steps=steps,
        level_memory=level_memory,
        gpu_mem=gpu_mem,
    )

    run_results = []
    eviction_policy = resolve_eviction_policy(eviction_policy_name)
    for rep in range(reps):
        set_seed(seed)
        graph_system = build_system(mem=gpu_mem)
        graph = build_graph(
            graph_system,
            grid_n=grid_n,
            steps=steps,
            level_memory=level_memory,
            domain_ratio=domain_ratio,
            arithmetic_intensity=arithmetic_intensity,
            arithmetic_complexity=arithmetic_complexity,
            boundary_complexity=boundary_complexity,
            memory_intensity=memory_intensity,
            boundary_width=boundary_width,
            r_interior=r_interior,
            r_boundary=r_boundary,
            workload_scale=workload_scale,
            workload_lower_bound=workload_lower_bound,
            workload_upper_bound=workload_upper_bound,
            workload_phase_length=workload_phase_length,
            randomize_initial_placement=randomize_initial_placement,
        )
        sim_system = build_system(mem=gpu_mem)
        sim_input = make_sim_input(
            graph,
            sim_system,
            mapper_name=mapper_name,
            eviction_policy=eviction_policy,
            top_k_candidates=top_k_candidates,
            transition_config=transition_config,
        )
        result = run_once(
            sim_input,
            mapper_name=mapper_name,
            memory_aware_eft_alpha=memory_aware_eft_alpha,
            memory_aware_eft_config=memory_aware_eft_config,
            external_mapper_config=external_mapper_config,
        )
        if result["status"] != ExecutionState.COMPLETE:
            print(f"  rep {rep + 1}: FAILED status={result['status']}")
            return None
        run_results.append(result)
        print(
            f"  rep {rep + 1}/{reps}: wall={result['wall_s']:.3f}s  "
            f"sim={result['sim_us'] / 1e6:.3f}s  "
            f"max_mem={_bytes_hr(result['max_mem'])}  "
            f"total_mv={_bytes_hr(result['total_mv'])}  "
            f"evict_mv={_bytes_hr(result['evict_mv'])}  "
            f"data_events={result['data_events']} "
            f"(movement={result['data_movement_events']})  "
            f"eviction_events={result['eviction_events']} "
            f"(movement={result['eviction_movement_events']})"
        )

    wall_times = [row["wall_s"] for row in run_results]
    sim_times = [row["sim_us"] for row in run_results]
    total_mvs = [row["total_mv"] for row in run_results]
    evict_mvs = [row["evict_mv"] for row in run_results]
    final = run_results[-1]
    print(
        f"  => best_wall={min(wall_times):.3f}s  "
        f"avg_wall={statistics.mean(wall_times):.3f}s  "
        f"avg_sim={statistics.mean(sim_times) / 1e6:.3f}s  "
        f"avg_total_mv={_bytes_hr(statistics.mean(total_mvs))}  "
        f"avg_evict_mv={_bytes_hr(statistics.mean(evict_mvs))}"
    )
    print(
        f"     data_events={final['data_events']} (movement={final['data_movement_events']})  "
        f"eviction_events={final['eviction_events']} "
        f"(movement={final['eviction_movement_events']})"
    )
    return run_results


def compare_scenarios(
    *,
    seed_start: int,
    n_seeds: int,
    reps: int,
    grid_sizes: list[int],
    step_counts: list[int],
    level_memories: list[int],
    mapper_names: list[str],
    eviction_policy_names: list[str],
    gpu_mem: int,
    top_k_candidates: int,
    randomize_initial_placement: bool,
    memory_aware_eft_alpha: float,
    memory_aware_eft_config: MemoryAwareEFTConfig,
    external_mapper_config: ExternalMapperConfig,
    transition_config: TransitionConfig,
    domain_ratio: float,
    arithmetic_intensity: float,
    arithmetic_complexity: float,
    boundary_complexity: float,
    memory_intensity: float,
    boundary_width: float,
    r_interior: float,
    r_boundary: float,
    workload_scale: float,
    workload_lower_bound: float,
    workload_upper_bound: float,
    workload_phase_length: int,
) -> None:
    print("=" * 72)
    print("Comparison Mode")
    print(f"Mappers         : {mapper_names}")
    print(f"Eviction policy : {eviction_policy_names}")
    print(f"Grid sweep      : {grid_sizes}")
    print(f"Step sweep      : {step_counts}")
    print(f"Level sweep     : {[_bytes_hr(v) for v in level_memories]}")
    print("=" * 72)

    for grid_n in grid_sizes:
        for steps in step_counts:
            for level_memory in level_memories:
                print()
                print(
                    f"Scenario grid={grid_n} steps={steps} level_memory={_bytes_hr(level_memory)}"
                )
                summary_rows = []
                for mapper_name in mapper_names:
                    for eviction_policy_name in eviction_policy_names:
                        sim_times = []
                        total_mvs = []
                        evict_mvs = []
                        for seed_offset in range(n_seeds):
                            seed = seed_start + seed_offset
                            results = run_scenario(
                                seed=seed,
                                grid_n=grid_n,
                                steps=steps,
                                level_memory=level_memory,
                                gpu_mem=gpu_mem,
                                mapper_name=mapper_name,
                                eviction_policy_name=eviction_policy_name,
                                reps=reps,
                                top_k_candidates=top_k_candidates,
                                randomize_initial_placement=randomize_initial_placement,
                                memory_aware_eft_alpha=memory_aware_eft_alpha,
                                memory_aware_eft_config=memory_aware_eft_config,
                                external_mapper_config=external_mapper_config,
                                transition_config=transition_config,
                                domain_ratio=domain_ratio,
                                arithmetic_intensity=arithmetic_intensity,
                                arithmetic_complexity=arithmetic_complexity,
                                boundary_complexity=boundary_complexity,
                                memory_intensity=memory_intensity,
                                boundary_width=boundary_width,
                                r_interior=r_interior,
                                r_boundary=r_boundary,
                                workload_scale=workload_scale,
                                workload_lower_bound=workload_lower_bound,
                                workload_upper_bound=workload_upper_bound,
                                workload_phase_length=workload_phase_length,
                            )
                            if results is None:
                                continue
                            for result in results:
                                sim_times.append(result["sim_us"])
                                total_mvs.append(result["total_mv"])
                                evict_mvs.append(result["evict_mv"])
                        if not sim_times:
                            continue
                        summary_rows.append(
                            {
                                "mapper_label": mapper_label(
                                    mapper_name,
                                    memory_aware_eft_alpha=memory_aware_eft_alpha,
                                    memory_aware_eft_config=memory_aware_eft_config,
                                ),
                                "eviction_policy_name": eviction_policy_name,
                                "avg_sim_us": statistics.mean(sim_times),
                                "avg_total_mv": statistics.mean(total_mvs),
                                "avg_evict_mv": statistics.mean(evict_mvs),
                            }
                        )

                summary_rows.sort(key=lambda row: row["avg_sim_us"])
                for row in summary_rows:
                    print(
                        f"  {row['mapper_label']:>28}  {row['eviction_policy_name']:<18}  "
                        f"sim={row['avg_sim_us'] / 1e6:8.3f}s  "
                        f"total_mv={_bytes_hr(row['avg_total_mv']):>10}  "
                        f"evict_mv={_bytes_hr(row['avg_evict_mv']):>10}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jacobi eviction stress benchmark")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS, help="Number of seeds")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Repetitions per seed")
    parser.add_argument("--grid-n", default=DEFAULT_GRID_N, help="Comma-separated grid sizes")
    parser.add_argument("--steps", default=DEFAULT_STEPS, help="Comma-separated step counts")
    parser.add_argument(
        "--level-memory-gb",
        default=DEFAULT_LEVEL_MEMORY_GB,
        help="Comma-separated per-level memory sizes in decimal GB",
    )
    parser.add_argument(
        "--gpu-mem-gb",
        type=float,
        default=GPU_MEM / 1e9,
        help="Per-GPU memory capacity in decimal GB",
    )
    parser.add_argument(
        "--mapper",
        choices=ALL_MAPPER_NAMES,
        default=DEFAULT_MAPPER,
        help="Mapper to benchmark",
    )
    parser.add_argument(
        "--eviction-policy",
        choices=("lru", "least_used_mapped"),
        default="lru",
        help="Eviction policy",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=TOP_K_CANDIDATES,
        help="Top-k candidate window passed to the mapper",
    )
    parser.add_argument(
        "--no-randomize-locations",
        action="store_true",
        help="Keep all initial data on CPU instead of randomizing over GPUs",
    )
    parser.add_argument(
        "--memory-aware-eft-alpha",
        type=float,
        default=1.0,
        help="Alpha multiplier for MemoryAwareEFTMapper eviction-cost penalty",
    )
    parser.add_argument(
        "--memory-aware-location-state",
        choices=("launched", "reserved", "mapped"),
        default="launched",
        help="Location state used to estimate MemoryAwareEFT eviction write-back cost",
    )
    parser.add_argument(
        "--memory-aware-overflow-state",
        choices=("reserved", "mapped", "launched"),
        default="launched",
        help="Device memory state used by MemoryAwareEFT overflow estimation",
    )
    parser.add_argument(
        "--memory-aware-overflow-mode",
        choices=("full_spill", "incoming_only"),
        default="full_spill",
        help="Whether MemoryAwareEFT clears the full spill or only enough room for the incoming task",
    )
    parser.add_argument("--compare", action="store_true", help="Run a comparison sweep")
    parser.add_argument(
        "--compare-mappers",
        default=DEFAULT_COMPARE_MAPPERS,
        help="Comma-separated mapper list for --compare",
    )
    parser.add_argument(
        "--compare-eviction-policies",
        default=DEFAULT_COMPARE_EVICTION_POLICIES,
        help="Comma-separated eviction-policy list for --compare",
    )
    parser.add_argument("--domain-ratio", type=float, default=1.0)
    parser.add_argument("--arithmetic-intensity", type=float, default=595.5555555)
    parser.add_argument("--arithmetic-complexity", type=float, default=1.0)
    parser.add_argument("--boundary-complexity", type=float, default=1.0)
    parser.add_argument("--memory-intensity", type=float, default=0.0)
    parser.add_argument("--boundary-width", type=float, default=0.25)
    parser.add_argument("--r-interior", type=float, default=10.0)
    parser.add_argument("--r-boundary", type=float, default=0.1)
    parser.add_argument("--workload-scale", type=float, default=0.3)
    parser.add_argument("--workload-lower-bound", type=float, default=1.0)
    parser.add_argument("--workload-upper-bound", type=float, default=5.0)
    parser.add_argument("--workload-phase-length", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=2, help="External block-cyclic row block size")
    parser.add_argument("--block-cols", type=int, default=2, help="External block-cyclic col block size")
    parser.add_argument("--processor-rows", type=int, default=None, help="External block-cyclic processor rows")
    parser.add_argument("--processor-cols", type=int, default=None, help="External block-cyclic processor cols")
    parser.add_argument(
        "--transition-kind",
        choices=("auto", "default", "batch", "device_threshold", "range", "hysteresis"),
        default="auto",
        help="Transition-condition mode. 'auto' preserves the current mapper-based default.",
    )
    parser.add_argument("--transition-batch-size", type=int, default=5)
    parser.add_argument("--transition-queue-threshold", type=int, default=5)
    parser.add_argument(
        "--transition-max-in-flight",
        type=int,
        default=None,
        help="Override BatchTransitionConditions.max_in_flight; defaults to top-k when omitted.",
    )
    parser.add_argument("--transition-mapped-threshold", type=int, default=0)
    parser.add_argument("--transition-reserved-threshold", type=int, default=-1)
    parser.add_argument("--transition-mapped-reserved-gap", type=int, default=5)
    parser.add_argument("--transition-reserved-launched-gap", type=int, default=5)
    parser.add_argument(
        "--transition-total-in-flight",
        type=int,
        default=None,
        help="Override RangeTransitionConditions.total_in_flight; defaults to top-k when omitted.",
    )
    parser.add_argument("--transition-hysteresis-open", type=int, default=16)
    parser.add_argument("--transition-hysteresis-close", type=int, default=36)
    parser.add_argument("--transition-hysteresis-starvation", type=int, default=2)
    args = parser.parse_args()

    grid_sizes = _parse_int_list(args.grid_n)
    step_counts = _parse_int_list(args.steps)
    level_memories = [int(value * 1e9) for value in _parse_float_list(args.level_memory_gb)]
    gpu_mem = int(args.gpu_mem_gb * 1e9)
    external_mapper_config = ExternalMapperConfig(
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        processor_rows=args.processor_rows,
        processor_cols=args.processor_cols,
    )
    transition_config = TransitionConfig(
        kind=args.transition_kind,
        batch_size=args.transition_batch_size,
        queue_threshold=args.transition_queue_threshold,
        max_in_flight=args.transition_max_in_flight,
        mapped_threshold=args.transition_mapped_threshold,
        reserved_threshold=args.transition_reserved_threshold,
        mapped_reserved_gap=args.transition_mapped_reserved_gap,
        reserved_launched_gap=args.transition_reserved_launched_gap,
        total_in_flight=args.transition_total_in_flight,
        hysteresis_open=args.transition_hysteresis_open,
        hysteresis_close=args.transition_hysteresis_close,
        hysteresis_starvation=args.transition_hysteresis_starvation,
    )
    memory_aware_eft_config = MemoryAwareEFTConfig(
        alpha=args.memory_aware_eft_alpha,
        eviction_cost_location_state=args.memory_aware_location_state,
        overflow_state=args.memory_aware_overflow_state,
        overflow_mode=args.memory_aware_overflow_mode,
    )

    print("=" * 72)
    print("Jacobi Eviction Benchmark")
    print(f"GPUs            : {N_DEVICES - 1}")
    print(f"GPU memory      : {_bytes_hr(gpu_mem)} per GPU")
    print(
        "Mapper          : "
        f"{mapper_label(args.mapper, memory_aware_eft_alpha=args.memory_aware_eft_alpha, memory_aware_eft_config=memory_aware_eft_config)}"
    )
    print(f"Eviction policy : {args.eviction_policy}")
    print(f"Seeds           : {args.seed}..{args.seed + args.n_seeds - 1}")
    print(f"Repetitions     : {args.reps}")
    print(f"Grid sweep      : {grid_sizes}")
    print(f"Step sweep      : {step_counts}")
    print(f"Level sweep     : {[_bytes_hr(v) for v in level_memories]}")
    print(f"Transition kind : {args.transition_kind}")
    if args.mapper in EXTERNAL_MAPPER_NAMES:
        print(
            f"External mapper : block_rows={args.block_rows}  block_cols={args.block_cols}  "
            f"processor_rows={args.processor_rows}  processor_cols={args.processor_cols}"
        )
    print("=" * 72)

    if args.compare:
        compare_scenarios(
            seed_start=args.seed,
            n_seeds=args.n_seeds,
            reps=args.reps,
            grid_sizes=grid_sizes,
            step_counts=step_counts,
            level_memories=level_memories,
            mapper_names=_parse_str_list(args.compare_mappers),
            eviction_policy_names=_parse_str_list(args.compare_eviction_policies),
            gpu_mem=gpu_mem,
            top_k_candidates=args.top_k_candidates,
            randomize_initial_placement=not args.no_randomize_locations,
            memory_aware_eft_alpha=args.memory_aware_eft_alpha,
            memory_aware_eft_config=memory_aware_eft_config,
            external_mapper_config=external_mapper_config,
            transition_config=transition_config,
            domain_ratio=args.domain_ratio,
            arithmetic_intensity=args.arithmetic_intensity,
            arithmetic_complexity=args.arithmetic_complexity,
            boundary_complexity=args.boundary_complexity,
            memory_intensity=args.memory_intensity,
            boundary_width=args.boundary_width,
            r_interior=args.r_interior,
            r_boundary=args.r_boundary,
            workload_scale=args.workload_scale,
            workload_lower_bound=args.workload_lower_bound,
            workload_upper_bound=args.workload_upper_bound,
            workload_phase_length=args.workload_phase_length,
        )
        print("=" * 72)
        return

    for seed_offset in range(args.n_seeds):
        seed = args.seed + seed_offset
        for grid_n in grid_sizes:
            for steps in step_counts:
                for level_memory in level_memories:
                    run_scenario(
                        seed=seed,
                        grid_n=grid_n,
                        steps=steps,
                        level_memory=level_memory,
                        gpu_mem=gpu_mem,
                        mapper_name=args.mapper,
                        eviction_policy_name=args.eviction_policy,
                        reps=args.reps,
                        top_k_candidates=args.top_k_candidates,
                        randomize_initial_placement=not args.no_randomize_locations,
                        memory_aware_eft_alpha=args.memory_aware_eft_alpha,
                        memory_aware_eft_config=memory_aware_eft_config,
                        external_mapper_config=external_mapper_config,
                        transition_config=transition_config,
                        domain_ratio=args.domain_ratio,
                        arithmetic_intensity=args.arithmetic_intensity,
                        arithmetic_complexity=args.arithmetic_complexity,
                        boundary_complexity=args.boundary_complexity,
                        memory_intensity=args.memory_intensity,
                        boundary_width=args.boundary_width,
                        r_interior=args.r_interior,
                        r_boundary=args.r_boundary,
                        workload_scale=args.workload_scale,
                        workload_lower_bound=args.workload_lower_bound,
                        workload_upper_bound=args.workload_upper_bound,
                        workload_phase_length=args.workload_phase_length,
                    )

    print("=" * 72)


if __name__ == "__main__":
    main()
