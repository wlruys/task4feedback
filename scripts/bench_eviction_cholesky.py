"""
bench_eviction_cholesky.py
Benchmark: block Cholesky factorization under fixed GPU memory.

This benchmark exposes two primary scaling knobs:
- the number of matrix blocks along one dimension (`n_blocks`)
- the bytes per lower-triangular matrix block (`block_bytes`)

The underlying Cholesky graph sizes its data by total lower-triangular memory, so
this script converts `(n_blocks, block_bytes)` into `level_memory` and reports
derived footprint estimates before each run.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

import numpy as np
import torch

from bench_mapper_support import (
    ALL_MAPPER_NAMES,
    EXTERNAL_MAPPER_NAMES,
    ExternalMapperConfig,
    TransitionConfig,
    is_external_mapper,
    make_external_mapper,
    make_internal_mapper,
    make_transition_conditions,
    mapper_label,
)
from task4feedback.graphs.cholesky import CholeskyConfig, CholeskyGraph
from task4feedback.graphs.mesh import build_geometry, generate_quad_mesh
from task4feedback.interface.wrappers import (
    SimulatorDriver,
    SimulatorInput,
    uniform_connected_devices,
)
from task4feedback.fastsim2 import (
    ExecutionState,
    EvictionPolicy,
    TaskNoise,
)

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

DEFAULT_BLOCKS = "16,24,32"
DEFAULT_BLOCK_BYTES_GB = "0.5,1.0"
DEFAULT_SEED = 0
DEFAULT_N_SEEDS = 1
DEFAULT_REPS = 1
DEFAULT_MAPPER = "dequeue_eft"
DEFAULT_EVICTION_POLICY = "lru"
DEFAULT_COMPARE_MAPPERS = "dequeue_eft,memory_aware_eft,metis,darts,darts_extended,block_cyclic,row_cyclic"
DEFAULT_COMPARE_EVICTION_POLICIES = "lru,least_used_mapped"
DEFAULT_PRESSURE_PRESETS = "light,moderate,heavy"
DEFAULT_MEMORY_AWARE_EFT_ALPHA = 1.0

PRESSURE_PRESET_RATIOS = {
    "light": 0.60,
    "moderate": 1.00,
    "heavy": 1.40,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def lower_triangular_block_count(n_blocks: int) -> int:
    return n_blocks * (n_blocks + 1) // 2


def align_block_bytes(block_bytes: int, bytes_per_element: int) -> int:
    if block_bytes < bytes_per_element:
        return bytes_per_element
    remainder = block_bytes % bytes_per_element
    if remainder == 0:
        return block_bytes
    return block_bytes - remainder


def estimate_footprint(
    n_blocks: int,
    block_bytes: int,
    gpu_mem: int,
    n_gpu_devices: int,
) -> dict[str, float]:
    total_blocks = lower_triangular_block_count(n_blocks)
    total_data_bytes = total_blocks * block_bytes
    max_task_data_bytes = 3 * block_bytes
    blocks_per_gpu_capacity = gpu_mem // block_bytes
    aggregate_gpu_mem = n_gpu_devices * gpu_mem
    mean_bytes_per_gpu = total_data_bytes / n_gpu_devices
    return {
        "total_blocks": total_blocks,
        "total_data_bytes": total_data_bytes,
        "max_task_data_bytes": max_task_data_bytes,
        "blocks_per_gpu_capacity": blocks_per_gpu_capacity,
        "aggregate_gpu_mem": aggregate_gpu_mem,
        "mean_bytes_per_gpu": mean_bytes_per_gpu,
        "overcommit_ratio": total_data_bytes / aggregate_gpu_mem,
    }


def _assign_fair_initial_locations(graph, gpu_locations: list[int]) -> None:
    lower_triangular_cells = [
        cell_id for cell_id, (i, j) in graph.data.cell_to_ij.items() if i >= j
    ]
    random.shuffle(lower_triangular_cells)

    location_dict = {}
    for index, cell_id in enumerate(lower_triangular_cells):
        location_dict[cell_id] = gpu_locations[index % len(gpu_locations)]

    graph.set_cell_locations_from_dict(location_dict)


def derive_block_bytes_for_pressure(
    *,
    n_blocks: int,
    gpu_mem: int,
    n_gpu_devices: int,
    target_overcommit_ratio: float,
    bytes_per_element: int = 4,
) -> int:
    total_blocks = lower_triangular_block_count(n_blocks)
    aggregate_gpu_mem = n_gpu_devices * gpu_mem
    total_bytes = int(target_overcommit_ratio * aggregate_gpu_mem)
    raw_block_bytes = max(total_bytes // total_blocks, bytes_per_element)
    return align_block_bytes(raw_block_bytes, bytes_per_element)


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
    n_blocks: int,
    block_bytes: int,
    *,
    randomize_initial_placement: bool,
    verbose: bool,
) -> CholeskyGraph:
    mesh = generate_quad_mesh(L=1, n=n_blocks)
    geometry = build_geometry(mesh)
    level_memory = lower_triangular_block_count(n_blocks) * block_bytes
    config = CholeskyConfig(
        n=n_blocks,
        level_memory=level_memory,
        bytes_per_element=4,
        arithmetic_intensity=1.0,
        arithmetic_complexity=1.0,
        memory_intensity=1.0,
        task_internal_memory=0,
        vcu_usage=1.0,
        verbose=verbose,
        morton_priority_enabled=False,
    )
    graph = CholeskyGraph(geometry, config, system=system)
    if randomize_initial_placement:
        _assign_fair_initial_locations(graph, list(range(1, N_DEVICES)))
    return graph


def resolve_eviction_policy(name: str) -> EvictionPolicy:
    policy_name = name.lower()
    if policy_name == "lru":
        return EvictionPolicy.LRU
    if policy_name in {"least_used_mapped", "lum"}:
        return EvictionPolicy.LEAST_USED_MAPPED
    raise ValueError(f"Unsupported eviction policy '{name}'")


def make_sim_input(
    graph: CholeskyGraph,
    system,
    mapper_name: str,
    eviction_policy: EvictionPolicy,
    transition_config: TransitionConfig,
):
    blocks = graph.get_blocks()
    task_noise = TaskNoise(graph.static_graph)
    return SimulatorInput(
        graph,
        blocks,
        system,
        task_noise=task_noise,
        transition_conditions=make_transition_conditions(
            mapper_name,
            top_k_candidates=TOP_K_CANDIDATES,
            config=transition_config,
        ),
        top_k_candidates=TOP_K_CANDIDATES,
        eviction_policy=eviction_policy,
    )


def run_once(
    sim_input: SimulatorInput,
    *,
    mapper_name: str,
    memory_aware_eft_alpha: float,
    external_mapper_config: ExternalMapperConfig,
):
    if is_external_mapper(mapper_name):
        driver = SimulatorDriver(sim_input)
        external_mapper = make_external_mapper(
            mapper_name,
            sim_input.graph,
            n_gpu_devices=N_DEVICES - 1,
            config=external_mapper_config,
        )
        driver.initialize()
        driver.initialize_data()
        driver.enable_external_mapper(external_mapper)
    else:
        driver = SimulatorDriver(
            sim_input,
            internal_mapper=make_internal_mapper(
                mapper_name,
                memory_aware_eft_alpha=memory_aware_eft_alpha,
            ),
        )
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


def print_problem_summary(n_blocks: int, block_bytes: int, gpu_mem: int) -> None:
    estimates = estimate_footprint(
        n_blocks=n_blocks,
        block_bytes=block_bytes,
        gpu_mem=gpu_mem,
        n_gpu_devices=N_DEVICES - 1,
    )
    print(
        f"  problem: n_blocks={n_blocks}  lower_tri_blocks={estimates['total_blocks']}  "
        f"block_bytes={_bytes_hr(block_bytes)}"
    )
    print(
        f"  matrix: total_data={_bytes_hr(estimates['total_data_bytes'])}  "
        f"mean_per_gpu_if_balanced={_bytes_hr(estimates['mean_bytes_per_gpu'])}  "
        f"aggregate_gpu_mem={_bytes_hr(estimates['aggregate_gpu_mem'])}"
    )
    print(
        f"  task_ws: max_unique_blocks=3  max_task_data={_bytes_hr(estimates['max_task_data_bytes'])}  "
        f"gpu_block_capacity~={int(estimates['blocks_per_gpu_capacity'])}"
    )
    print(
        f"  pressure: overcommit_ratio={estimates['overcommit_ratio']:.3f}x  "
        f"single_gpu_capacity={_bytes_hr(gpu_mem)}"
    )


def run_scenario(
    *,
    seed: int,
    n_blocks: int,
    block_bytes: int,
    gpu_mem: int,
    mapper_name: str,
    eviction_policy_name: str,
    reps: int,
    randomize_initial_placement: bool,
    verbose_graph: bool,
    memory_aware_eft_alpha: float,
    external_mapper_config: ExternalMapperConfig,
    transition_config: TransitionConfig,
):
    eviction_policy = resolve_eviction_policy(eviction_policy_name)
    print()
    print(
        f"[seed={seed}] mapper={mapper_label(mapper_name, memory_aware_eft_alpha=memory_aware_eft_alpha)} eviction={eviction_policy_name} "
        f"gpu_mem={_bytes_hr(gpu_mem)}"
    )
    print_problem_summary(n_blocks, block_bytes, gpu_mem)

    run_results = []
    for rep in range(reps):
        set_seed(seed)
        graph_system = build_system(mem=gpu_mem)
        graph = build_graph(
            graph_system,
            n_blocks=n_blocks,
            block_bytes=block_bytes,
            randomize_initial_placement=randomize_initial_placement,
            verbose=verbose_graph and rep == 0,
        )
        sim_system = build_system(mem=gpu_mem)
        sim_input = make_sim_input(
            graph,
            sim_system,
            mapper_name,
            eviction_policy,
            transition_config,
        )
        result = run_once(
            sim_input,
            mapper_name=mapper_name,
            memory_aware_eft_alpha=memory_aware_eft_alpha,
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
            f"evict_events={result['eviction_events']} "
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


def build_explicit_scenarios(n_blocks_list: list[int], block_bytes_list: list[int]) -> list[dict[str, int | str]]:
    scenarios: list[dict[str, int | str]] = []
    for n_blocks in n_blocks_list:
        for block_bytes in block_bytes_list:
            scenarios.append(
                {
                    "label": f"n{n_blocks}_block{_bytes_hr(block_bytes)}",
                    "n_blocks": n_blocks,
                    "block_bytes": block_bytes,
                }
            )
    return scenarios


def build_pressure_scenarios(n_blocks_list: list[int], preset_names: list[str], gpu_mem: int) -> list[dict[str, int | str]]:
    scenarios: list[dict[str, int | str]] = []
    for n_blocks in n_blocks_list:
        for preset_name in preset_names:
            if preset_name not in PRESSURE_PRESET_RATIOS:
                raise ValueError(f"Unsupported pressure preset '{preset_name}'")
            ratio = PRESSURE_PRESET_RATIOS[preset_name]
            block_bytes = derive_block_bytes_for_pressure(
                n_blocks=n_blocks,
                gpu_mem=gpu_mem,
                n_gpu_devices=N_DEVICES - 1,
                target_overcommit_ratio=ratio,
            )
            scenarios.append(
                {
                    "label": f"{preset_name}_n{n_blocks}",
                    "n_blocks": n_blocks,
                    "block_bytes": block_bytes,
                }
            )
    return scenarios


def compare_scenarios(
    *,
    seed_start: int,
    n_seeds: int,
    reps: int,
    scenarios: list[dict[str, int | str]],
    mapper_names: list[str],
    eviction_policy_names: list[str],
    gpu_mem: int,
    randomize_initial_placement: bool,
    verbose_graph: bool,
    memory_aware_eft_alpha: float,
    external_mapper_config: ExternalMapperConfig,
    transition_config: TransitionConfig,
) -> None:
    print("=" * 72)
    print("Comparison Mode")
    print(f"Mappers         : {mapper_names}")
    print(f"Eviction policy : {eviction_policy_names}")
    print(f"Scenarios       : {[scenario['label'] for scenario in scenarios]}")
    print("=" * 72)

    for scenario in scenarios:
        print()
        print(f"Scenario {scenario['label']}")
        print_problem_summary(int(scenario["n_blocks"]), int(scenario["block_bytes"]), gpu_mem)
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
                        n_blocks=int(scenario["n_blocks"]),
                        block_bytes=int(scenario["block_bytes"]),
                        gpu_mem=gpu_mem,
                        mapper_name=mapper_name,
                        eviction_policy_name=eviction_policy_name,
                        reps=reps,
                        randomize_initial_placement=randomize_initial_placement,
                        verbose_graph=verbose_graph,
                        memory_aware_eft_alpha=memory_aware_eft_alpha,
                        external_mapper_config=external_mapper_config,
                        transition_config=transition_config,
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
                        "mapper_name": mapper_name,
                        "eviction_policy_name": eviction_policy_name,
                        "avg_sim_us": statistics.mean(sim_times),
                        "avg_total_mv": statistics.mean(total_mvs),
                        "avg_evict_mv": statistics.mean(evict_mvs),
                    }
                )

        summary_rows.sort(key=lambda row: row["avg_sim_us"])
        print("  summary:")
        for row in summary_rows:
            print(
                f"    {mapper_label(row['mapper_name'], memory_aware_eft_alpha=memory_aware_eft_alpha):>28}  {row['eviction_policy_name']:<18}  "
                f"sim={row['avg_sim_us'] / 1e6:8.3f}s  "
                f"total_mv={_bytes_hr(row['avg_total_mv']):>10}  "
                f"evict_mv={_bytes_hr(row['avg_evict_mv']):>10}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cholesky eviction stress benchmark")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of consecutive seeds to run (seed, seed+1, ...)",
    )
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Repetitions per seed")
    parser.add_argument(
        "--mapper",
        choices=ALL_MAPPER_NAMES,
        default=DEFAULT_MAPPER,
        help="Mapper to benchmark",
    )
    parser.add_argument(
        "--eviction-policy",
        choices=("lru", "least_used_mapped"),
        default=DEFAULT_EVICTION_POLICY,
        help="Eviction policy",
    )
    parser.add_argument(
        "--n-blocks",
        default=DEFAULT_BLOCKS,
        help="Comma-separated matrix block counts to benchmark, e.g. '16,24,32'",
    )
    parser.add_argument(
        "--block-bytes-gb",
        default=DEFAULT_BLOCK_BYTES_GB,
        help="Comma-separated per-block sizes in GiB, e.g. '0.5,1.0'",
    )
    parser.add_argument(
        "--gpu-mem-gb",
        type=float,
        default=GPU_MEM / 1e9,
        help="Per-GPU memory capacity in decimal GB",
    )
    parser.add_argument(
        "--no-randomize-locations",
        action="store_true",
        help="Keep all initial data on CPU instead of randomizing over GPUs",
    )
    parser.add_argument(
        "--verbose-graph",
        action="store_true",
        help="Enable Cholesky graph construction logging on the first rep of each scenario",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run a comparison matrix across multiple mappers and eviction policies",
    )
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
    parser.add_argument(
        "--pressure-presets",
        default="",
        help=(
            "Comma-separated pressure presets to derive block sizes from target overcommit ratios. "
            f"Available: {', '.join(PRESSURE_PRESET_RATIOS.keys())}"
        ),
    )
    parser.add_argument(
        "--memory-aware-eft-alpha",
        type=float,
        default=DEFAULT_MEMORY_AWARE_EFT_ALPHA,
        help="Alpha multiplier for MemoryAwareEFTMapper eviction-cost penalty",
    )
    parser.add_argument("--block-rows", type=int, default=1, help="External block-cyclic row block size")
    parser.add_argument("--block-cols", type=int, default=1, help="External block-cyclic col block size")
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

    n_blocks_list = _parse_int_list(args.n_blocks)
    block_bytes_list = [
        align_block_bytes(int(block_gb * 1e9), 4) for block_gb in _parse_float_list(args.block_bytes_gb)
    ]
    gpu_mem = int(args.gpu_mem_gb * 1e9)
    pressure_presets = _parse_str_list(args.pressure_presets)
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

    print("=" * 72)
    print("Cholesky Eviction Benchmark")
    print(f"GPUs            : {N_DEVICES - 1}")
    print(f"GPU memory      : {_bytes_hr(gpu_mem)} per GPU")
    print(f"Mapper          : {mapper_label(args.mapper, memory_aware_eft_alpha=args.memory_aware_eft_alpha)}")
    print(f"Eviction policy : {args.eviction_policy}")
    print(f"Transition kind : {args.transition_kind}")
    print(f"Seeds           : {args.seed}..{args.seed + args.n_seeds - 1}")
    print(f"Repetitions     : {args.reps}")
    print(f"n_blocks sweep  : {n_blocks_list}")
    if pressure_presets:
        print(f"pressure preset : {pressure_presets}")
    else:
        print(f"block sweep     : {[_bytes_hr(v) for v in block_bytes_list]}")
    if args.mapper in EXTERNAL_MAPPER_NAMES:
        print(
            f"External mapper : block_rows={args.block_rows}  block_cols={args.block_cols}  "
            f"processor_rows={args.processor_rows}  processor_cols={args.processor_cols}"
        )
    print("=" * 72)

    if pressure_presets:
        scenarios = build_pressure_scenarios(n_blocks_list, pressure_presets, gpu_mem)
    else:
        scenarios = build_explicit_scenarios(n_blocks_list, block_bytes_list)

    if args.compare:
        compare_scenarios(
            seed_start=args.seed,
            n_seeds=args.n_seeds,
            reps=args.reps,
            scenarios=scenarios,
            mapper_names=_parse_str_list(args.compare_mappers),
            eviction_policy_names=_parse_str_list(args.compare_eviction_policies),
            gpu_mem=gpu_mem,
            randomize_initial_placement=not args.no_randomize_locations,
            verbose_graph=args.verbose_graph,
            memory_aware_eft_alpha=args.memory_aware_eft_alpha,
            external_mapper_config=external_mapper_config,
            transition_config=transition_config,
        )
        print("=" * 72)
        return

    for seed_offset in range(args.n_seeds):
        seed = args.seed + seed_offset
        for scenario in scenarios:
                run_scenario(
                    seed=seed,
                    n_blocks=int(scenario["n_blocks"]),
                    block_bytes=int(scenario["block_bytes"]),
                    gpu_mem=gpu_mem,
                    mapper_name=args.mapper,
                    eviction_policy_name=args.eviction_policy,
                    reps=args.reps,
                    randomize_initial_placement=not args.no_randomize_locations,
                    verbose_graph=args.verbose_graph,
                    memory_aware_eft_alpha=args.memory_aware_eft_alpha,
                    external_mapper_config=external_mapper_config,
                    transition_config=transition_config,
                )

    print("=" * 72)


if __name__ == "__main__":
    main()
