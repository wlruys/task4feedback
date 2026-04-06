from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_fastsim_extension_from_env() -> None:
    extension_path = os.environ.get("TASK4FEEDBACK_FASTSIM2_PATH")
    if not extension_path:
        return

    spec = importlib.util.spec_from_file_location("task4feedback.fastsim2", extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {extension_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["task4feedback.fastsim2"] = module
    sys.modules["fastsim2"] = module
    spec.loader.exec_module(module)


_load_fastsim_extension_from_env()

from bench_eviction_jacobi import DEFAULT_COMPARE_EVICTION_POLICIES, DEFAULT_COMPARE_MAPPERS
from bench_eviction_jacobi import GPU_MEM as DEFAULT_GPU_MEM
from bench_eviction_jacobi import run_scenario
from bench_mapper_support import (
    DARTSConfig,
    EnhancedDARTSConfig,
    ExternalMapperConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    mapper_label,
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _bytes_hr(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _metric_axis_label(metric: str) -> str:
    if metric == "sim_s":
        return "Simulation time (s)"
    if metric == "total_mv":
        return "Total movement (bytes)"
    if metric == "evict_mv":
        return "Eviction movement (bytes)"
    raise ValueError(f"Unsupported metric '{metric}'")


def _metric_filename(metric: str) -> str:
    if metric == "sim_s":
        return "simulation_time"
    if metric == "total_mv":
        return "total_movement"
    if metric == "evict_mv":
        return "evict_movement"
    raise ValueError(f"Unsupported metric '{metric}'")


def _metric_from_result(metric: str, result: dict[str, float | int]) -> float:
    if metric == "sim_s":
        return float(result["sim_us"]) / 1e6
    if metric == "total_mv":
        return float(result["total_mv"])
    if metric == "evict_mv":
        return float(result["evict_mv"])
    raise ValueError(f"Unsupported metric '{metric}'")


def _series_label(
    mapper_name: str,
    eviction_policy_name: str,
    memory_aware_eft_alpha: float,
    memory_aware_eft_config: MemoryAwareEFTConfig,
) -> str:
    return (
        f"{mapper_label(mapper_name, memory_aware_eft_alpha=memory_aware_eft_alpha, memory_aware_eft_config=memory_aware_eft_config)}"
        f" + {eviction_policy_name.upper()}"
    )


def _formatter_for_metric(metric: str):
    if metric == "sim_s":
        return lambda value, _pos: f"{value:.0f}s" if value >= 10 else f"{value:.2f}s"
    return lambda value, _pos: _bytes_hr(value)


def collect_results(
    *,
    level_memory_gb_values: list[float],
    mapper_names: list[str],
    eviction_policy_names: list[str],
    seed_start: int,
    n_seeds: int,
    reps: int,
    gpu_mem_gb: float,
    grid_n: int,
    steps: int,
    randomize_initial_placement: bool,
    memory_aware_eft_alpha: float,
    memory_aware_eft_config: MemoryAwareEFTConfig,
    external_mapper_config: ExternalMapperConfig,
    transition_config: TransitionConfig,
    darts_config: DARTSConfig,
    enhanced_darts_config: EnhancedDARTSConfig,
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
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    gpu_mem = int(gpu_mem_gb * 1e9)
    for level_memory_gb in level_memory_gb_values:
        level_memory = int(level_memory_gb * 1e9)
        for mapper_name in mapper_names:
            for eviction_policy_name in eviction_policy_names:
                sim_values: list[float] = []
                total_values: list[float] = []
                evict_values: list[float] = []
                for seed_offset in range(n_seeds):
                    seed = seed_start + seed_offset
                    try:
                        results = run_scenario(
                            seed=seed,
                            grid_n=grid_n,
                            steps=steps,
                            level_memory=level_memory,
                            gpu_mem=gpu_mem,
                            mapper_name=mapper_name,
                            eviction_policy_name=eviction_policy_name,
                            reps=reps,
                            top_k_candidates=64,
                            randomize_initial_placement=randomize_initial_placement,
                            memory_aware_eft_alpha=memory_aware_eft_alpha,
                            memory_aware_eft_config=memory_aware_eft_config,
                            external_mapper_config=external_mapper_config,
                            transition_config=transition_config,
                            darts_config=(
                                enhanced_darts_config
                                if mapper_name == "enhanced_darts"
                                else darts_config
                            ),
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
                    except RuntimeError as exc:
                        print(
                            f"Skipping mapper={mapper_name} eviction={eviction_policy_name} "
                            f"level_memory_gb={level_memory_gb:g}: {exc}"
                        )
                        results = None
                    if results is None:
                        continue
                    for result in results:
                        sim_values.append(float(result["sim_us"]) / 1e6)
                        total_values.append(float(result["total_mv"]))
                        evict_values.append(float(result["evict_mv"]))
                if not sim_values:
                    continue
                rows.append(
                    {
                        "benchmark": "jacobi",
                        "grid_n": grid_n,
                        "steps": steps,
                        "level_memory_gb": level_memory_gb,
                        "level_memory_bytes": level_memory,
                        "mapper_name": mapper_name,
                        "eviction_policy_name": eviction_policy_name,
                        "series_label": _series_label(
                            mapper_name,
                            eviction_policy_name,
                            memory_aware_eft_alpha,
                            memory_aware_eft_config,
                        ),
                        "avg_sim_s": sum(sim_values) / len(sim_values),
                        "avg_total_mv": sum(total_values) / len(total_values),
                        "avg_evict_mv": sum(evict_values) / len(evict_values),
                        "n_runs": len(sim_values),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "benchmark",
        "grid_n",
        "steps",
        "level_memory_gb",
        "level_memory_bytes",
        "mapper_name",
        "eviction_policy_name",
        "series_label",
        "avg_sim_s",
        "avg_total_mv",
        "avg_evict_mv",
        "n_runs",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, float | int | str]],
    *,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    series_labels = sorted({str(row["series_label"]) for row in rows})
    all_ys: list[float] = []
    series_data: list[tuple[list[float], list[float], str]] = []
    for label in series_labels:
        series_rows = [row for row in rows if row["series_label"] == label]
        series_rows.sort(key=lambda row: float(row["level_memory_gb"]))
        xs = [float(row["level_memory_gb"]) for row in series_rows]
        ys = [_metric_from_result(metric, {
            "sim_us": float(row["avg_sim_s"]) * 1e6,
            "total_mv": float(row["avg_total_mv"]),
            "evict_mv": float(row["avg_evict_mv"]),
        }) for row in series_rows]
        series_data.append((xs, ys, label))
        all_ys.extend(ys)

    has_positive = any(v > 0 for v in all_ys)
    plot_fn = ax.semilogy if has_positive else ax.plot
    for xs, ys, label in series_data:
        plot_fn(xs, ys, marker="o", linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Per-level memory (decimal GB)")
    ax.set_ylabel(_metric_axis_label(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    if has_positive:
        ax.yaxis.set_major_formatter(_formatter_for_metric(metric))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Jacobi eviction benchmark sweeps")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per seed")
    parser.add_argument("--grid-n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument(
        "--level-memory-gb",
        default="60,80,100,120,140",
        help="Comma-separated per-level memory sizes in decimal GB",
    )
    parser.add_argument(
        "--gpu-mem-gb",
        type=float,
        default=DEFAULT_GPU_MEM / 1e9,
        help="Per-GPU memory capacity in decimal GB",
    )
    parser.add_argument(
        "--mappers",
        default=DEFAULT_COMPARE_MAPPERS,
        help="Comma-separated mapper list",
    )
    parser.add_argument(
        "--eviction-policies",
        default=DEFAULT_COMPARE_EVICTION_POLICIES,
        help="Comma-separated eviction-policy list",
    )
    parser.add_argument("--memory-aware-eft-alpha", type=float, default=1.0)
    parser.add_argument(
        "--memory-aware-location-state",
        choices=("launched", "reserved", "mapped"),
        default="reserved",
    )
    parser.add_argument(
        "--memory-aware-overflow-state",
        choices=("reserved", "mapped", "launched"),
        default="reserved",
    )
    parser.add_argument(
        "--memory-aware-overflow-mode",
        choices=("full_spill", "incoming_only"),
        default="incoming_only",
    )
    parser.add_argument("--block-rows", type=int, default=1)
    parser.add_argument("--block-cols", type=int, default=1)
    parser.add_argument("--processor-rows", type=int, default=None)
    parser.add_argument("--processor-cols", type=int, default=None)
    parser.add_argument(
        "--transition-kind",
        choices=("auto", "planned", "default", "batch", "range", "hysteresis"),
        default="auto",
    )
    parser.add_argument("--transition-planned-threshold", type=int, default=1)
    parser.add_argument("--transition-max-reserved-threshold", type=int, default=16)
    parser.add_argument("--transition-batch-size", type=int, default=5)
    parser.add_argument("--transition-queue-threshold", type=int, default=5)
    parser.add_argument("--transition-max-in-flight", type=int, default=None)
    parser.add_argument("--transition-mapped-reserved-gap", type=int, default=5)
    parser.add_argument("--transition-reserved-launched-gap", type=int, default=5)
    parser.add_argument("--transition-total-in-flight", type=int, default=None)
    parser.add_argument("--transition-hysteresis-open", type=int, default=16)
    parser.add_argument("--transition-hysteresis-close", type=int, default=36)
    parser.add_argument("--transition-hysteresis-starvation", type=int, default=2)
    parser.add_argument("--darts-short-horizon-threshold", type=int, default=4)
    parser.add_argument("--darts-medium-horizon-threshold", type=int, default=8)
    parser.add_argument("--darts-emit-short-horizon", action="store_true", default=True)
    parser.add_argument("--darts-emit-medium-horizon", action="store_true", default=True)
    parser.add_argument("--darts-short-horizon-k", type=int, default=4)
    parser.add_argument("--darts-medium-horizon-k", type=int, default=4)
    parser.add_argument("--enhanced-darts-short-horizon-threshold", type=int, default=4)
    parser.add_argument("--enhanced-darts-medium-horizon-threshold", type=int, default=8)
    parser.add_argument("--enhanced-darts-emit-short-horizon", action="store_true", default=True)
    parser.add_argument("--enhanced-darts-emit-medium-horizon", action="store_true", default=True)
    parser.add_argument("--enhanced-darts-short-horizon-k", type=int, default=4)
    parser.add_argument("--enhanced-darts-medium-horizon-k", type=int, default=4)
    parser.add_argument("--enhanced-darts-no-finish-time-aware", action="store_true", default=False)
    parser.add_argument("--enhanced-darts-no-local-data-first", action="store_true", default=False)
    parser.add_argument("--enhanced-darts-simulate-memory", action="store_true", default=True)
    parser.add_argument("--enhanced-darts-cascade-passes", type=int, default=1)
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
    parser.add_argument(
        "--no-randomize-locations",
        action="store_true",
        help="Keep all initial data on CPU",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bench_eviction_jacobi"))
    parser.add_argument("--stem", type=str, default="bench_eviction_jacobi")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    external_mapper_config = ExternalMapperConfig(
        block_rows=args.block_rows,
        block_cols=args.block_cols,
        processor_rows=args.processor_rows,
        processor_cols=args.processor_cols,
    )
    transition_config = TransitionConfig(
        kind=args.transition_kind,
        planned_threshold=args.transition_planned_threshold,
        max_reserved_threshold=args.transition_max_reserved_threshold,
        batch_size=args.transition_batch_size,
        queue_threshold=args.transition_queue_threshold,
        max_in_flight=args.transition_max_in_flight,
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
    darts_config = DARTSConfig(
        short_horizon_threshold=args.darts_short_horizon_threshold,
        medium_horizon_threshold=args.darts_medium_horizon_threshold,
        emit_short_horizon=args.darts_emit_short_horizon,
        emit_medium_horizon=args.darts_emit_medium_horizon,
        short_horizon_k=args.darts_short_horizon_k,
        medium_horizon_k=args.darts_medium_horizon_k,
    )
    enhanced_darts_config = EnhancedDARTSConfig(
        short_horizon_threshold=args.enhanced_darts_short_horizon_threshold,
        medium_horizon_threshold=args.enhanced_darts_medium_horizon_threshold,
        emit_short_horizon=args.enhanced_darts_emit_short_horizon,
        emit_medium_horizon=args.enhanced_darts_emit_medium_horizon,
        short_horizon_k=args.enhanced_darts_short_horizon_k,
        medium_horizon_k=args.enhanced_darts_medium_horizon_k,
        finish_time_aware=not args.enhanced_darts_no_finish_time_aware,
        local_data_first=not args.enhanced_darts_no_local_data_first,
    )
    rows = collect_results(
        level_memory_gb_values=_parse_float_list(args.level_memory_gb),
        mapper_names=_parse_str_list(args.mappers),
        eviction_policy_names=_parse_str_list(args.eviction_policies),
        seed_start=args.seed,
        n_seeds=args.n_seeds,
        reps=args.reps,
        gpu_mem_gb=args.gpu_mem_gb,
        grid_n=args.grid_n,
        steps=args.steps,
        randomize_initial_placement=not args.no_randomize_locations,
        memory_aware_eft_alpha=args.memory_aware_eft_alpha,
        memory_aware_eft_config=memory_aware_eft_config,
        external_mapper_config=external_mapper_config,
        transition_config=transition_config,
        darts_config=darts_config,
        enhanced_darts_config=enhanced_darts_config,
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
    if not rows:
        raise RuntimeError("No Jacobi benchmark data was collected")

    csv_path = args.output_dir / f"{args.stem}.csv"
    write_csv(rows, csv_path)
    for metric in ("sim_s", "total_mv", "evict_mv"):
        output_path = args.output_dir / f"{args.stem}_{_metric_filename(metric)}.png"
        plot_metric(
            rows,
            metric=metric,
            output_path=output_path,
            title=(
                f"Jacobi eviction sweep: {metric.replace('_', ' ')} vs level memory "
                f"(grid={args.grid_n}, steps={args.steps})"
            ),
        )
        print(output_path)
    print(csv_path)


if __name__ == "__main__":
    main()
