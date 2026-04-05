from __future__ import annotations

import argparse
import csv
import importlib.util
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

from bench_eviction_cholesky import DEFAULT_COMPARE_EVICTION_POLICIES, DEFAULT_COMPARE_MAPPERS
from bench_eviction_cholesky import GPU_MEM as DEFAULT_GPU_MEM
from bench_eviction_cholesky import derive_block_bytes_for_pressure, run_scenario
from bench_mapper_support import ExternalMapperConfig, TransitionConfig, mapper_label


def _parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _bytes_hr(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _series_label(mapper_name: str, eviction_policy_name: str, memory_aware_eft_alpha: float) -> str:
    return (
        f"{mapper_label(mapper_name, memory_aware_eft_alpha=memory_aware_eft_alpha)}"
        f" + {eviction_policy_name.upper()}"
    )


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


def _formatter_for_metric(metric: str):
    if metric == "sim_s":
        return lambda value, _pos: f"{value:.0f}s" if value >= 10 else f"{value:.2f}s"
    return lambda value, _pos: _bytes_hr(value)


def collect_results(
    *,
    sweep_axis: str,
    sweep_values: list[float],
    mapper_names: list[str],
    eviction_policy_names: list[str],
    seed_start: int,
    n_seeds: int,
    reps: int,
    gpu_mem_gb: float,
    fixed_n_blocks: int,
    fixed_block_bytes_gb: float,
    randomize_initial_placement: bool,
    verbose_graph: bool,
    memory_aware_eft_alpha: float,
    external_mapper_config: ExternalMapperConfig,
    transition_config: TransitionConfig,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    gpu_mem = int(gpu_mem_gb * 1e9)
    fixed_block_bytes = int(fixed_block_bytes_gb * 1e9)
    for sweep_value in sweep_values:
        if sweep_axis == "block_bytes_gb":
            n_blocks = fixed_n_blocks
            block_bytes_gb = float(sweep_value)
            block_bytes = int(block_bytes_gb * 1e9)
        elif sweep_axis == "n_blocks":
            n_blocks = int(sweep_value)
            block_bytes_gb = fixed_block_bytes_gb
            block_bytes = fixed_block_bytes
        else:
            raise ValueError(f"Unsupported sweep axis '{sweep_axis}'")

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
                            n_blocks=n_blocks,
                            block_bytes=block_bytes,
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
                    except RuntimeError as exc:
                        print(
                            f"Skipping mapper={mapper_name} eviction={eviction_policy_name} "
                            f"{sweep_axis}={sweep_value}: {exc}"
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
                        "benchmark": "cholesky",
                        "sweep_axis": sweep_axis,
                        "x_value": sweep_value,
                        "n_blocks": n_blocks,
                        "block_bytes_gb": block_bytes_gb,
                        "block_bytes": block_bytes,
                        "mapper_name": mapper_name,
                        "eviction_policy_name": eviction_policy_name,
                        "series_label": _series_label(
                            mapper_name,
                            eviction_policy_name,
                            memory_aware_eft_alpha,
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
        "sweep_axis",
        "x_value",
        "n_blocks",
        "block_bytes_gb",
        "block_bytes",
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
    sweep_axis: str,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    series_labels = sorted({str(row["series_label"]) for row in rows})
    for label in series_labels:
        series_rows = [row for row in rows if row["series_label"] == label]
        series_rows.sort(key=lambda row: float(row["x_value"]))
        xs = [float(row["x_value"]) for row in series_rows]
        if metric == "sim_s":
            ys = [float(row["avg_sim_s"]) for row in series_rows]
        elif metric == "total_mv":
            ys = [float(row["avg_total_mv"]) for row in series_rows]
        elif metric == "evict_mv":
            ys = [float(row["avg_evict_mv"]) for row in series_rows]
        else:
            raise ValueError(f"Unsupported metric '{metric}'")
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)

    if sweep_axis == "block_bytes_gb":
        ax.set_xlabel("Block memory (decimal GB)")
    else:
        ax.set_xlabel("Matrix blocks per dimension")
    ax.set_title(title)
    ax.set_ylabel(_metric_axis_label(metric))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(_formatter_for_metric(metric))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Cholesky eviction benchmark sweeps")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per seed")
    parser.add_argument(
        "--sweep-axis",
        choices=("block_bytes_gb", "n_blocks"),
        default="block_bytes_gb",
        help="Which Cholesky memory-growth knob to sweep",
    )
    parser.add_argument(
        "--sweep-values",
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated values for the chosen sweep axis",
    )
    parser.add_argument("--fixed-n-blocks", type=int, default=16)
    parser.add_argument("--fixed-block-bytes-gb", type=float, default=0.5)
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
    parser.add_argument("--block-rows", type=int, default=1)
    parser.add_argument("--block-cols", type=int, default=1)
    parser.add_argument("--processor-rows", type=int, default=None)
    parser.add_argument("--processor-cols", type=int, default=None)
    parser.add_argument(
        "--transition-kind",
        choices=("auto", "default", "batch", "device_threshold", "range"),
        default="auto",
    )
    parser.add_argument("--transition-batch-size", type=int, default=5)
    parser.add_argument("--transition-queue-threshold", type=int, default=5)
    parser.add_argument("--transition-max-in-flight", type=int, default=None)
    parser.add_argument("--transition-mapped-threshold", type=int, default=0)
    parser.add_argument("--transition-reserved-threshold", type=int, default=-1)
    parser.add_argument("--transition-mapped-reserved-gap", type=int, default=5)
    parser.add_argument("--transition-reserved-launched-gap", type=int, default=5)
    parser.add_argument("--transition-total-in-flight", type=int, default=None)
    parser.add_argument("--no-randomize-locations", action="store_true")
    parser.add_argument("--verbose-graph", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bench_eviction_cholesky"))
    parser.add_argument("--stem", type=str, default="bench_eviction_cholesky")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sweep_axis == "block_bytes_gb":
        sweep_values = _parse_float_list(args.sweep_values)
    else:
        sweep_values = [float(value) for value in _parse_int_list(args.sweep_values)]

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
    )
    rows = collect_results(
        sweep_axis=args.sweep_axis,
        sweep_values=sweep_values,
        mapper_names=_parse_str_list(args.mappers),
        eviction_policy_names=_parse_str_list(args.eviction_policies),
        seed_start=args.seed,
        n_seeds=args.n_seeds,
        reps=args.reps,
        gpu_mem_gb=args.gpu_mem_gb,
        fixed_n_blocks=args.fixed_n_blocks,
        fixed_block_bytes_gb=args.fixed_block_bytes_gb,
        randomize_initial_placement=not args.no_randomize_locations,
        verbose_graph=args.verbose_graph,
        memory_aware_eft_alpha=args.memory_aware_eft_alpha,
        external_mapper_config=external_mapper_config,
        transition_config=transition_config,
    )
    if not rows:
        raise RuntimeError("No Cholesky benchmark data was collected")

    csv_path = args.output_dir / f"{args.stem}.csv"
    write_csv(rows, csv_path)
    for metric in ("sim_s", "total_mv", "evict_mv"):
        output_path = args.output_dir / f"{args.stem}_{_metric_filename(metric)}.png"
        plot_metric(
            rows,
            metric=metric,
            sweep_axis=args.sweep_axis,
            output_path=output_path,
            title=f"Cholesky eviction sweep: {metric.replace('_', ' ')} vs {args.sweep_axis}",
        )
        print(output_path)
    print(csv_path)


if __name__ == "__main__":
    main()
