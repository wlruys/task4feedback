#!/usr/bin/env python3
"""
Comprehensive benchmarking script to test all memory_aware_policy combinations
and identify the best performing configuration.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
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

from bench_eviction_jacobi import run_scenario
from bench_eviction_jacobi import GPU_MEM as DEFAULT_GPU_MEM
from bench_mapper_support import (
    DARTSConfig,
    EnhancedDARTSConfig,
    ExternalMapperConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
)


def _bytes_hr(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def _get_config_label(location_state: str, overflow_state: str, overflow_mode: str) -> str:
    """Generate a readable label for a configuration."""
    return f"L:{location_state[:3]}_O:{overflow_state[:3]}_M:{overflow_mode[:4]}"


def collect_all_combinations_results(
    *,
    level_memory_gb: float,
    mapper_name: str = "memory_aware_eft",
    eviction_policy_name: str = "lru",
    seed_start: int,
    n_seeds: int,
    reps: int,
    gpu_mem_gb: float,
    grid_n: int,
    steps: int,
    randomize_initial_placement: bool,
    memory_aware_eft_alpha: float,
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
    """Collect results for all memory_aware_policy combinations."""

    # All possible combinations
    location_states = ["launched", "reserved", "mapped"]
    overflow_states = ["reserved", "mapped", "launched"]
    overflow_modes = ["full_spill", "incoming_only"]

    rows: list[dict[str, float | int | str]] = []
    gpu_mem = int(gpu_mem_gb * 1e9)
    level_memory = int(level_memory_gb * 1e9)

    total_combinations = len(location_states) * len(overflow_states) * len(overflow_modes)
    current_combination = 0

    for location_state, overflow_state, overflow_mode in itertools.product(
        location_states, overflow_states, overflow_modes
    ):
        current_combination += 1
        config_label = _get_config_label(location_state, overflow_state, overflow_mode)

        print(f"Testing combination {current_combination}/{total_combinations}: {config_label}")

        memory_aware_eft_config = MemoryAwareEFTConfig(
            alpha=memory_aware_eft_alpha,
            eviction_cost_location_state=location_state,
            overflow_state=overflow_state,
            overflow_mode=overflow_mode,
        )

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
                    darts_config=darts_config,
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
                print(f"  FAILED: {exc}")
                results = None

            if results is None:
                continue

            for result in results:
                sim_values.append(float(result["sim_us"]) / 1e6)
                total_values.append(float(result["total_mv"]))
                evict_values.append(float(result["evict_mv"]))

        if not sim_values:
            print(f"  No valid results for {config_label}")
            continue

        avg_sim_s = sum(sim_values) / len(sim_values)
        avg_total_mv = sum(total_values) / len(total_values)
        avg_evict_mv = sum(evict_values) / len(evict_values)

        print(f"  Results: sim={avg_sim_s:.3f}s, total_mv={_bytes_hr(avg_total_mv)}, evict_mv={_bytes_hr(avg_evict_mv)}")

        rows.append({
            "config_label": config_label,
            "location_state": location_state,
            "overflow_state": overflow_state,
            "overflow_mode": overflow_mode,
            "avg_sim_s": avg_sim_s,
            "avg_total_mv": avg_total_mv,
            "avg_evict_mv": avg_evict_mv,
            "n_runs": len(sim_values),
        })

    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    """Write results to CSV file."""
    if not rows:
        return
    fieldnames = [
        "config_label",
        "location_state",
        "overflow_state",
        "overflow_mode",
        "avg_sim_s",
        "avg_total_mv",
        "avg_evict_mv",
        "n_runs",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict[str, float | int | str]], output_dir: Path, title_prefix: str) -> None:
    """Generate comparison plots for all metrics."""

    if not rows:
        print("No data to plot")
        return

    metrics = [
        ("avg_sim_s", "Simulation Time (s)", "simulation_time"),
        ("avg_total_mv", "Total Movement (bytes)", "total_movement"),
        ("avg_evict_mv", "Eviction Movement (bytes)", "eviction_movement"),
    ]

    for metric_key, ylabel, filename_suffix in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Sort by performance (ascending for sim_s, descending for movement)
        if metric_key == "avg_sim_s":
            sorted_rows = sorted(rows, key=lambda r: float(r[metric_key]))
        else:
            sorted_rows = sorted(rows, key=lambda r: float(r[metric_key]), reverse=True)

        labels = [row["config_label"] for row in sorted_rows]
        values = [float(row[metric_key]) for row in sorted_rows]

        # Use bar plot for better readability with many configurations
        bars = ax.barh(labels, values)

        # Color bars: green for best performers, red for worst
        n_configs = len(values)
        colors = []
        for i in range(n_configs):
            if i < n_configs // 3:  # Top third
                colors.append('green')
            elif i < 2 * n_configs // 3:  # Middle third
                colors.append('orange')
            else:  # Bottom third
                colors.append('red')

        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)

        ax.set_xlabel(ylabel)
        ax.set_ylabel("Configuration")
        ax.set_title(f"{title_prefix}: {ylabel} Comparison")
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            width = bar.get_width()
            if metric_key == "avg_sim_s":
                label = f"{value:.3f}s"
            else:
                label = _bytes_hr(value)
            ax.text(width * 0.5, bar.get_y() + bar.get_height() * 0.5,
                   label, ha='center', va='center', fontsize=8, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / f"memory_aware_policy_comparison_{filename_suffix}.png"
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {output_path}")


def analyze_best_configs(rows: list[dict[str, float | int | str]]) -> None:
    """Analyze and report the best configurations for each metric."""

    if not rows:
        return

    print("\n" + "="*80)
    print("BEST CONFIGURATION ANALYSIS")
    print("="*80)

    metrics = [
        ("avg_sim_s", "Simulation Time", "lower is better"),
        ("avg_total_mv", "Total Movement", "lower is better"),
        ("avg_evict_mv", "Eviction Movement", "lower is better"),
    ]

    for metric_key, metric_name, direction in metrics:
        print(f"\n{metric_name} ({direction}):")
        print("-" * 40)

        sorted_rows = sorted(rows, key=lambda r: float(r[metric_key]))

        # Show top 5 configurations
        for i, row in enumerate(sorted_rows[:5]):
            value = float(row[metric_key])
            if metric_key == "avg_sim_s":
                value_str = f"{value:.3f}s"
            else:
                value_str = _bytes_hr(value)

            print(f"{i+1:2d}. {row['config_label']:20s} = {value_str:>12s} "
                  f"(L:{row['location_state']}, O:{row['overflow_state']}, M:{row['overflow_mode']})")

    # Overall best configuration (weighted scoring)
    print(f"\n{'OVERALL BEST CONFIGURATION'}")
    print("-" * 40)

    # Normalize and weight metrics (sim_time = 50%, evict_movement = 30%, total_movement = 20%)
    sim_values = [float(r["avg_sim_s"]) for r in rows]
    evict_values = [float(r["avg_evict_mv"]) for r in rows]
    total_values = [float(r["avg_total_mv"]) for r in rows]

    sim_min, sim_max = min(sim_values), max(sim_values)
    evict_min, evict_max = min(evict_values), max(evict_values)
    total_min, total_max = min(total_values), max(total_values)

    scored_rows = []
    for row in rows:
        # Normalize to 0-1 (lower is better for all metrics)
        sim_norm = (float(row["avg_sim_s"]) - sim_min) / (sim_max - sim_min) if sim_max > sim_min else 0
        evict_norm = (float(row["avg_evict_mv"]) - evict_min) / (evict_max - evict_min) if evict_max > evict_min else 0
        total_norm = (float(row["avg_total_mv"]) - total_min) / (total_max - total_min) if total_max > total_min else 0

        # Weighted score (lower is better)
        overall_score = 0.5 * sim_norm + 0.3 * evict_norm + 0.2 * total_norm
        scored_rows.append((row, overall_score))

    scored_rows.sort(key=lambda x: x[1])  # Sort by score ascending

    best_row, best_score = scored_rows[0]
    print(f"Best Overall: {best_row['config_label']}")
    print(f"  Location State: {best_row['location_state']}")
    print(f"  Overflow State: {best_row['overflow_state']}")
    print(f"  Overflow Mode: {best_row['overflow_mode']}")
    print(f"  Simulation Time: {float(best_row['avg_sim_s']):.3f}s")
    print(f"  Total Movement: {_bytes_hr(float(best_row['avg_total_mv']))}")
    print(f"  Eviction Movement: {_bytes_hr(float(best_row['avg_evict_mv']))}")
    print(f"  Overall Score: {best_score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive memory_aware_policy configuration benchmarking"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per seed")
    parser.add_argument("--grid-n", type=int, default=8, help="Grid size")
    parser.add_argument("--steps", type=int, default=256, help="Number of steps")
    parser.add_argument(
        "--level-memory-gb",
        type=float,
        default=80.0,
        help="Per-level memory size in decimal GB",
    )
    parser.add_argument(
        "--gpu-mem-gb",
        type=float,
        default=DEFAULT_GPU_MEM / 1e9,
        help="Per-GPU memory capacity in decimal GB",
    )
    parser.add_argument("--memory-aware-eft-alpha", type=float, default=1.0)

    # Workload parameters
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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/memory_aware_policy_sweep"))

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Use minimal configs for other components
    external_mapper_config = ExternalMapperConfig(
        block_rows=1, block_cols=1, processor_rows=None, processor_cols=None
    )
    transition_config = TransitionConfig(kind="auto")
    darts_config = DARTSConfig()
    enhanced_darts_config = EnhancedDARTSConfig()

    print(f"Starting comprehensive memory_aware_policy benchmarking...")
    print(f"Grid: {args.grid_n}x{args.grid_n}, Steps: {args.steps}, Level Memory: {args.level_memory_gb}GB")
    print(f"Testing all 18 combinations...")

    rows = collect_all_combinations_results(
        level_memory_gb=args.level_memory_gb,
        seed_start=args.seed,
        n_seeds=args.n_seeds,
        reps=args.reps,
        gpu_mem_gb=args.gpu_mem_gb,
        grid_n=args.grid_n,
        steps=args.steps,
        randomize_initial_placement=not args.no_randomize_locations,
        memory_aware_eft_alpha=args.memory_aware_eft_alpha,
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
        raise RuntimeError("No benchmark data was collected")

    # Save results to CSV
    csv_path = args.output_dir / "memory_aware_policy_sweep.csv"
    write_csv(rows, csv_path)

    # Generate comparison plots
    title_prefix = f"Memory Aware Policy Sweep (grid={args.grid_n}, steps={args.steps})"
    plot_comparison(rows, args.output_dir, title_prefix)

    # Analyze and report best configurations
    analyze_best_configs(rows)

    print(f"\nResults saved to: {csv_path}")
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()