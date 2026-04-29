from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import plot_grouped_time_vs_mem as plotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV with the original mem value closest to relative "
            "problem size 1.0 for each graph/interior/boundary group."
        )
    )
    parser.add_argument(
        "--ngpus",
        default="4",
        help=(
            "GPU counts to process. Accepts a single value like 4 or a "
            'comma-separated list like "4,8". Default: 4.'
        ),
    )
    parser.add_argument(
        "-e",
        "--extend",
        action="store_true",
        help=(
            "Use the 512-step result set and the existing normalization.csv "
            "baselines, matching plot_grouped_time_vs_mem.py --extend."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path. For one --ngpus value, defaults to "
            "results/<ngpus>gpus/normalization.csv. For multiple "
            "GPU counts, this must be omitted."
        ),
    )
    return parser.parse_args()


def nearest_problem_size_1_rows(
    n_gpus: int, extended: bool
) -> list[dict[str, str | int]]:
    csv_path, _rl_csv_path = plotter.resolve_results_paths(
        n_gpus, include_rl=False, noise=False, extended=extended
    )
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    normalization_csv_path: Path | None = None
    if extended:
        normalization_csv_path = plotter.resolve_normalization_path(n_gpus)
        normalization_csv_path = normalization_csv_path.expanduser().resolve()
        if not normalization_csv_path.is_file():
            raise FileNotFoundError(
                f"Normalization CSV file not found: {normalization_csv_path}"
            )

    plot_args = SimpleNamespace(
        inf=True,
        raw=False,
        extend=extended,
        norm=False,
    )
    _grouped_rows, _zero_eviction_points, config_points, _baseline_mems, _tops = (
        plotter.prepare_plot_data(
            csv_path,
            None,
            normalization_csv_path,
            n_gpus,
            plot_args,
        )
    )

    rows: list[dict[str, str | int]] = []
    for (graph, interior, boundary), candidates in sorted(
        config_points.items(),
        key=lambda item: (item[0][0], float(item[0][1]), float(item[0][2])),
    ):
        if not candidates:
            continue
        _relative_size, original_mem = min(
            candidates, key=lambda item: abs(item[0] - 1.0)
        )
        rows.append(
            {
                "graph": graph,
                "mem": int(original_mem),
                "interior": plotter.json_number(float(interior)),
                "boundary": plotter.json_number(float(boundary)),
            }
        )
    return rows


def write_rows(rows: list[dict[str, str | int]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["graph", "mem", "interior", "boundary"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def default_output_path(n_gpus: int, extended: bool) -> Path:
    prefix = "512_" if extended else ""
    return (
        Path(__file__).resolve().parent
        / "results"
        / f"{prefix}{n_gpus}gpus"
        / "normalization.csv"
    )


def main() -> None:
    args = parse_args()
    ngpus_values = plotter.parse_ngpus_list(args.ngpus)
    if args.output is not None and len(ngpus_values) != 1:
        raise ValueError("--output can only be used with a single --ngpus value")

    for n_gpus in ngpus_values:
        rows = nearest_problem_size_1_rows(n_gpus, args.extend)
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else default_output_path(n_gpus, args.extend)
        )
        write_rows(rows, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
