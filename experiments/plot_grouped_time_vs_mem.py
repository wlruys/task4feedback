from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_GPUS = 8

REQUIRED_COLUMNS = {
    "graph",
    "mem",
    "single_device",
    "interior",
    "boundary",
    "mapper",
    "time",
    "eviction",
}
STATIC_MAPPERS = {"b2", "b4", "rc"}
STATIC_LABEL = "Static"
RL_LABEL = "RL"
RELATIVE_PROBLEM_SIZES = [0.8, 0.9, 1.0, 1.1, 1.2]
TRAIN_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parent
    / "launch_conf"
    / "sc26"
    / f"{N_GPUS}gpus"
    / "train.json"
)
NORM_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parent
    / "launch_conf"
    / "sc26"
    / f"{N_GPUS}gpus"
    / "norm.json"
)
PICKLE_EVAL_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parent
    / "launch_conf"
    / "sc26"
    / f"{N_GPUS}gpus"
    / "pickle_eval.json"
)
MAPPER_COLORS = {
    "Static": "tab:blue",
    "RL": "tab:orange",
    "eft": "tab:green",
    "parmetis": "tab:red",
}
FALLBACK_COLORS = [
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one mem-vs-time plot per "
            "(graph, interior, boundary) group from a CSV file."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "rl_csv_path",
        nargs="?",
        type=Path,
        help="Optional path to RL results CSV.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "value"


# def transform_mem(mem: float, interior: float, boundary: float) -> float:
# Mem is sum of interior memory
# interior_mem = mem * 1  # Current and next
# boundary_mem = mem * (boundary / interior) * 2  # current + shadow
# return (interior_mem + boundary_mem) / 4 / 96e9


def transform_mem(mem: float, interior: float, boundary: float) -> float:
    return mem / N_GPUS / 96e9


def normalize_mapper(mapper: str) -> str:
    if mapper in STATIC_MAPPERS:
        return STATIC_LABEL
    return mapper


def format_mem_value(mem: float) -> str | int:
    if mem % 1e9 == 0:
        return f"{int(mem / 1e9)}e9"
    return int(mem)


def mapper_color(mapper: str) -> str:
    if mapper in MAPPER_COLORS:
        return MAPPER_COLORS[mapper]

    fallback_index = sum(ord(char) for char in mapper) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[fallback_index]


def canonical_number_str(value: str) -> str:
    return format(Decimal(value.strip()), "f")


def json_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def normalize_json_numbers(value: object) -> object:
    if isinstance(value, float):
        return json_number(value)
    if isinstance(value, list):
        return [normalize_json_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json_numbers(item) for key, item in value.items()}
    return value


def make_base_lookup_key(
    graph: str, mem: str, interior: str, boundary: str
) -> tuple[str, str, str, str]:
    try:
        return (
            graph.strip(),
            canonical_number_str(mem),
            canonical_number_str(interior),
            canonical_number_str(boundary),
        )
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(
            f"Invalid lookup key values: graph={graph}, mem={mem}, interior={interior}, boundary={boundary}"
        ) from exc


def read_rows(
    csv_path: Path,
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    dict[tuple[str, str, str], list[tuple[float, float]]],
    dict[tuple[str, str, str, str], tuple[tuple[str, str, str], float]],
]:
    grouped_rows: dict[tuple[str, str, str], dict[str, dict[float, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    zero_eviction_max_mem: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        dict
    )
    config_candidates: dict[tuple[str, str, str], dict[float, float]] = defaultdict(
        dict
    )
    base_lookup: dict[
        tuple[str, str, str, str], tuple[tuple[str, str, str], float]
    ] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_str}")

        for row_number, row in enumerate(reader, start=2):
            try:
                group_key = (
                    row["graph"].strip(),
                    row["interior"].strip(),
                    row["boundary"].strip(),
                )
                mapper = row["mapper"].strip() or "unknown"
                if "_inf" in mapper:
                    continue
                mapper = normalize_mapper(mapper)
                interior = float(row["interior"])
                boundary = float(row["boundary"])
                transformed_mem = transform_mem(
                    float(row["single_device"]), interior, boundary
                )
                original_mem = float(row["mem"])
                time = float(row["time"])
                eviction = float(row["eviction"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data in row {row_number}: {row}") from exc

            previous_time = grouped_rows[group_key][mapper].get(transformed_mem)
            if previous_time is None or time < previous_time:
                grouped_rows[group_key][mapper][transformed_mem] = time

            if eviction == 0:
                previous_zero_eviction_mem = zero_eviction_max_mem[group_key].get(
                    mapper
                )
                if (
                    previous_zero_eviction_mem is None
                    or transformed_mem > previous_zero_eviction_mem
                ):
                    zero_eviction_max_mem[group_key][mapper] = transformed_mem

            config_candidates[group_key][transformed_mem] = original_mem
            lookup_key = make_base_lookup_key(
                row["graph"], row["mem"], row["interior"], row["boundary"]
            )
            base_lookup[lookup_key] = (group_key, transformed_mem)
    series_by_group = {
        group_key: {
            mapper: sorted(mem_to_time.items())
            for mapper, mem_to_time in mapper_rows.items()
        }
        for group_key, mapper_rows in grouped_rows.items()
    }
    zero_eviction_points = {
        group_key: {
            mapper: (max_mem, dict(points)[max_mem])
            for mapper, max_mem in mapper_max_mem.items()
            if mapper in series_by_group.get(group_key, {})
            for points in [series_by_group[group_key][mapper]]
            if max_mem in dict(points)
        }
        for group_key, mapper_max_mem in zero_eviction_max_mem.items()
    }
    config_points = {
        group_key: sorted(relative_to_mem.items())
        for group_key, relative_to_mem in config_candidates.items()
    }

    return series_by_group, zero_eviction_points, config_points, base_lookup


def merge_optional_rl_results(
    rl_csv_path: Path,
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    zero_eviction_points: dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    base_lookup: dict[tuple[str, str, str, str], tuple[tuple[str, str, str], float]],
) -> None:
    rl_series: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
    rl_zero_eviction_mem: dict[tuple[str, str, str], float] = {}

    with rl_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "graph",
            "mem",
            "interior",
            "boundary",
            "mapper",
            "time",
            "eviction",
        }
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"RL CSV is missing required columns: {missing_str}")

        for row_number, row in enumerate(reader, start=2):
            try:
                lookup_key = make_base_lookup_key(
                    row["graph"], row["mem"], row["interior"], row["boundary"]
                )
                time = float(row["time"])
                eviction = float(row["eviction"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid RL data in row {row_number}: {row}") from exc

            matched_entry = base_lookup.get(lookup_key)
            if matched_entry is None:
                continue
            group_key, transformed_mem = matched_entry

            previous_time = rl_series[group_key].get(transformed_mem)
            if previous_time is None or time < previous_time:
                rl_series[group_key][transformed_mem] = time

            if eviction == 0:
                previous_zero_eviction_mem = rl_zero_eviction_mem.get(group_key)
                if (
                    previous_zero_eviction_mem is None
                    or transformed_mem > previous_zero_eviction_mem
                ):
                    rl_zero_eviction_mem[group_key] = transformed_mem

    for group_key, mem_to_time in rl_series.items():
        grouped_rows.setdefault(group_key, {})
        grouped_rows[group_key][RL_LABEL] = sorted(mem_to_time.items())

    for group_key, transformed_mem in rl_zero_eviction_mem.items():
        time = rl_series[group_key][transformed_mem]
        zero_eviction_points.setdefault(group_key, {})
        zero_eviction_points[group_key][RL_LABEL] = (transformed_mem, time)


def build_config_from_template(
    config_points: dict[tuple[str, str, str], list[tuple[float, float]]],
    template_path: Path,
    output_name: str,
    output_dir: Path,
) -> Path:
    with template_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    experiments: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (graph, interior, boundary), candidates in sorted(config_points.items()):
        if not candidates:
            continue

        mem_values = []
        for target_size in RELATIVE_PROBLEM_SIZES:
            closest_relative_size, closest_mem = min(
                candidates, key=lambda item: abs(item[0] - target_size)
            )
            _ = closest_relative_size
            mem_values.append(format_mem_value(closest_mem))

        deduped_mem_values = list(dict.fromkeys(mem_values))
        experiments[graph].append(
            {
                "interior": json_number(float(interior)),
                "boundary": json_number(float(boundary)),
                "mem": deduped_mem_values,
            }
        )

    config["experiments"] = dict(experiments)
    config = normalize_json_numbers(config)

    output_path = output_dir / output_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
        handle.write("\n")

    return output_path


def make_plot(
    group_key: tuple[str, str, str],
    series: dict[str, list[tuple[float, float]]],
    zero_eviction_points: dict[str, tuple[float, float]],
    output_dir: Path,
) -> Path:
    graph, interior, boundary = group_key

    fig, ax = plt.subplots(figsize=(8, 5))

    for mapper, points in sorted(series.items()):
        sorted_points = sorted(points)
        mem_values = [mem for mem, _time in sorted_points]
        time_values = [time for _mem, time in sorted_points]
        (line,) = ax.plot(
            mem_values,
            time_values,
            marker="o",
            linewidth=1.8,
            label=mapper,
            color=mapper_color(mapper),
        )

        marker_point = zero_eviction_points.get(mapper)
        if marker_point is not None:
            marker_mem, marker_time = marker_point
            ax.scatter(
                [marker_mem],
                [marker_time],
                color=line.get_color(),
                edgecolors="black",
                linewidths=1.0,
                marker="D",
                s=70,
                zorder=3,
            )

    ax.set_title(f"graph={graph}, interior={interior}, boundary={boundary}")
    ax.set_xlabel("Relative Problem Size")
    ax.set_ylabel("time")
    ax.grid(True, alpha=0.3)

    if len(series) > 1:
        ax.legend()

    fig.tight_layout()

    filename = (
        f"{slugify(graph)}__interior_{slugify(interior)}"
        f"__boundary_{slugify(boundary)}.png"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    rl_csv_path = None
    if args.rl_csv_path is not None:
        rl_csv_path = args.rl_csv_path.expanduser().resolve()
        if not rl_csv_path.is_file():
            raise FileNotFoundError(f"RL CSV file not found: {rl_csv_path}")

    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "plots" / f"{N_GPUS}gpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_rows, zero_eviction_points, config_points, base_lookup = read_rows(csv_path)
    if not grouped_rows:
        raise ValueError(f"No data rows found in CSV: {csv_path}")
    if rl_csv_path is not None:
        merge_optional_rl_results(
            rl_csv_path, grouped_rows, zero_eviction_points, base_lookup
        )

    output_paths = []
    for group_key, series in sorted(grouped_rows.items()):
        output_paths.append(
            make_plot(
                group_key, series, zero_eviction_points.get(group_key, {}), output_dir
            )
        )
    config_paths = [
        build_config_from_template(
            config_points, TRAIN_CONFIG_TEMPLATE, "train.json", output_dir
        ),
        build_config_from_template(
            config_points, NORM_CONFIG_TEMPLATE, "norm.json", output_dir
        ),
        build_config_from_template(
            config_points,
            PICKLE_EVAL_CONFIG_TEMPLATE,
            "pickle_eval.json",
            output_dir,
        ),
    ]

    print(f"Generated {len(output_paths)} plot(s) in {output_dir}")
    for output_path in output_paths:
        print(output_path)
    for config_path in config_paths:
        print(config_path)


if __name__ == "__main__":
    main()
