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
from matplotlib.ticker import FuncFormatter, LogLocator

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "Latin Modern Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    }
)
FONT_SIZE = 14

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
STATIC_MAPPERS = {"b2", "b4", "rc", "b1"}
STATIC_LABEL = "Static"
RL_LABEL = "RL"
RELATIVE_PROBLEM_SIZES = [0.8, 0.9, 1.0, 1.1, 1.2]
DEFAULT_RELATIVE_SIZE_MIN = 0.8
DEFAULT_RELATIVE_SIZE_MAX = 2.0
LIMIT_Y = 20.0
DEFAULT_INCLUDED_MAPPERS = (
    RL_LABEL,
    "darts",
    "memory_aware_eft",
    "parmetis",
    "Static",
    "Unconst",
)
MAPPER_COLORS = {
    "Static": "tab:blue",
    "RL": "tab:orange",
    "darts": "tab:pink",
    "eft": "tab:green",
    "memory_aware_eft": "tab:green",
    "parmetis": "tab:red",
    "Unconst": "tab:purple",
}
FALLBACK_COLORS = [
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
MICROSECONDS_PER_SECOND = 1000000.0
THRESHOLD = 1.05
UNCONST_MAPPER = "Unconst"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one mem-vs-time plot per (graph, interior, boundary) group "
            "using the standard results file pattern for a given GPU count."
        )
    )
    parser.add_argument(
        "--ngpus",
        required=True,
        help=(
            "GPU counts to process. Accepts a single value like 4 or a "
            'comma-separated list like "4,8".'
        ),
    )
    parser.add_argument(
        "--rl",
        action="store_true",
        help="Include RL results by reading the matching RL CSV.",
    )
    parser.add_argument(
        "-e",
        "--extend",
        action="store_true",
        help=(
            "Use the 512-step result set instead of 256 and normalize relative "
            "problem size with results/<ngpus>gpus/normalization.csv."
        ),
    )
    parser.add_argument(
        "--inf",
        action="store_true",
        help='Include an "Unconst" series from the minimum time of any mapper containing "_inf".',
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help='Use raw "mem" values on the x-axis instead of transformed "single_device" memory.',
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use a logarithmic scale for the y-axis.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Normalize all runtime values by the Unconst runtime at problem size 1.0.",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Generate an additional plot set from noise result CSVs.",
    )
    parser.add_argument(
        "--mappers",
        default=",".join(DEFAULT_INCLUDED_MAPPERS),
        help=(
            "Comma-separated mapper list to include globally. "
            f"Default: {', '.join(DEFAULT_INCLUDED_MAPPERS)}"
        ),
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "value"


def microseconds_to_seconds(value_us: float) -> float:
    return value_us / MICROSECONDS_PER_SECOND


def parse_ngpus_list(value: str) -> list[int]:
    ngpus_values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ngpus = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid --ngpus value: {item!r}") from exc
        if ngpus <= 0:
            raise ValueError("--ngpus values must be positive integers")
        ngpus_values.append(ngpus)
    if not ngpus_values:
        raise ValueError("At least one GPU count must be provided via --ngpus")
    return ngpus_values


# def transform_mem(mem: float, interior: float, boundary: float) -> float:
# Mem is sum of interior memory
# interior_mem = mem * 1  # Current and next
# boundary_mem = mem * (boundary / interior) * 2  # current + shadow
# return (interior_mem + boundary_mem) / 4 / 96e9


def transform_mem(mem: float, interior: float, boundary: float, n_gpus: int) -> float:
    return mem / n_gpus / 96e9


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


def display_mapper_name(mapper: str) -> str:
    display_names = {
        "darts": "DARTS",
        "memory_aware_eft": "EFT-MA",
        "eft": "EFT",
        RL_LABEL: RL_LABEL,
        STATIC_LABEL: STATIC_LABEL,
        "parmetis": "ParMETIS",
        # "Unconst": "Unconst",
    }
    return display_names.get(mapper, mapper)


def mapper_sort_key(mapper: str) -> tuple[int, str]:
    return (mapper == "Unconst", mapper)


def parse_included_mappers(mapper_spec: str) -> set[str]:
    parsed = {item.strip() for item in mapper_spec.split(",") if item.strip()}
    if not parsed:
        raise ValueError("At least one mapper must be provided via --mappers")
    return parsed


def apply_log_yaxis_format(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))


def interpolate_time(points: list[tuple[float, float]], mem: float) -> float | None:
    sorted_points = sorted(points)
    if not sorted_points:
        return None

    if mem < sorted_points[0][0] or mem > sorted_points[-1][0]:
        return None

    for point_mem, point_time in sorted_points:
        if point_mem == mem:
            return point_time

    for (left_mem, left_time), (right_mem, right_time) in zip(
        sorted_points, sorted_points[1:]
    ):
        if left_mem <= mem <= right_mem:
            if right_mem == left_mem:
                return left_time
            fraction = (mem - left_mem) / (right_mem - left_mem)
            return left_time + fraction * (right_time - left_time)

    return None


def is_baseline_candidate_mapper(mapper: str) -> bool:
    return (
        mapper != UNCONST_MAPPER
        and mapper != RL_LABEL
        and "_rl" not in mapper
        and "_inf" not in mapper
    )


def select_baseline_candidate_points(
    series: dict[str, list[tuple[float, float]]],
) -> list[tuple[float, float]]:
    baseline_by_mem: dict[float, float] = {}

    for mapper, points in series.items():
        if not is_baseline_candidate_mapper(mapper):
            continue
        for mem, time in points:
            previous_time = baseline_by_mem.get(mem)
            if previous_time is None or time < previous_time:
                baseline_by_mem[mem] = time

    return sorted(baseline_by_mem.items())


def fallback_problem_size_baseline(
    series: dict[str, list[tuple[float, float]]],
) -> float:
    candidate_mems = [
        mem for points in series.values() for mem, _time in points if mem > 0
    ]
    if not candidate_mems:
        raise ValueError(
            "Unable to determine a positive fallback problem size baseline."
        )
    return min(candidate_mems)


def compute_problem_size_baseline(
    baseline_points: list[tuple[float, float]],
    unconst_points: list[tuple[float, float]],
) -> float:
    candidate_mems = sorted(
        {
            mem
            for mem, _time in baseline_points
            if interpolate_time(unconst_points, mem) is not None
        }
        | {
            mem
            for mem, _time in unconst_points
            if interpolate_time(baseline_points, mem) is not None
        }
    )
    if not candidate_mems:
        raise ValueError(
            "Unable to compute problem size baseline because the constrained baseline "
            "and Unconst "
            "do not overlap on any memory values."
        )

    ratio_samples: list[tuple[float, float]] = []
    for mem in candidate_mems:
        baseline_time = interpolate_time(baseline_points, mem)
        unconst_time = interpolate_time(unconst_points, mem)
        if baseline_time is None or unconst_time is None or unconst_time == 0:
            continue
        ratio_samples.append((mem, baseline_time / unconst_time))

    if not ratio_samples:
        raise ValueError(
            "Unable to compute problem size baseline because constrained/Unconst ratios "
            "could not be evaluated."
        )

    max_qualifying_mem: float | None = None

    for mem, ratio in ratio_samples:
        if ratio < THRESHOLD:
            max_qualifying_mem = mem

    for (left_mem, left_ratio), (right_mem, right_ratio) in zip(
        ratio_samples, ratio_samples[1:]
    ):
        if left_ratio == right_ratio:
            continue

        left_above = left_ratio >= THRESHOLD
        right_above = right_ratio >= THRESHOLD
        if left_above == right_above:
            continue

        fraction = (THRESHOLD - left_ratio) / (right_ratio - left_ratio)
        crossing_mem = left_mem + fraction * (right_mem - left_mem)
        if left_above:
            max_qualifying_mem = crossing_mem
        else:
            max_qualifying_mem = max(
                crossing_mem,
                max_qualifying_mem if max_qualifying_mem is not None else crossing_mem,
            )

    if max_qualifying_mem is not None:
        return max_qualifying_mem

    raise ValueError(
        "Unable to compute problem size baseline because the constrained baseline/Unconst never "
        f"reaches THRESHOLD={THRESHOLD}."
    )


def compute_problem_size_baselines(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
) -> dict[tuple[str, str, str], float]:
    baseline_mems: dict[tuple[str, str, str], float] = {}

    for group_key, series in grouped_rows.items():
        baseline_points = select_baseline_candidate_points(series)
        unconst_points = series.get(UNCONST_MAPPER)
        if baseline_points and unconst_points:
            try:
                baseline_mem = compute_problem_size_baseline(
                    baseline_points, unconst_points
                )
            except ValueError:
                baseline_mem = fallback_problem_size_baseline(series)
        else:
            baseline_mem = fallback_problem_size_baseline(series)

        if baseline_mem <= 0:
            raise ValueError(
                f"Computed non-positive problem size baseline for group {group_key}: {baseline_mem}"
            )
        baseline_mems[group_key] = baseline_mem

    return baseline_mems


def normalize_problem_sizes(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    zero_eviction_points: dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    config_points: dict[tuple[str, str, str], list[tuple[float, float]]],
    baseline_mems: dict[tuple[str, str, str], float] | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    dict[tuple[str, str, str], list[tuple[float, float]]],
    dict[tuple[str, str, str], float],
]:
    normalized_rows: dict[
        tuple[str, str, str], dict[str, list[tuple[float, float]]]
    ] = {}
    normalized_zero_eviction: dict[
        tuple[str, str, str], dict[str, tuple[float, float]]
    ] = {}
    normalized_config_points: dict[tuple[str, str, str], list[tuple[float, float]]] = {}

    if baseline_mems is None:
        baseline_mems = compute_problem_size_baselines(grouped_rows)

    for group_key, series in grouped_rows.items():
        baseline_mem = baseline_mems.get(group_key)
        if baseline_mem is None:
            raise ValueError(f"Missing problem size baseline for group {group_key}.")
        if baseline_mem <= 0:
            raise ValueError(
                f"Computed non-positive problem size baseline for group {group_key}: {baseline_mem}"
            )

        normalized_rows[group_key] = {
            mapper: [(mem / baseline_mem, time) for mem, time in points]
            for mapper, points in series.items()
        }

        zero_eviction = zero_eviction_points.get(group_key, {})
        if zero_eviction:
            normalized_zero_eviction[group_key] = {
                mapper: (mem / baseline_mem, time)
                for mapper, (mem, time) in zero_eviction.items()
            }

        normalized_config_points[group_key] = sorted(
            (mem / baseline_mem, original_mem)
            for mem, original_mem in config_points.get(group_key, [])
        )

    return (
        normalized_rows,
        normalized_zero_eviction,
        normalized_config_points,
        baseline_mems,
    )


def compute_runtime_normalizers(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
) -> dict[tuple[str, str, str], float]:
    normalizers: dict[tuple[str, str, str], float] = {}

    for group_key, series in grouped_rows.items():
        unconst_points = series.get(UNCONST_MAPPER)
        if not unconst_points:
            raise ValueError(
                f"Missing {UNCONST_MAPPER} data for group {group_key}; "
                "cannot normalize runtimes."
            )

        baseline_time = interpolate_time(unconst_points, 1.0)
        if baseline_time is None or baseline_time <= 0:
            raise ValueError(
                f"Unable to compute {UNCONST_MAPPER} runtime at problem size 1.0 "
                f"for group {group_key}."
            )
        normalizers[group_key] = baseline_time

    return normalizers


def normalize_runtimes(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    zero_eviction_points: dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    runtime_normalizers: dict[tuple[str, str, str], float],
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str, str], dict[str, tuple[float, float]]],
]:
    normalized_rows: dict[
        tuple[str, str, str], dict[str, list[tuple[float, float]]]
    ] = {}
    normalized_zero_eviction: dict[
        tuple[str, str, str], dict[str, tuple[float, float]]
    ] = {}

    for group_key, series in grouped_rows.items():
        normalizer = runtime_normalizers[group_key]
        normalized_rows[group_key] = {
            mapper: [(mem, time / normalizer) for mem, time in points]
            for mapper, points in series.items()
        }

        zero_eviction = zero_eviction_points.get(group_key, {})
        if zero_eviction:
            normalized_zero_eviction[group_key] = {
                mapper: (mem, time / normalizer)
                for mapper, (mem, time) in zero_eviction.items()
            }

    return normalized_rows, normalized_zero_eviction


def filter_series(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    zero_eviction_points: dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    included_mappers: set[str],
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str, str], dict[str, tuple[float, float]]],
]:
    filtered_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] = {}
    filtered_zero_eviction: dict[
        tuple[str, str, str], dict[str, tuple[float, float]]
    ] = {}

    for group_key, mapper_rows in grouped_rows.items():
        kept_rows = {
            mapper: points
            for mapper, points in mapper_rows.items()
            if mapper in included_mappers
        }
        if not kept_rows:
            continue
        filtered_rows[group_key] = kept_rows

        kept_zero_eviction = {
            mapper: point
            for mapper, point in zero_eviction_points.get(group_key, {}).items()
            if mapper in included_mappers
        }
        if kept_zero_eviction:
            filtered_zero_eviction[group_key] = kept_zero_eviction

    return filtered_rows, filtered_zero_eviction


def select_target_points(
    points: list[tuple[float, float]], target_sizes: list[float]
) -> list[tuple[float, float]]:
    eligible_points = filter_points_for_display(points, use_raw_mem=False)
    if not eligible_points:
        return []

    selected: list[tuple[float, float]] = []
    seen_mem: set[float] = set()
    for target_size in target_sizes:
        closest_mem, closest_time = min(
            eligible_points, key=lambda item: abs(item[0] - target_size)
        )
        if closest_mem in seen_mem:
            continue
        seen_mem.add(closest_mem)
        selected.append((closest_mem, closest_time))

    return sorted(selected)


def filter_points_for_display(
    points: list[tuple[float, float]], use_raw_mem: bool
) -> list[tuple[float, float]]:
    if use_raw_mem:
        return sorted(points)

    sorted_points = sorted(points)
    in_window = [
        (mem, time)
        for mem, time in sorted_points
        if DEFAULT_RELATIVE_SIZE_MIN <= mem <= DEFAULT_RELATIVE_SIZE_MAX
    ]
    below_window = [
        (mem, time) for mem, time in sorted_points if mem < DEFAULT_RELATIVE_SIZE_MIN
    ]
    above_window = [
        (mem, time) for mem, time in sorted_points if mem > DEFAULT_RELATIVE_SIZE_MAX
    ]

    selected_points = list(in_window)
    if below_window:
        selected_points.append(below_window[-1])
    if above_window:
        selected_points.append(above_window[0])

    return sorted(dict.fromkeys(selected_points))


def select_nearest_point(
    points: list[tuple[float, float]], target_size: float, use_raw_mem: bool = False
) -> tuple[float, float] | None:
    eligible_points = filter_points_for_display(points, use_raw_mem=use_raw_mem)
    if not eligible_points:
        return None
    return min(eligible_points, key=lambda item: abs(item[0] - target_size))


def compute_practical_problem_size(
    unconst_points: list[tuple[float, float]],
    mapper_points: list[tuple[float, float]],
) -> float | None:
    candidate_mems = sorted(
        {
            mem
            for mem, _time in unconst_points
            if interpolate_time(mapper_points, mem) is not None
        }
        | {
            mem
            for mem, _time in mapper_points
            if interpolate_time(unconst_points, mem) is not None
        }
    )
    if not candidate_mems:
        return None

    ratio_samples: list[tuple[float, float]] = []
    for mem in candidate_mems:
        unconst_time = interpolate_time(unconst_points, mem)
        mapper_time = interpolate_time(mapper_points, mem)
        if (
            unconst_time is None
            or mapper_time is None
            or unconst_time <= 0
            or mapper_time <= 0
        ):
            continue
        ratio_samples.append((mem, mapper_time / unconst_time))

    if not ratio_samples:
        return None

    max_qualifying_mem: float | None = None
    for mem, ratio in ratio_samples:
        if ratio <= THRESHOLD:
            max_qualifying_mem = mem

    for (left_mem, left_ratio), (right_mem, right_ratio) in zip(
        ratio_samples, ratio_samples[1:]
    ):
        left_qualifies = left_ratio <= THRESHOLD
        right_qualifies = right_ratio <= THRESHOLD
        if left_qualifies == right_qualifies or left_ratio == right_ratio:
            continue

        fraction = (THRESHOLD - left_ratio) / (right_ratio - left_ratio)
        crossing_mem = left_mem + fraction * (right_mem - left_mem)
        if left_qualifies:
            max_qualifying_mem = max(
                crossing_mem,
                max_qualifying_mem if max_qualifying_mem is not None else crossing_mem,
            )

    return max_qualifying_mem


def build_mapper_summary_csv(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    zero_eviction_points: dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    included_mappers: list[str],
    output_path: Path,
) -> Path:
    def format_optional_float(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    def compute_percent_increase(value: float | None, baseline: float | None) -> str:
        if value is None or baseline is None or baseline <= 0:
            return "n/a"
        return f"{((value / baseline) - 1.0) * 100.0:.2f}"

    fieldnames = [
        "graph",
        "interior",
        "boundary",
        "mapper",
        "practical_problem_size",
        "largest_problem_size_eviction_0",
        "rl_practical_problem_size_increase_pct_over_best_non_rl",
        "rl_largest_problem_size_eviction_0_increase_pct_over_best_non_rl",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for group_key, series in sorted(grouped_rows.items()):
            unconst_points = series.get(UNCONST_MAPPER, [])
            zero_eviction = zero_eviction_points.get(group_key, {})
            practical_sizes_by_mapper: dict[str, float | None] = {}
            zero_eviction_sizes_by_mapper: dict[str, float | None] = {}

            for mapper in included_mappers:
                if mapper == UNCONST_MAPPER:
                    continue
                mapper_points = series.get(mapper)
                if not mapper_points:
                    continue

                practical_problem_size = (
                    compute_practical_problem_size(unconst_points, mapper_points)
                    if unconst_points
                    else None
                )
                zero_eviction_point = zero_eviction.get(mapper)
                practical_sizes_by_mapper[mapper] = practical_problem_size
                zero_eviction_sizes_by_mapper[mapper] = (
                    None if zero_eviction_point is None else zero_eviction_point[0]
                )

            non_rl_practical_sizes = [
                value
                for mapper, value in practical_sizes_by_mapper.items()
                if mapper != RL_LABEL and value is not None
            ]
            non_rl_zero_eviction_sizes = [
                value
                for mapper, value in zero_eviction_sizes_by_mapper.items()
                if mapper != RL_LABEL and value is not None
            ]
            best_non_rl_practical_size = (
                max(non_rl_practical_sizes) if non_rl_practical_sizes else None
            )
            best_non_rl_zero_eviction_size = (
                max(non_rl_zero_eviction_sizes) if non_rl_zero_eviction_sizes else None
            )

            for mapper in included_mappers:
                if mapper == UNCONST_MAPPER:
                    continue
                if mapper not in practical_sizes_by_mapper:
                    continue

                practical_problem_size = practical_sizes_by_mapper[mapper]
                zero_eviction_size = zero_eviction_sizes_by_mapper[mapper]
                writer.writerow(
                    {
                        "graph": group_key[0],
                        "interior": group_key[1],
                        "boundary": group_key[2],
                        "mapper": mapper,
                        "practical_problem_size": format_optional_float(
                            practical_problem_size
                        ),
                        "largest_problem_size_eviction_0": format_optional_float(
                            zero_eviction_size
                        ),
                        "rl_practical_problem_size_increase_pct_over_best_non_rl": (
                            compute_percent_increase(
                                practical_problem_size,
                                best_non_rl_practical_size,
                            )
                            if mapper == RL_LABEL
                            else "n/a"
                        ),
                        "rl_largest_problem_size_eviction_0_increase_pct_over_best_non_rl": (
                            compute_percent_increase(
                                zero_eviction_size,
                                best_non_rl_zero_eviction_size,
                            )
                            if mapper == RL_LABEL
                            else "n/a"
                        ),
                    }
                )

    return output_path


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
    n_gpus: int,
    include_inf: bool = False,
    use_raw_mem: bool = False,
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    dict[tuple[str, str, str], list[tuple[float, float]]],
    dict[tuple[str, str, str, str], tuple[tuple[str, str, str], float]],
]:
    grouped_rows: dict[tuple[str, str, str], dict[str, dict[float, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    inf_rows: dict[tuple[str, str, str], dict[float, float]] = defaultdict(dict)
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
                interior = float(row["interior"])
                boundary = float(row["boundary"])
                original_mem = float(row["mem"])
                plot_mem = (
                    original_mem
                    if use_raw_mem
                    else transform_mem(
                        float(row["single_device"]), interior, boundary, n_gpus
                    )
                )
                time = microseconds_to_seconds(float(row["time"]))
                eviction = float(row["eviction"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data in row {row_number}: {row}") from exc

            if "_inf" in mapper:
                if include_inf:
                    previous_time = inf_rows[group_key].get(plot_mem)
                    if previous_time is None or time < previous_time:
                        inf_rows[group_key][plot_mem] = time
                continue

            mapper = normalize_mapper(mapper)

            previous_time = grouped_rows[group_key][mapper].get(plot_mem)
            if previous_time is None or time < previous_time:
                grouped_rows[group_key][mapper][plot_mem] = time

            if eviction == 0:
                previous_zero_eviction_mem = zero_eviction_max_mem[group_key].get(
                    mapper
                )
                if (
                    previous_zero_eviction_mem is None
                    or plot_mem > previous_zero_eviction_mem
                ):
                    zero_eviction_max_mem[group_key][mapper] = plot_mem

            config_candidates[group_key][plot_mem] = original_mem
            lookup_key = make_base_lookup_key(
                row["graph"], row["mem"], row["interior"], row["boundary"]
            )
            base_lookup[lookup_key] = (group_key, plot_mem)

    if include_inf:
        for group_key, mem_to_time in inf_rows.items():
            if mem_to_time:
                grouped_rows[group_key]["Unconst"] = mem_to_time

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


def resolve_results_paths(
    n_gpus: int, include_rl: bool, noise: bool = False, extended: bool = False
) -> tuple[Path, Path | None]:
    repo_root = Path(__file__).resolve().parent
    results_dir = repo_root / "results" / f"{n_gpus}gpus"
    steps = 512 if extended else 256
    csv_name = (
        f"noise_level_sweep_results_{steps}.csv"
        if noise
        else f"level_sweep_results_{steps}.csv"
    )
    rl_csv_name = f"noise_results_rl_{steps}.csv" if noise else f"results_rl_{steps}.csv"
    csv_path = results_dir / csv_name
    rl_csv_path = results_dir / rl_csv_name if include_rl else None
    return csv_path, rl_csv_path


def resolve_normalization_path(n_gpus: int) -> Path:
    repo_root = Path(__file__).resolve().parent
    return repo_root / "results" / f"{n_gpus}gpus" / "normalization.csv"


def config_template_paths(n_gpus: int) -> tuple[Path, Path, Path]:
    config_dir = Path(__file__).resolve().parent / "launch_conf"
    return (
        config_dir / "train.json",
        config_dir / "norm.json",
        config_dir / "pickle_eval.json",
    )


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
                time = microseconds_to_seconds(float(row["time"]))
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


def load_normalization_baselines(
    normalization_csv_path: Path,
    base_lookup: dict[tuple[str, str, str, str], tuple[tuple[str, str, str], float]],
) -> dict[tuple[str, str, str], float]:
    baseline_mems: dict[tuple[str, str, str], float] = {}
    required_fields = {"graph", "mem", "interior", "boundary"}

    with normalization_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Normalization CSV is missing required columns: {missing_str}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                lookup_key = make_base_lookup_key(
                    row["graph"], row["mem"], row["interior"], row["boundary"]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid normalization data in row {row_number}: {row}"
                ) from exc

            matched_entry = base_lookup.get(lookup_key)
            if matched_entry is None:
                raise ValueError(
                    "Normalization entry does not match any results row: "
                    f"{row['graph']}, mem={row['mem']}, interior={row['interior']}, "
                    f"boundary={row['boundary']}"
                )
            group_key, transformed_mem = matched_entry
            baseline_mems[group_key] = transformed_mem

    return baseline_mems


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


def build_results_csv(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    included_mappers: list[str],
    output_path: Path,
) -> Path:
    fieldnames = ["graph", "interior", "boundary", "target_size", *included_mappers]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for (graph, interior, boundary), series in sorted(grouped_rows.items()):
            for target_size in RELATIVE_PROBLEM_SIZES:
                row: dict[str, str | float] = {
                    "graph": graph,
                    "interior": interior,
                    "boundary": boundary,
                    "target_size": target_size,
                }
                for mapper in included_mappers:
                    point = select_nearest_point(
                        series.get(mapper, []), target_size, use_raw_mem=False
                    )
                    row[mapper] = "" if point is None else point[1]
                writer.writerow(row)

    return output_path


def build_rl_vs_best_baseline_csv(
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    output_path: Path,
) -> Path:
    fieldnames = [
        "graph",
        "interior",
        "boundary",
        "target_size",
        "best_non_inf_non_rl_mapper",
        "best_non_inf_non_rl_time",
        "rl_time",
        "best_time_over_rl_time",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for (graph, interior, boundary), series in sorted(grouped_rows.items()):
            rl_points = series.get(RL_LABEL, [])
            if not rl_points:
                continue

            baseline_mappers = sorted(
                mapper for mapper in series if is_baseline_candidate_mapper(mapper)
            )

            for target_size in RELATIVE_PROBLEM_SIZES:
                rl_point = select_nearest_point(
                    rl_points, target_size, use_raw_mem=False
                )
                if rl_point is None:
                    continue

                best_mapper: str | None = None
                best_time: float | None = None
                for mapper in baseline_mappers:
                    point = select_nearest_point(
                        series.get(mapper, []), target_size, use_raw_mem=False
                    )
                    if point is None:
                        continue
                    _mem, mapper_time = point
                    if best_time is None or mapper_time < best_time:
                        best_time = mapper_time
                        best_mapper = mapper

                rl_time = rl_point[1]
                ratio = (
                    ""
                    if best_time is None or rl_time <= 0
                    else f"{best_time / rl_time:.2f}"
                )

                writer.writerow(
                    {
                        "graph": graph,
                        "interior": interior,
                        "boundary": boundary,
                        "target_size": f"{target_size:.2f}",
                        "best_non_inf_non_rl_mapper": (
                            "" if best_mapper is None else best_mapper
                        ),
                        "best_non_inf_non_rl_time": (
                            "" if best_time is None else f"{best_time:.2f}"
                        ),
                        "rl_time": f"{rl_time:.2f}",
                        "best_time_over_rl_time": ratio,
                    }
                )

    return output_path


def make_plot(
    n_gpus: int,
    group_key: tuple[str, str, str],
    series: dict[str, list[tuple[float, float]]],
    zero_eviction_points: dict[str, tuple[float, float]],
    output_dir: Path,
    use_raw_mem: bool = False,
    log_scale: bool = False,
    normalize_runtime: bool = False,
    graph_filename_prefix: str = "",
    y_axis_top: float | None = None,
) -> Path:
    graph, interior, boundary = group_key

    fig, ax = plt.subplots(figsize=(6, 4))
    large_font_size = FONT_SIZE + 1
    practical_problem_sizes: dict[str, float] = {}
    unconst_points = series.get(UNCONST_MAPPER)
    if unconst_points:
        for mapper, points in sorted(series.items()):
            if mapper == UNCONST_MAPPER:
                continue
            practical_problem_size = compute_practical_problem_size(
                unconst_points, points
            )
            if practical_problem_size is not None:
                practical_problem_sizes[mapper] = practical_problem_size

    for mapper, points in sorted(series.items()):
        sorted_points = filter_points_for_display(points, use_raw_mem)
        if not sorted_points:
            continue
        mem_values = [mem for mem, _time in sorted_points]
        time_values = [time for _mem, time in sorted_points]
        (line,) = ax.plot(
            mem_values,
            time_values,
            marker="o",
            linewidth=1.8,
            label=display_mapper_name(mapper),
            color=mapper_color(mapper),
        )

        marker_point = zero_eviction_points.get(mapper)
        if marker_point is not None:
            marker_mem, marker_time = marker_point
            if not use_raw_mem and marker_mem not in {mem for mem, _ in sorted_points}:
                continue
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

    for mapper, practical_problem_size in sorted(practical_problem_sizes.items()):
        ax.axvline(
            practical_problem_size,
            color=mapper_color(mapper),
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
        )

    ax.set_title(
        # f"{n_gpus}gpus graph={graph}, interior={interior}, boundary={boundary}",
        "",
        fontsize=large_font_size,
    )
    ax.set_xlabel(
        "mem" if use_raw_mem else "Relative Problem Size",
        fontsize=large_font_size,
    )
    ax.set_ylabel(
        "Normalized Runtime" if normalize_runtime else "Runtime (s)",
        fontsize=large_font_size,
    )
    ax.set_yscale("log" if log_scale else "linear")
    if log_scale:
        apply_log_yaxis_format(ax)
    if y_axis_top is not None:
        ax.set_ylim(top=y_axis_top)
    # if not use_raw_mem:
    #     ax.set_xlim(DEFAULT_RELATIVE_SIZE_MIN, DEFAULT_RELATIVE_SIZE_MAX)
    # ax.set_xlim(left=100e9,)
    ax.tick_params(axis="both", labelsize=large_font_size)
    ax.grid(True, alpha=0.3)

    if len(series) > 1:
        ax.legend(loc="upper left", fontsize=large_font_size - 4)

    fig.tight_layout()

    filename = (
        f"{graph_filename_prefix}{slugify(graph)}__interior_{slugify(interior)}"
        f"__boundary_{slugify(boundary)}.pdf"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def make_small_plot(
    n_gpus: int,
    graph: str,
    graph_groups: list[
        tuple[tuple[str, str, str], dict[str, list[tuple[float, float]]]]
    ],
    output_dir: Path,
    log_scale: bool = False,
    normalize_runtime: bool = False,
    graph_filename_prefix: str = "",
    y_axis_tops: dict[tuple[str, str, str], float] | None = None,
) -> list[Path]:
    # del n_gpus

    sorted_groups = sorted(
        graph_groups, key=lambda item: (float(item[0][1]), float(item[0][2]))
    )
    mapper_markers = ["o", "s", "^", "+", "D", "x", "*", "v"]
    mappers = sorted(
        {
            mapper
            for _group_key, series in sorted_groups
            for mapper, points in series.items()
            if select_target_points(points, RELATIVE_PROBLEM_SIZES)
        },
        key=mapper_sort_key,
    )
    mapper_to_marker = {
        mapper: mapper_markers[index % len(mapper_markers)]
        for index, mapper in enumerate(mappers)
    }

    subplot_data: list[
        tuple[tuple[str, str, str], dict[str, list[tuple[float, float]]]]
    ] = []
    y_values: list[float] = []
    for group_key, series in sorted_groups:
        selected_series: dict[str, list[tuple[float, float]]] = {}
        for mapper in mappers:
            target_points = select_target_points(
                series.get(mapper, []), RELATIVE_PROBLEM_SIZES
            )
            if not target_points:
                continue
            selected_series[mapper] = target_points
            y_values.extend(time for _mem, time in target_points)
        subplot_data.append((group_key, selected_series))

    fig, axes = plt.subplots(
        1, len(subplot_data), figsize=(3.1 * len(subplot_data), 2.8), sharey=True
    )
    if len(subplot_data) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    for index, ((_, interior, boundary), series) in enumerate(subplot_data):
        ax = axes[index]
        for mapper in mappers:
            target_points = series.get(mapper)
            if not target_points:
                continue
            (line,) = ax.plot(
                [mem for mem, _ in target_points],
                [time for _, time in target_points],
                marker=mapper_to_marker[mapper],
                color=mapper_color(mapper),
                linewidth=2.0,
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.2,
                label=display_mapper_name(mapper),
            )
            if mapper not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(mapper)

        ax.set_title(rf"$C_I = {interior},\; C_B = {boundary}$", fontsize=FONT_SIZE)
        ax.set_xlabel("Problem Size", fontsize=FONT_SIZE)
        ax.set_xticks(RELATIVE_PROBLEM_SIZES)
        ax.set_yscale("log" if log_scale else "linear")
        if log_scale:
            apply_log_yaxis_format(ax)
        if y_axis_tops is not None:
            y_axis_top = y_axis_tops.get((graph, interior, boundary))
            if y_axis_top is not None:
                ax.set_ylim(top=y_axis_top)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if index == 0:
            ax.set_ylabel(
                "Normalized Runtime" if normalize_runtime else "Runtime (s)",
                fontsize=FONT_SIZE,
            )
        else:
            ax.set_ylabel("")

    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        if log_scale:
            lower = y_min * 0.9
            upper = y_max * 1.1
        else:
            margin = (y_max - y_min) * 0.05 if y_max > y_min else max(y_max * 0.05, 1.0)
            lower = max(0.0, y_min - margin)
            upper = y_max + margin
        for ax in axes:
            current_top = ax.get_ylim()[1]
            if y_axis_tops is not None:
                ax.set_ylim(lower, min(upper, current_top))
            else:
                ax.set_ylim(lower, upper)

    legend = None
    if legend_handles:
        legend = fig.legend(
            legend_handles,
            [display_mapper_name(mapper) for mapper in legend_labels],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=FONT_SIZE,
        )

    plt.subplots_adjust(wspace=0.16)

    output_paths: list[Path] = []
    output_with_legend = (
        output_dir / f"{n_gpus}gpus_{graph_filename_prefix}{slugify(graph)}.pdf"
    )
    fig.savefig(output_with_legend, dpi=200, bbox_inches="tight")
    output_paths.append(output_with_legend)

    if legend is not None:
        legend.remove()

    output_without_legend = (
        output_dir
        / f"{n_gpus}gpus_{graph_filename_prefix}{slugify(graph)}_no_legend.pdf"
    )
    fig.savefig(output_without_legend, dpi=200, bbox_inches="tight")
    output_paths.append(output_without_legend)

    plt.close(fig)
    return output_paths


def prepare_plot_data(
    csv_path: Path,
    rl_csv_path: Path | None,
    normalization_csv_path: Path | None,
    n_gpus: int,
    args: argparse.Namespace,
    baseline_mems: dict[tuple[str, str, str], float] | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    dict[tuple[str, str, str], list[tuple[float, float]]],
    dict[tuple[str, str, str], float] | None,
    dict[tuple[str, str, str], float] | None,
]:
    grouped_rows, zero_eviction_points, config_points, base_lookup = read_rows(
        csv_path, n_gpus, include_inf=args.inf, use_raw_mem=args.raw
    )
    if not grouped_rows:
        raise ValueError(f"No data rows found in CSV: {csv_path}")
    if rl_csv_path is not None:
        merge_optional_rl_results(
            rl_csv_path, grouped_rows, zero_eviction_points, base_lookup
        )
    if args.extend and baseline_mems is None:
        if normalization_csv_path is None:
            raise ValueError("Missing normalization CSV path for --extend mode.")
        baseline_mems = load_normalization_baselines(normalization_csv_path, base_lookup)

    runtime_normalizers: dict[tuple[str, str, str], float] | None = None
    normalized_rows_for_runtime: (
        dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]] | None
    ) = None
    if not args.raw:
        (
            grouped_rows,
            zero_eviction_points,
            config_points,
            baseline_mems,
        ) = normalize_problem_sizes(
            grouped_rows,
            zero_eviction_points,
            config_points,
            baseline_mems=baseline_mems,
        )
        normalized_rows_for_runtime = grouped_rows
    elif args.norm:
        (
            normalized_rows_for_runtime,
            _,
            _,
            baseline_mems,
        ) = normalize_problem_sizes(
            grouped_rows,
            zero_eviction_points,
            config_points,
            baseline_mems=baseline_mems,
        )

    if normalized_rows_for_runtime is not None:
        runtime_normalizers = compute_runtime_normalizers(normalized_rows_for_runtime)

    if runtime_normalizers is not None:
        y_axis_tops = {
            group_key: LIMIT_Y if args.norm else LIMIT_Y * normalizer
            for group_key, normalizer in runtime_normalizers.items()
        }
        if args.norm:
            grouped_rows, zero_eviction_points = normalize_runtimes(
                grouped_rows, zero_eviction_points, runtime_normalizers
            )
    else:
        y_axis_tops = None

    return grouped_rows, zero_eviction_points, config_points, baseline_mems, y_axis_tops


def emit_plot_pdfs(
    n_gpus: int,
    grouped_rows: dict[tuple[str, str, str], dict[str, list[tuple[float, float]]]],
    zero_eviction_points: dict[tuple[str, str, str], dict[str, tuple[float, float]]],
    large_output_dir: Path,
    small_output_dir: Path,
    args: argparse.Namespace,
    graph_filename_prefix: str = "",
    y_axis_tops: dict[tuple[str, str, str], float] | None = None,
) -> tuple[list[Path], list[Path]]:
    large_output_paths: list[Path] = []
    for group_key, series in sorted(grouped_rows.items()):
        large_output_paths.append(
            make_plot(
                n_gpus,
                group_key,
                series,
                zero_eviction_points.get(group_key, {}),
                large_output_dir,
                use_raw_mem=args.raw,
                log_scale=args.log,
                normalize_runtime=args.norm,
                graph_filename_prefix=graph_filename_prefix,
                y_axis_top=y_axis_tops.get(group_key)
                if y_axis_tops is not None
                else None,
            )
        )

    small_groups_by_graph: dict[
        str, list[tuple[tuple[str, str, str], dict[str, list[tuple[float, float]]]]]
    ] = defaultdict(list)
    for group_key, series in sorted(grouped_rows.items()):
        small_groups_by_graph[group_key[0]].append((group_key, series))

    small_output_paths: list[Path] = []
    for graph, graph_groups in sorted(small_groups_by_graph.items()):
        small_output_paths.extend(
            make_small_plot(
                n_gpus,
                graph,
                graph_groups,
                small_output_dir,
                log_scale=args.log,
                normalize_runtime=args.norm,
                graph_filename_prefix=graph_filename_prefix,
                y_axis_tops=y_axis_tops,
            )
        )

    return large_output_paths, small_output_paths


def run_for_ngpus(args: argparse.Namespace, n_gpus: int) -> None:
    included_mappers = parse_included_mappers(args.mappers)
    include_rl = args.rl
    if not include_rl:
        included_mappers.discard(RL_LABEL)
    csv_path, rl_csv_path = resolve_results_paths(
        n_gpus, include_rl, noise=False, extended=args.extend
    )
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if rl_csv_path is not None:
        rl_csv_path = rl_csv_path.expanduser().resolve()
        if not rl_csv_path.is_file():
            raise FileNotFoundError(f"RL CSV file not found: {rl_csv_path}")
    normalization_csv_path: Path | None = None
    if args.extend:
        normalization_csv_path = resolve_normalization_path(n_gpus).expanduser().resolve()
        if not normalization_csv_path.is_file():
            raise FileNotFoundError(
                f"Normalization CSV file not found: {normalization_csv_path}"
            )

    noise_csv_path: Path | None = None
    noise_rl_csv_path: Path | None = None
    if args.noise:
        noise_csv_path, noise_rl_csv_path = resolve_results_paths(
            n_gpus, include_rl, noise=True, extended=args.extend
        )
        noise_csv_path = noise_csv_path.expanduser().resolve()
        if not noise_csv_path.is_file():
            raise FileNotFoundError(f"Noise CSV file not found: {noise_csv_path}")
        if noise_rl_csv_path is not None:
            noise_rl_csv_path = noise_rl_csv_path.expanduser().resolve()
            if not noise_rl_csv_path.is_file():
                raise FileNotFoundError(
                    f"Noise RL CSV file not found: {noise_rl_csv_path}"
                )

    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "plots" / f"{n_gpus}gpus"
    large_output_dir = output_dir / "large"
    small_output_dir = output_dir / "small"
    reports_output_dir = output_dir / "reports"
    large_output_dir.mkdir(parents=True, exist_ok=True)
    small_output_dir.mkdir(parents=True, exist_ok=True)
    reports_output_dir.mkdir(parents=True, exist_ok=True)
    train_config_template, norm_config_template, pickle_eval_config_template = (
        config_template_paths(n_gpus)
    )
    for template_path in (
        train_config_template,
        norm_config_template,
        pickle_eval_config_template,
    ):
        if not template_path.is_file():
            raise FileNotFoundError(f"Config template not found: {template_path}")

    grouped_rows, zero_eviction_points, config_points, baseline_mems, y_axis_tops = (
        prepare_plot_data(
            csv_path, rl_csv_path, normalization_csv_path, n_gpus, args
        )
    )
    mapper_summary_csv_path = build_mapper_summary_csv(
        grouped_rows,
        zero_eviction_points,
        sorted(included_mappers),
        reports_output_dir / "mapper_summary.csv",
    )
    rl_vs_best_baseline_csv_path = build_rl_vs_best_baseline_csv(
        grouped_rows,
        reports_output_dir / "rl_vs_best_non_inf_non_rl_baseline.csv",
    )
    grouped_rows, zero_eviction_points = filter_series(
        grouped_rows, zero_eviction_points, included_mappers
    )
    if not grouped_rows:
        mapper_str = ", ".join(sorted(included_mappers))
        raise ValueError(f"No data rows found after filtering mappers: {mapper_str}")

    output_paths, small_output_paths = emit_plot_pdfs(
        n_gpus,
        grouped_rows,
        zero_eviction_points,
        large_output_dir,
        small_output_dir,
        args,
        y_axis_tops=y_axis_tops,
    )

    config_paths = [
        build_config_from_template(
            config_points, train_config_template, "train.json", reports_output_dir
        ),
        build_config_from_template(
            config_points, norm_config_template, "norm.json", reports_output_dir
        ),
        build_config_from_template(
            config_points,
            pickle_eval_config_template,
            "pickle_eval.json",
            reports_output_dir,
        ),
    ]
    results_csv_path = build_results_csv(
        grouped_rows,
        sorted(included_mappers),
        reports_output_dir / "results.csv",
    )

    if args.noise:
        if baseline_mems is None:
            raise ValueError("Missing cached standard problem size baselines.")
        noise_grouped_rows, noise_zero_eviction_points, _, _, noise_y_axis_tops = (
            prepare_plot_data(
                noise_csv_path,
                noise_rl_csv_path,
                normalization_csv_path,
                n_gpus,
                args,
                baseline_mems=baseline_mems,
            )
        )
        noise_grouped_rows, noise_zero_eviction_points = filter_series(
            noise_grouped_rows, noise_zero_eviction_points, included_mappers
        )
        if not noise_grouped_rows:
            mapper_str = ", ".join(sorted(included_mappers))
            raise ValueError(
                f"No noise data rows found after filtering mappers: {mapper_str}"
            )
        noise_output_paths, noise_small_output_paths = emit_plot_pdfs(
            n_gpus,
            noise_grouped_rows,
            noise_zero_eviction_points,
            large_output_dir,
            small_output_dir,
            args,
            graph_filename_prefix="n",
            y_axis_tops=noise_y_axis_tops,
        )
        output_paths.extend(noise_output_paths)
        small_output_paths.extend(noise_small_output_paths)

    print(f"Generated {len(output_paths)} large plot(s) in {large_output_dir}")
    for output_path in output_paths:
        print(output_path)
    print(f"Generated {len(small_output_paths)} small plot(s) in {small_output_dir}")
    for output_path in small_output_paths:
        print(output_path)
    print(results_csv_path)
    print(mapper_summary_csv_path)
    print(rl_vs_best_baseline_csv_path)
    for config_path in config_paths:
        print(config_path)


def main() -> None:
    args = parse_args()
    for n_gpus in parse_ngpus_list(args.ngpus):
        run_for_ngpus(args, n_gpus)


if __name__ == "__main__":
    main()
