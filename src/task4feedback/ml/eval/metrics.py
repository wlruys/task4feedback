from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List

import numpy as np

from task4feedback.graphs.mesh import plot as mesh_plot
from task4feedback.logging import training

ComputeFn = Callable[["RuntimeEnv", "MetricContext", Dict[str, Any]], Dict[str, float]]


@dataclass
class MetricContext:
    """Lightweight context shared by metric computations."""

    env_times: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    rollout: Any = None


@dataclass
class MetricDef:
    id: str
    compute_fn: ComputeFn
    default_params: Dict[str, Any] = field(default_factory=dict)
    default_aggregations: List[str] = field(default_factory=lambda: ["mean"])
    description: str = ""


def _safe_agg(values: List[float], agg: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "std":
        return float(np.std(arr))
    if agg == "min":
        return float(np.min(arr))
    if agg == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown aggregation: {agg}")


def aggregate_metric_sets(
    metric_sets: List[Dict[str, float]],
    aggregations: List[str],
) -> Dict[str, float]:
    """Aggregate a list of metric dicts (same keys) with provided reductions."""
    aggregated: Dict[str, float] = {}
    if not metric_sets:
        return aggregated

    suffix_map = {
        "mean": "",
        "std": "_std",
        "min": "_min",
        "max": "_max",
    }

    keys = set().union(*[set(m.keys()) for m in metric_sets])
    for key in keys:
        values = [m[key] for m in metric_sets if key in m]
        for agg in aggregations:
            suffix = suffix_map.get(agg)
            if suffix is None:
                raise ValueError(f"Unknown aggregation: {agg}")
            out_key = f"{key}{suffix}"
            aggregated[out_key] = _safe_agg(values, agg)
    return aggregated


def _compute_mean_time(env, context: MetricContext, params: Dict[str, Any]) -> Dict[str, float]:
    times = context.env_times or ([getattr(env.simulator, "time", 0.0)] if hasattr(env, "simulator") else [])
    mean_time = float(np.mean(times)) if times else 0.0
    std_time = float(np.std(times)) if len(times) > 1 else 0.0
    return {"mean_time": mean_time, "std_time": std_time}


def _compute_vs_baseline(env, context: MetricContext, params: Dict[str, Any]) -> Dict[str, float]:
    baseline_name = params.get("baseline_name", "EFT")
    times = context.env_times or ([getattr(env.simulator, "time", 0.0)] if hasattr(env, "simulator") else [])
    mean_time = float(np.mean(times)) if times else 0.0
    if not hasattr(env, "_get_baseline"):
        return {"mean_vs_baseline": 0.0}
    baseline_time = env._get_baseline(baseline_name)
    ratio = baseline_time / mean_time if mean_time > 0 else 0.0
    return {f"mean_vs_{baseline_name}": ratio}


def _compute_load_balance(env, context: MetricContext, params: Dict[str, Any]) -> Dict[str, float]:
    try:
        static_state, _ = mesh_plot._build_state(env)  # Uses existing helper; safe for evaluation
    except Exception as exc:
        training.warning(f"Failed to build state for load balance metric: {exc}")
        return {}

    interval = int(params.get("interval_us", 0) or 0)
    lb = mesh_plot.compute_load_balance(static_state)
    metrics: Dict[str, float] = {
        "load_balance": lb.load_balance,
        "in_comm_balance": lb.in_comm_balance,
        "out_comm_balance": lb.out_comm_balance,
        "total_work": lb.total_work,
        "total_in_communication": lb.total_in_communication,
        "total_out_communication": lb.total_out_communication,
    }

    if interval > 0:
        try:
            over_time = mesh_plot.load_balance_over_time(static_state, interval)
            if over_time:
                metrics["load_balance_over_time_mean"] = float(np.mean([x.load_balance for x in over_time]))
                metrics["in_comm_balance_over_time_mean"] = float(np.mean([x.in_comm_balance for x in over_time]))
                metrics["out_comm_balance_over_time_mean"] = float(np.mean([x.out_comm_balance for x in over_time]))
        except Exception as exc:
            training.warning(f"Failed load balance over-time computation: {exc}")

    return metrics


def _compute_idle_ratio(env, context: MetricContext, params: Dict[str, Any]) -> Dict[str, float]:
    try:
        static_state, _ = mesh_plot._build_state(env)
    except Exception as exc:
        training.warning(f"Failed to build state for idle ratio metric: {exc}")
        return {}

    end_time = getattr(getattr(env, "simulator", None), "time", None)
    if end_time is None or end_time <= 0:
        return {}

    include_comm = params.get("include_comm", True)
    idle_type = mesh_plot.IdleType(compute=True, in_comm=include_comm, out_comm=include_comm)
    idle = mesh_plot.get_total_idle_time(static_state, idle_type, simulation_end_time=end_time)
    if idle.size == 0:
        return {}

    denom = float(end_time) if end_time > 0 else 1.0
    ratios = idle / denom
    return {
        "idle_ratio_mean": float(np.mean(ratios)),
        "idle_ratio_max": float(np.max(ratios)),
        "idle_ratio_min": float(np.min(ratios)),
    }


def _compute_throughput(env, context: MetricContext, params: Dict[str, Any]) -> Dict[str, float]:
    try:
        static_state, _ = mesh_plot._build_state(env)
    except Exception as exc:
        training.warning(f"Failed to build state for throughput metric: {exc}")
        return {}

    end_time = getattr(getattr(env, "simulator", None), "time", None)
    if end_time is None or end_time <= 0:
        return {"throughput": 0.0}

    completed = np.sum(static_state.ct_complete_time >= 0)
    throughput = float(completed) / float(end_time) if end_time > 0 else 0.0
    return {"throughput": throughput, "tasks_completed": float(completed)}


METRIC_REGISTRY: Dict[str, MetricDef] = {
    "mean_time": MetricDef(
        id="mean_time",
        compute_fn=_compute_mean_time,
        default_aggregations=["mean"],
        description="Mean and std of completion time across samples.",
    ),
    "vs_baseline": MetricDef(
        id="vs_baseline",
        compute_fn=_compute_vs_baseline,
        default_params={"baseline_name": "EFT"},
        default_aggregations=["mean"],
        description="Ratio to baseline (EFT by default).",
    ),
    "load_balance": MetricDef(
        id="load_balance",
        compute_fn=_compute_load_balance,
        default_params={"interval_us": 0},
        default_aggregations=["mean"],
        description="Load balance and comm balance ratios.",
    ),
    "idle_ratio": MetricDef(
        id="idle_ratio",
        compute_fn=_compute_idle_ratio,
        default_params={"include_comm": True},
        default_aggregations=["mean"],
        description="Idle ratio per device averaged over time.",
    ),
    "throughput": MetricDef(
        id="throughput",
        compute_fn=_compute_throughput,
        default_aggregations=["mean"],
        description="Tasks completed per unit simulated time.",
    ),
}


def resolve_metric_ids(configured: List[str] | None) -> List[str]:
    """Return a deduplicated list of metric ids to compute."""
    if not configured:
        return ["mean_time", "vs_baseline"]
    seen = set()
    ordered: List[str] = []
    for mid in configured:
        if mid in seen:
            continue
        seen.add(mid)
        ordered.append(mid)
    return ordered


def compute_metrics(
    metric_ids: Iterable[str],
    env,
    context: MetricContext,
    metric_params: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics by id, returning a mapping metric_id -> metric_values."""
    metric_params = metric_params or {}
    results: Dict[str, Dict[str, float]] = {}

    for metric_id in metric_ids:
        metric_def = METRIC_REGISTRY.get(metric_id)
        if metric_def is None:
            training.warning(f"Unknown metric id '{metric_id}' requested; skipping.")
            continue
        params = dict(metric_def.default_params)
        params.update(metric_params.get(metric_id, {}))
        try:
            values = metric_def.compute_fn(env, context, params)
            if values:
                results[metric_id] = values
        except Exception as exc:
            training.warning(f"Metric '{metric_id}' failed: {exc}")
    return results


def aggregate_all_metrics(
    per_run_metrics: List[Dict[str, Dict[str, float]]],
    metric_ids: Iterable[str],
    aggregation_overrides: Dict[str, List[str]] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across runs per metric id."""
    aggregation_overrides = aggregation_overrides or {}
    aggregated: Dict[str, Dict[str, float]] = {}

    for metric_id in metric_ids:
        metric_def = METRIC_REGISTRY.get(metric_id)
        if metric_def is None:
            continue
        agg_ops = aggregation_overrides.get(metric_id) or metric_def.default_aggregations

        metric_sets = [run[metric_id] for run in per_run_metrics if metric_id in run]
        aggregated[metric_id] = aggregate_metric_sets(metric_sets, agg_ops)
    return aggregated
