from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import task4feedback.fastsim2 as fastsim
from task4feedback.interface.wrappers import SimulatorDriver


@dataclass(frozen=True)
class TransitionConfig:
    kind: str = "auto"
    planned_threshold: int = 1
    max_reserved_threshold: int = 16
    batch_size: int = 5
    queue_threshold: int = 5
    max_in_flight: Optional[int] = None
    mapped_reserved_gap: int = 5
    reserved_launched_gap: int = 5
    total_in_flight: Optional[int] = None
    hysteresis_open: int = 16
    hysteresis_close: int = 36
    hysteresis_starvation: int = 2


@dataclass(frozen=True)
class DARTSConfig:
    short_horizon_threshold: int = 2
    medium_horizon_threshold: int = 4
    emit_short_horizon: bool = True
    emit_medium_horizon: bool = True
    short_horizon_k: int = 1
    medium_horizon_k: int = 1


@dataclass(frozen=True)
class EnhancedDARTSConfig:
    short_horizon_threshold: int = 4
    medium_horizon_threshold: int = 8
    emit_short_horizon: bool = True
    emit_medium_horizon: bool = True
    short_horizon_k: int = 2
    medium_horizon_k: int = 4
    finish_time_aware: bool = True
    local_data_first: bool = True


@dataclass(frozen=True)
class MemoryAwareEFTConfig:
    alpha: float = 1.0
    eviction_cost_location_state: str = "reserved"
    overflow_state: str = "reserved"
    overflow_mode: str = "incoming_only"


def _resolve_memory_aware_location_state(name: str) -> int:
    normalized = name.lower()
    if normalized == "launched":
        return fastsim.MemoryAwareLocationState.LAUNCHED
    if normalized == "reserved":
        return fastsim.MemoryAwareLocationState.RESERVED
    if normalized == "mapped":
        return fastsim.MemoryAwareLocationState.MAPPED
    raise ValueError(f"Unsupported MemoryAwareEFT location state '{name}'")


def _resolve_memory_aware_overflow_state(name: str) -> int:
    normalized = name.lower()
    if normalized == "reserved":
        return fastsim.MemoryAwareOverflowState.RESERVED
    if normalized == "mapped":
        return fastsim.MemoryAwareOverflowState.MAPPED
    if normalized == "launched":
        return fastsim.MemoryAwareOverflowState.LAUNCHED
    raise ValueError(f"Unsupported MemoryAwareEFT overflow state '{name}'")


def _resolve_memory_aware_overflow_mode(name: str) -> int:
    normalized = name.lower()
    if normalized == "full_spill":
        return fastsim.MemoryAwareOverflowMode.FULL_SPILL
    if normalized in {"incoming_only", "task_only"}:
        return fastsim.MemoryAwareOverflowMode.INCOMING_ONLY
    raise ValueError(f"Unsupported MemoryAwareEFT overflow mode '{name}'")


def make_darts_mapper(cfg: DARTSConfig | None = None) -> fastsim.DARTSMapper:
    cfg = cfg or DARTSConfig()
    mapper = fastsim.DARTSMapper()
    mapper.short_horizon_threshold = cfg.short_horizon_threshold
    mapper.medium_horizon_threshold = cfg.medium_horizon_threshold
    mapper.emit_short_horizon = cfg.emit_short_horizon
    mapper.emit_medium_horizon = cfg.emit_medium_horizon
    mapper.short_horizon_k = cfg.short_horizon_k
    mapper.medium_horizon_k = cfg.medium_horizon_k
    return mapper


def make_enhanced_darts_mapper(
    cfg: EnhancedDARTSConfig | None = None,
) -> fastsim.EnhancedDARTSMapper:
    cfg = cfg or EnhancedDARTSConfig()
    mapper = fastsim.EnhancedDARTSMapper()
    mapper.short_horizon_threshold = cfg.short_horizon_threshold
    mapper.medium_horizon_threshold = cfg.medium_horizon_threshold
    mapper.emit_short_horizon = cfg.emit_short_horizon
    mapper.emit_medium_horizon = cfg.emit_medium_horizon
    mapper.short_horizon_k = cfg.short_horizon_k
    mapper.medium_horizon_k = cfg.medium_horizon_k
    mapper.finish_time_aware = cfg.finish_time_aware
    mapper.local_data_first = cfg.local_data_first
    return mapper


def make_internal_mapper(
    name: str,
    *,
    memory_aware_eft_config: MemoryAwareEFTConfig | None = None,
    darts_config: DARTSConfig | None = None,
    enhanced_darts_config: EnhancedDARTSConfig | None = None,
):
    mapper_name = name.lower()
    if mapper_name == "dequeue_eft":
        return fastsim.DequeueEFTMapper()
    if mapper_name == "memory_aware_eft":
        config = memory_aware_eft_config or MemoryAwareEFTConfig()
        mapper = fastsim.MemoryAwareEFTMapper()
        mapper.alpha = config.alpha
        mapper.eviction_cost_location_state = _resolve_memory_aware_location_state(
            config.eviction_cost_location_state
        )
        mapper.overflow_state = _resolve_memory_aware_overflow_state(
            config.overflow_state
        )
        mapper.overflow_mode = _resolve_memory_aware_overflow_mode(config.overflow_mode)
        return mapper
    if mapper_name in {"darts", "darts_extended"}:
        return make_darts_mapper(darts_config)
    if mapper_name == "enhanced_darts":
        return make_enhanced_darts_mapper(enhanced_darts_config)
    raise ValueError(f"Unsupported internal mapper '{name}'")


def make_transition_conditions(
    mapper_name: str,
    *,
    top_k_candidates: int,
    config: TransitionConfig | None = None,
):
    transition_config = config or TransitionConfig()
    kind = transition_config.kind.lower()
    if kind == "auto":
        if mapper_name.lower() in {"darts", "darts_extended", "enhanced_darts"}:
            kind = "planned"
        else:
            kind = "hysteresis"

    if kind == "planned":
        return fastsim.PlannedThresholdTransitionConditions(
            transition_config.planned_threshold,
            transition_config.max_reserved_threshold,
        )

    if kind == "default":
        return fastsim.DefaultTransitionConditions()
    if kind == "batch":
        max_in_flight = (
            top_k_candidates
            if transition_config.max_in_flight is None
            else transition_config.max_in_flight
        )
        return fastsim.BatchTransitionConditions(
            transition_config.batch_size,
            transition_config.queue_threshold,
            max_in_flight,
        )
    if kind == "range":
        total_in_flight = (
            top_k_candidates
            if transition_config.total_in_flight is None
            else transition_config.total_in_flight
        )
        return fastsim.RangeTransitionConditions(
            transition_config.mapped_reserved_gap,
            transition_config.reserved_launched_gap,
            total_in_flight,
        )
    if kind == "hysteresis":
        return fastsim.HysteresisTransitionConditions(
            transition_config.hysteresis_open,
            transition_config.hysteresis_close,
            transition_config.hysteresis_starvation,
        )
    raise ValueError(
        f"Unsupported transition condition kind '{transition_config.kind}'"
    )


class ReplayMapper:
    """
    A mapper that replays the mapping decisions from a previous simulator execution.
    """

    def __init__(self, prev_simulator: SimulatorDriver):
        assert prev_simulator.status == fastsim.ExecutionState.COMPLETE, (
            "Previous simulator must be complete to create a replay mapper."
        )
        runtime = prev_simulator.state.get_task_runtime()
        self.history = {}
        for task_id, _ in prev_simulator.input.graph.tasks.items():
            self.history[task_id] = runtime.get_compute_task_mapped_device(task_id)

    def map_tasks(self, simulator: SimulatorDriver) -> list[fastsim.Action]:
        candidates = torch.zeros(
            (simulator.observer.graph_spec.max_candidates), dtype=torch.int64
        )
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            device = self.history[global_task_id]
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(
                global_task_id
            )
            mapping_result.append(
                fastsim.Action(i, device, mapping_priority, mapping_priority)
            )
        return mapping_result
