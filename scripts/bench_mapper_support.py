from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

import task4feedback.fastsim2 as fastsim
from task4feedback.graphs.mesh.partition import block_cyclic, col_cyclic, ij_partition, row_cyclic


INTERNAL_MAPPER_NAMES = (
    "dequeue_eft",
    "memory_aware_eft",
    "kahypar",
    "metis",
    "darts",
    "darts_extended",
)

EXTERNAL_MAPPER_NAMES = (
    "block_cyclic",
    "row_cyclic",
    "col_cyclic",
    "checkerboard",
    "reference",
)

ALL_MAPPER_NAMES = INTERNAL_MAPPER_NAMES + EXTERNAL_MAPPER_NAMES


@dataclass(frozen=True)
class ExternalMapperConfig:
    block_rows: int = 1
    block_cols: int = 1
    processor_rows: Optional[int] = None
    processor_cols: Optional[int] = None
    offset: int = 1


@dataclass(frozen=True)
class TransitionConfig:
    kind: str = "auto"
    batch_size: int = 5
    queue_threshold: int = 5
    max_in_flight: Optional[int] = None
    mapped_threshold: int = 0
    reserved_threshold: int = -1
    mapped_reserved_gap: int = 5
    reserved_launched_gap: int = 5
    total_in_flight: Optional[int] = None
    hysteresis_open: int = 16
    hysteresis_close: int = 36
    hysteresis_starvation: int = 2
    pipeline_depth: int = 4
    pipeline_starvation: int = 1


@dataclass(frozen=True)
class DARTSConfig:
    """Configuration for DARTSMapper variants.

    Pipeline-depth mode (pipeline_depth > 0):
        pipeline_depth      -- normal pipelining target per GPU device.
        starvation_threshold -- emergency lower bound; devices below this are
                               selected even when max_in_flight is hit (must be
                               <= pipeline_depth).  Mirrors the same field in
                               DARTSPipelineTransitionConditions.
        max_in_flight       -- global cap for device selection (0 = disabled).
                               Set to the same value as the transition condition's
                               max_in_flight so the two stay coherent.

    Legacy threshold mode (pipeline_depth == 0):
        mapped_threshold / reserved_threshold control device selection via
        DeviceThresholdState.
    """
    mapped_threshold: int = 0
    reserved_threshold: int = -1
    extended_frontier: bool = False
    extended_batch: bool = False
    extended_batch_cap: int = 4
    intra_window_coordination: bool = False
    cascade_passes: int = 3
    finish_time_aware: bool = False
    pipeline_depth: int = 0
    starvation_threshold: int = 1
    max_in_flight: int = 0


@dataclass(frozen=True)
class MemoryAwareEFTConfig:
    alpha: float = 1.0
    eviction_cost_location_state: str = "launched"
    overflow_state: str = "launched"
    overflow_mode: str = "full_spill"


class CellBasedExternalMapper:
    def __init__(
        self,
        mapper: Optional["CellBasedExternalMapper"] = None,
        *,
        cell_to_device: Optional[dict[int, int]] = None,
    ):
        if mapper is not None:
            self.cell_to_device = dict(mapper.cell_to_device)
        elif cell_to_device is not None:
            self.cell_to_device = dict(cell_to_device)
        else:
            self.cell_to_device = {}

    def map_tasks(self, simulator) -> list[fastsim.Action]:
        candidates = torch.full((simulator.input.top_k_candidates,), -1, dtype=torch.int64)
        count = simulator.get_mappable_candidates(candidates)
        graph = simulator.input.graph
        state = simulator.state
        actions: list[fastsim.Action] = []
        for local_index in range(count):
            task_id = int(candidates[local_index].item())
            cell_id = int(graph.task_to_cell[task_id])
            device = int(self.cell_to_device[cell_id])
            priority = state.get_mapping_priority(task_id)
            actions.append(fastsim.Action(local_index, device, priority, priority))
        return actions


def mapper_label(
    name: str,
    *,
    memory_aware_eft_alpha: float = 1.0,
    memory_aware_eft_config: Optional[MemoryAwareEFTConfig] = None,
    darts_config: Optional["DARTSConfig"] = None,
) -> str:
    mapper_name = name.lower()
    if mapper_name == "dequeue_eft":
        return "DequeueEFTMapper"
    if mapper_name == "memory_aware_eft":
        config = memory_aware_eft_config or MemoryAwareEFTConfig(alpha=memory_aware_eft_alpha)
        return (
            "MemoryAwareEFTMapper("
            f"alpha={config.alpha:g},"
            f"locations={config.eviction_cost_location_state},"
            f"overflow_state={config.overflow_state},"
            f"overflow_mode={config.overflow_mode})"
        )
    if mapper_name == "kahypar":
        return "KaHyParMapper"
    if mapper_name == "metis":
        return "METISMapper"
    if mapper_name in {"darts", "darts_extended"}:
        cfg = darts_config or (
            DARTSConfig(extended_frontier=True, extended_batch=True, extended_batch_cap=4)
            if mapper_name == "darts_extended"
            else DARTSConfig()
        )
        mt = cfg.mapped_threshold
        rt = cfg.reserved_threshold
        threshold_str = f"mt={mt}" if rt < 0 else f"rt={rt}"
        iwc_str = f",iwc={cfg.cascade_passes}" if cfg.intra_window_coordination else ""
        if cfg.extended_frontier:
            cap_str = f",cap={cfg.extended_batch_cap}" if cfg.extended_batch else ""
            return f"DARTSMapper(ext{'_batch' if cfg.extended_batch else ''}{cap_str},{threshold_str}{iwc_str})"
        return f"DARTSMapper({threshold_str}{iwc_str})"
    if mapper_name == "block_cyclic":
        return "ExternalBlockCyclicMapper"
    if mapper_name == "row_cyclic":
        return "ExternalRowCyclicMapper"
    if mapper_name == "col_cyclic":
        return "ExternalColCyclicMapper"
    if mapper_name == "checkerboard":
        return "ExternalCheckerboardMapper"
    if mapper_name == "reference":
        return "ExternalReferencePartitionMapper"
    raise ValueError(f"Unsupported mapper '{name}'")


def is_external_mapper(name: str) -> bool:
    return name.lower() in EXTERNAL_MAPPER_NAMES


def _resolve_memory_aware_location_state(name: str) -> int:
    location_state = name.lower()
    if location_state == "launched":
        return 0
    if location_state == "reserved":
        return 1
    if location_state == "mapped":
        return 2
    raise ValueError(f"Unsupported MemoryAwareEFT location state '{name}'")


def _resolve_memory_aware_overflow_state(name: str) -> int:
    overflow_state = name.lower()
    if overflow_state == "reserved":
        return 0
    if overflow_state == "mapped":
        return 1
    if overflow_state == "launched":
        return 2
    raise ValueError(f"Unsupported MemoryAwareEFT overflow state '{name}'")


def _resolve_memory_aware_overflow_mode(name: str) -> int:
    overflow_mode = name.lower()
    if overflow_mode == "full_spill":
        return 0
    if overflow_mode in {"incoming_only", "task_only"}:
        return 1
    raise ValueError(f"Unsupported MemoryAwareEFT overflow mode '{name}'")


def make_darts_mapper(cfg: "DARTSConfig") -> "fastsim.DARTSMapper":
    mapper = fastsim.DARTSMapper()
    mapper.mapped_threshold = cfg.mapped_threshold
    mapper.reserved_threshold = cfg.reserved_threshold
    mapper.extended_frontier_enabled = cfg.extended_frontier
    mapper.extended_batch_emission_enabled = cfg.extended_batch
    mapper.extended_batch_emission_cap = cfg.extended_batch_cap
    mapper.intra_window_coordination = cfg.intra_window_coordination
    mapper.cascade_passes = cfg.cascade_passes
    mapper.finish_time_aware = cfg.finish_time_aware
    mapper.pipeline_depth = cfg.pipeline_depth
    mapper.starvation_threshold = cfg.starvation_threshold
    mapper.max_in_flight = cfg.max_in_flight
    return mapper


def make_internal_mapper(
    name: str,
    *,
    memory_aware_eft_alpha: float = 1.0,
    memory_aware_eft_config: Optional[MemoryAwareEFTConfig] = None,
    darts_config: Optional["DARTSConfig"] = None,
):
    mapper_name = name.lower()
    if mapper_name == "dequeue_eft":
        return fastsim.DequeueEFTMapper()
    if mapper_name == "memory_aware_eft":
        config = memory_aware_eft_config or MemoryAwareEFTConfig(alpha=memory_aware_eft_alpha)
        mapper = fastsim.MemoryAwareEFTMapper()
        mapper.alpha = config.alpha
        mapper.eviction_cost_location_state = _resolve_memory_aware_location_state(
            config.eviction_cost_location_state
        )
        mapper.overflow_state = _resolve_memory_aware_overflow_state(config.overflow_state)
        mapper.overflow_mode = _resolve_memory_aware_overflow_mode(config.overflow_mode)
        return mapper
    if mapper_name == "kahypar":
        return fastsim.KaHyParMapper()
    if mapper_name == "metis":
        mapper_cls = getattr(fastsim, "METISMapper", None)
        if mapper_cls is None:
            raise RuntimeError("METISMapper is not available in this build")
        return mapper_cls()
    if mapper_name in {"darts", "darts_extended"}:
        if darts_config is not None:
            return make_darts_mapper(darts_config)
        # darts: base frontier, no batch emission (classic)
        # darts_extended: extended frontier + batch emission cap=2 (sweet spot from sweep)
        if mapper_name == "darts":
            return make_darts_mapper(DARTSConfig())
        return make_darts_mapper(
            DARTSConfig(extended_frontier=True, extended_batch=True, extended_batch_cap=2)
        )
    raise ValueError(f"Unsupported internal mapper '{name}'")


def make_external_mapper(
    name: str,
    graph,
    *,
    n_gpu_devices: int,
    config: Optional[ExternalMapperConfig] = None,
) -> CellBasedExternalMapper:
    mapper_name = name.lower()
    external_config = config or ExternalMapperConfig()
    geometry = graph.data.geometry
    offset = external_config.offset

    if mapper_name == "row_cyclic":
        partition = row_cyclic(geometry, n_parts=n_gpu_devices)
    elif mapper_name == "col_cyclic":
        partition = col_cyclic(geometry, n_parts=n_gpu_devices)
    elif mapper_name == "block_cyclic":
        proc_rows, proc_cols = _resolve_processor_grid(
            n_gpu_devices,
            external_config.processor_rows,
            external_config.processor_cols,
        )
        partition = block_cyclic(
            geometry,
            n_row_parts=proc_rows,
            n_col_parts=proc_cols,
            parts_per_row=external_config.block_rows,
            parts_per_column=external_config.block_cols,
            n_devices=n_gpu_devices,
        )
    elif mapper_name == "checkerboard":
        partition = _checkerboard_partition(geometry, n_gpu_devices)
    elif mapper_name == "reference":
        if not hasattr(graph, "reference_partition"):
            raise ValueError("Graph does not expose a reference_partition")
        partition = [int(part % n_gpu_devices) for part in graph.reference_partition]
    else:
        raise ValueError(f"Unsupported external mapper '{name}'")

    cell_to_device = {cell_id: int(part) + offset for cell_id, part in enumerate(partition)}
    return CellBasedExternalMapper(cell_to_device=cell_to_device)


def make_transition_conditions(
    mapper_name: str,
    *,
    top_k_candidates: int,
    config: Optional[TransitionConfig] = None,
):
    transition_config = config or TransitionConfig()
    kind = transition_config.kind.lower()
    if kind == "auto":
        if mapper_name.lower() in {"darts", "darts_extended"}:
            kind = "device_threshold"
        else:
            kind = "batch"

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
    if kind == "device_threshold":
        return fastsim.DeviceThresholdTransitionConditions(
            transition_config.mapped_threshold,
            transition_config.reserved_threshold,
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
    if kind == "darts_adaptive":
        max_mapped = (
            top_k_candidates
            if transition_config.max_in_flight is None
            else transition_config.max_in_flight
        )
        return fastsim.DARTSAdaptiveTransitionConditions(
            transition_config.reserved_threshold,
            max_mapped,
            transition_config.pipeline_starvation,
        )
    if kind == "darts_pipeline":
        max_in_flight = (
            top_k_candidates
            if transition_config.max_in_flight is None
            else transition_config.max_in_flight
        )
        return fastsim.DARTSPipelineTransitionConditions(
            transition_config.pipeline_depth,
            max_in_flight,
            transition_config.pipeline_starvation,
        )
    raise ValueError(f"Unsupported transition condition kind '{transition_config.kind}'")


def _resolve_processor_grid(
    n_gpu_devices: int,
    requested_rows: Optional[int],
    requested_cols: Optional[int],
) -> tuple[int, int]:
    if requested_rows is not None and requested_cols is not None:
        return requested_rows, requested_cols
    if requested_rows is not None:
        return requested_rows, max(1, math.ceil(n_gpu_devices / requested_rows))
    if requested_cols is not None:
        return max(1, math.ceil(n_gpu_devices / requested_cols)), requested_cols

    rows = int(math.floor(math.sqrt(n_gpu_devices)))
    while rows > 1 and n_gpu_devices % rows != 0:
        rows -= 1
    cols = max(1, math.ceil(n_gpu_devices / rows))
    return rows, cols


def _checkerboard_partition(geometry, n_gpu_devices: int) -> list[int]:
    _, _, row_keys, col_keys, ij_map = ij_partition(geometry, round=2)
    partition = [0] * len(geometry.cells)
    for row_index, row_key in enumerate(row_keys):
        for col_index, col_key in enumerate(col_keys):
            if n_gpu_devices == 2:
                part = (row_index + col_index) & 1
            elif n_gpu_devices == 4:
                part = (row_index & 1) * 2 + (col_index & 1)
            else:
                part = (row_index + col_index) % n_gpu_devices
            for cell_id in ij_map[(row_key, col_key)]:
                partition[cell_id] = part
    return partition
