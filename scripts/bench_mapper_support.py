from __future__ import annotations

import math
import warnings
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
    "enhanced_darts",
    # Deprecated alias retained for CLI backward compatibility.
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
    """Configuration for the new DARTSMapper.

    Active fields:
        short_horizon_threshold
        medium_horizon_threshold

    All remaining fields are legacy/deprecated and retained only so older
    scripts can still instantiate DARTSConfig without failing.
    """
    # New DARTS knobs.
    short_horizon_threshold: int = 2
    medium_horizon_threshold: int = 4
    emit_short_horizon: bool = True
    emit_medium_horizon: bool = True
    short_horizon_k: int = 1
    medium_horizon_k: int = 1

    # Deprecated legacy knobs retained for backward compatibility. The new
    # DARTS mapper ignores these, but we use mapped/reserved thresholds to
    # derive short/medium horizon defaults when explicitly set.
    mapped_threshold: int = 0
    reserved_threshold: int = -1
    # Frontier / batch emission (enabled by default — sweet spot from sweep)
    extended_frontier: bool = True
    extended_batch: bool = True
    extended_batch_cap: int = 2  # cap=2 consistently outperformed cap=4 in sweeps
    finish_time_aware: bool = False
    # Push-pipeline mode: number of cascade passes per device per trigger.
    # 0 = classical DARTS (governed by cascade_passes + intra_window_coordination).
    # >0 = fill the pipeline with this many blocks per device per trigger;
    #      cross-device IWC is implicitly enabled to prevent N-fold data duplication.
    push_pipeline_depth: int = 0
    # Global EFT batch mode: task-first EFT with planned-data tracking across
    # all selected devices.  Matches DequeueEFTMapper quality in abundant-memory
    # regimes while keeping DeviceThreshold transition semantics.
    global_eft_batch: bool = False
    # Per-device task cap for global_eft_batch (1=safe, 4-16=pipelined).
    global_eft_batch_cap: int = 1
    # All-devices mode: considers ALL devices per task (not just idle ones).
    # Also persists dev_eft across triggers.  Matches DequeueEFTMapper quality.
    global_eft_all_devices: bool = False
    # Classical DARTS mode: one device per trigger (matches StarPU's per-GPU model).
    single_device_per_trigger: bool = False
    # Pipeline-depth mode (disabled by default)
    pipeline_depth: int = 0
    starvation_threshold: int = 1
    max_in_flight: int = 0


@dataclass(frozen=True)
class EnhancedDARTSConfig:
    """Configuration for EnhancedDARTSMapper.

    This mirrors the C++ EnhancedDARTSMapper::Config struct.
    """
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
        if mapper_name == "darts_extended":
            warnings.warn(
                "Mapper 'darts_extended' is deprecated; use 'darts' with DARTSConfig thresholds.",
                DeprecationWarning,
                stacklevel=2,
            )
        cfg = darts_config or DARTSConfig()
        sh, mh = _resolve_darts_thresholds(cfg)
        parts = [f"sh={sh}", f"mh={mh}"]
        if cfg.emit_short_horizon:
            parts.append(f"esh=True,shk={cfg.short_horizon_k}")
        if cfg.emit_medium_horizon:
            parts.append(f"emh=True,mhk={cfg.medium_horizon_k}")
        return f"DARTSMapper({','.join(parts)})"
    if mapper_name == "enhanced_darts":
        cfg = darts_config if isinstance(darts_config, EnhancedDARTSConfig) else EnhancedDARTSConfig()
        parts = [
            f"sh={cfg.short_horizon_threshold}",
            f"mh={cfg.medium_horizon_threshold}",
        ]
        if cfg.emit_short_horizon:
            parts.append(f"esh=True,shk={cfg.short_horizon_k}")
        if cfg.emit_medium_horizon:
            parts.append(f"emh=True,mhk={cfg.medium_horizon_k}")
        if not cfg.finish_time_aware:
            parts.append("ft=False")
        if not cfg.local_data_first:
            parts.append("ldf=False")
        return f"EnhancedDARTSMapper({','.join(parts)})"
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
    sh, mh = _resolve_darts_thresholds(cfg)
    mapper.short_horizon_threshold = sh
    mapper.medium_horizon_threshold = mh
    mapper.emit_short_horizon = cfg.emit_short_horizon
    mapper.emit_medium_horizon = cfg.emit_medium_horizon
    mapper.short_horizon_k = cfg.short_horizon_k
    mapper.medium_horizon_k = cfg.medium_horizon_k
    return mapper


def make_enhanced_darts_mapper(cfg: "EnhancedDARTSConfig") -> "fastsim.EnhancedDARTSMapper":
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
        if mapper_name == "darts_extended":
            warnings.warn(
                "Mapper 'darts_extended' is deprecated; use 'darts'.",
                DeprecationWarning,
                stacklevel=2,
            )
        if darts_config is not None:
            return make_darts_mapper(darts_config)
        return make_darts_mapper(DARTSConfig())
    if mapper_name == "enhanced_darts":
        cfg = darts_config if isinstance(darts_config, EnhancedDARTSConfig) else EnhancedDARTSConfig()
        mapper_cls = getattr(fastsim, "EnhancedDARTSMapper", None)
        if mapper_cls is None:
            raise RuntimeError("EnhancedDARTSMapper is not available in this build")
        return make_enhanced_darts_mapper(cfg)
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
    raise ValueError(f"Unsupported transition condition kind '{transition_config.kind}'")


def _resolve_darts_thresholds(cfg: "DARTSConfig") -> tuple[int, int]:
    short_horizon = cfg.short_horizon_threshold
    medium_horizon = cfg.medium_horizon_threshold

    legacy_mapped_used = cfg.mapped_threshold != 0
    legacy_reserved_used = cfg.reserved_threshold >= 0
    if legacy_mapped_used or legacy_reserved_used:
        warnings.warn(
            "DARTSConfig mapped/reserved thresholds are deprecated; use short_horizon_threshold and medium_horizon_threshold.",
            DeprecationWarning,
            stacklevel=3,
        )
        short_horizon = max(1, cfg.mapped_threshold + 1) if cfg.mapped_threshold >= 0 else short_horizon
        if cfg.reserved_threshold >= 0:
            medium_horizon = max(short_horizon + 1, cfg.reserved_threshold + 1)
        elif medium_horizon <= short_horizon:
            medium_horizon = short_horizon + 1

    return short_horizon, medium_horizon


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
