from collections.abc import Iterable, Iterator
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def test() -> str: ...

class UInt32Vector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: UInt32Vector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> UInt32Vector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: UInt32Vector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: UInt32Vector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class UInt64Vector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: UInt64Vector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> UInt64Vector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: UInt64Vector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: UInt64Vector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class Int32Vector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Int32Vector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Int32Vector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Int32Vector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Int32Vector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class Int64Vector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: Int64Vector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[int], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[int]: ...

    @overload
    def __getitem__(self, arg: int, /) -> int: ...

    @overload
    def __getitem__(self, arg: slice, /) -> Int64Vector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: int, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: int, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> int:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: Int64Vector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: int, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: Int64Vector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: int, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: int, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: int, /) -> None:
        """Remove first occurrence of `arg`."""

class FloatVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: FloatVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[float], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: slice, /) -> FloatVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: float, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: float, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> float:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: FloatVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: FloatVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: float, /) -> None:
        """Remove first occurrence of `arg`."""

class DoubleVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: DoubleVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[float], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[float]: ...

    @overload
    def __getitem__(self, arg: int, /) -> float: ...

    @overload
    def __getitem__(self, arg: slice, /) -> DoubleVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: float, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: float, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> float:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: DoubleVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: float, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: DoubleVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: float, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: float, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: float, /) -> None:
        """Remove first occurrence of `arg`."""

class StringVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: StringVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[str], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[str]: ...

    @overload
    def __getitem__(self, arg: int, /) -> str: ...

    @overload
    def __getitem__(self, arg: slice, /) -> StringVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: str, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: str, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> str:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: StringVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: str, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: StringVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: str, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: str, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: str, /) -> None:
        """Remove first occurrence of `arg`."""

class EventType(enum.IntEnum):
    MAPPER = 0

    RESERVER = 1

    LAUNCHER = 2

    EVICTOR = 3

    COMPUTE_COMPLETER = 4

    DATA_COMPLETER = 5

    EVICTOR_COMPLETER = 6

MAPPER: EventType = EventType.MAPPER

RESERVER: EventType = EventType.RESERVER

LAUNCHER: EventType = EventType.LAUNCHER

EVICTOR: EventType = EventType.EVICTOR

COMPUTE_COMPLETER: EventType = EventType.COMPUTE_COMPLETER

DATA_COMPLETER: EventType = EventType.DATA_COMPLETER

EVICTOR_COMPLETER: EventType = EventType.EVICTOR_COMPLETER

BYTES_IN_POWER: int = 1000

MAX_VCUS: int = 1000

MAX_TIME: int = 9223372036854775807

MAX_COPIES: int = 10

num_resource_types: int = 3

class ResourceType(enum.IntEnum):
    VCUS = 0

    MEM = 1

    TIME = 2

VCUS: ResourceType = ResourceType.VCUS

MEM: ResourceType = ResourceType.MEM

TIME: ResourceType = ResourceType.TIME

class DeviceType(enum.IntEnum):
    CPU = 1

    GPU = 2

CPU: DeviceType = DeviceType.CPU

GPU: DeviceType = DeviceType.GPU

class Resources:
    def __init__(self, vcus: int, mem: int) -> None: ...

    @property
    def vcus(self) -> int: ...

    @property
    def mem(self) -> int: ...

class ResourceEventVector:
    def __init__(self) -> None: ...

    @property
    def times(self) -> Int64Vector: ...

    @times.setter
    def times(self, arg: Int64Vector, /) -> None: ...

    @property
    def resources(self) -> Int64Vector: ...

    @resources.setter
    def resources(self, arg: Int64Vector, /) -> None: ...

class Device:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, id: int, arch: DeviceType, h2d_max_copy: int, d2d_max_copy: int, vcu: int, mem: int) -> None: ...

    @property
    def id(self) -> int: ...

    @property
    def arch(self) -> DeviceType: ...

    @property
    def h2d_max_copy(self) -> int: ...

    @property
    def d2d_max_copy(self) -> int: ...

    @property
    def max_resources(self) -> Resources: ...

    def get_mem(self) -> int: ...

    def get_vcu(self) -> int: ...

class Devices:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, n_devices: int) -> None: ...

    def create_device(self, id: int, name: str, arch: DeviceType, h2d_max_copy: int, d2d_max_copy: int, mem: int) -> None: ...

    def append_device(self, name: str, arch: DeviceType, h2d_max_copy: int, d2d_max_copy: int, mem: int) -> int: ...

    def get_device(self, id: int) -> Device: ...

    def get_device_id(self, name: str) -> int: ...

    def get_local_id(self, global_id: int) -> int: ...

    def get_global_id(self, arch: DeviceType, local_id: int) -> int: ...

    def get_type(self, id: int) -> DeviceType: ...

    def size(self) -> int: ...

    def get_name(self, id: int) -> str: ...

class TaskState(enum.IntEnum):
    SPAWNED = 1

    MAPPED = 2

    RESERVED = 4

    LAUNCHED = 8

    COMPLETED = 16

SPAWNED: TaskState = TaskState.SPAWNED

MAPPED: TaskState = TaskState.MAPPED

RESERVED: TaskState = TaskState.RESERVED

LAUNCHED: TaskState = TaskState.LAUNCHED

COMPLETED: TaskState = TaskState.COMPLETED

class TaskStatus(enum.IntEnum):
    NON = -1

    MAPPABLE = 0

    RESERVABLE = 2

    LAUNCHABLE = 4

NON: TaskStatus = TaskStatus.NON

MAPPABLE: TaskStatus = TaskStatus.MAPPABLE

RESERVABLE: TaskStatus = TaskStatus.RESERVABLE

LAUNCHABLE: TaskStatus = TaskStatus.LAUNCHABLE

class VariantVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: VariantVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[Variant], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[Variant]: ...

    @overload
    def __getitem__(self, arg: int, /) -> Variant: ...

    @overload
    def __getitem__(self, arg: slice, /) -> VariantVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: Variant, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: Variant, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> Variant:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: VariantVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: Variant, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: VariantVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

class DeviceTypeVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: DeviceTypeVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[DeviceType], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[DeviceType]: ...

    @overload
    def __getitem__(self, arg: int, /) -> DeviceType: ...

    @overload
    def __getitem__(self, arg: slice, /) -> DeviceTypeVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: DeviceType, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: DeviceType, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> DeviceType:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: DeviceTypeVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: DeviceType, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: DeviceTypeVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: DeviceType, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: DeviceType, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: DeviceType, /) -> None:
        """Remove first occurrence of `arg`."""

class Variant:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arch: DeviceType, vcu: int, mem: int, time: int) -> None: ...

    def get_arch(self) -> DeviceType: ...

    def get_vcus(self) -> int: ...

    def get_mem(self) -> int: ...

    def get_mean_duration(self) -> int: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

class Graph:
    def __init__(self) -> None: ...

    def size(self) -> int: ...

    def add_task(self, name: str) -> int: ...

    def add_read_data(self, id: int, read_data: Int32Vector) -> None: ...

    def add_write_data(self, id: int, write_data: Int32Vector) -> None: ...

    def add_retire_data(self, id: int, retire_data: Int32Vector) -> None: ...

    def set_tag(self, id: int, tag: int) -> None: ...

    def set_type(self, id: int, type: int) -> None: ...

    def add_dependency(self, from_task: int, to_task: int) -> None: ...

    def add_dependencies(self, from_task: int, to_tasks: Int32Vector) -> None: ...

    def set_variant(self, id: int, arch: DeviceType, vcu: int, mem: int, time: int) -> None: ...

    def get_n_compute_tasks(self) -> int: ...

    def get_n_data_tasks(self) -> int: ...

    def get_time(self, task_id: int, arch: DeviceType) -> int: ...

    def clear_variants(self, task_id: int) -> None: ...

    def clear_all_variants(self) -> None: ...

    def get_task_dependencies(self, task_id: int) -> Int32Vector: ...

    def finalize(self, ensure_dependencies: bool = False, create_data_tasks: bool = True) -> None: ...

class StaticTaskInfo:
    def __init__(self, graph: Graph) -> None: ...

    def get_data_id(self, task_id: int) -> int: ...

    def get_compute_task(self, task_id: int) -> int: ...

    def set_grid_shape(self, h: int, w: int) -> None: ...

    def get_grid_h(self) -> int: ...

    def get_grid_w(self) -> int: ...

    def has_grid_shape(self) -> bool: ...

    def set_morton_priority_enabled(self, enabled: bool) -> None: ...

    def get_morton_priority_enabled(self) -> bool: ...

    def set_use_random_priority(self, enabled: bool) -> None: ...

    def use_random_priority(self) -> bool: ...

    def set_random_priority_enabled(self, enabled: bool) -> None: ...

    def get_random_priority_enabled(self) -> bool: ...

    def update_variants(self, graph: Graph) -> None: ...

class RuntimeTaskInfo:
    def get_n_compute_tasks(self) -> int: ...

    def get_n_data_tasks(self) -> int: ...

    def get_n_eviction_tasks(self) -> int: ...

    def get_n_tasks(self) -> int: ...

    def get_compute_task_state_at_time(self, arg0: int, arg1: int, /) -> TaskState: ...

    def get_data_task_state_at_time(self, arg0: int, arg1: int, /) -> TaskState: ...

    def get_data_task_mapped_device(self, arg: int, /) -> int: ...

    def get_compute_task_mapped_device(self, arg: int, /) -> int: ...

    def get_data_task_source_device(self, arg: int, /) -> int: ...

    @overload
    def is_data_task_virtual(self, arg: int, /) -> bool: ...

    @overload
    def is_data_task_virtual(self, arg: int, /) -> bool: ...

    def get_data_task_launched_time(self, arg: int, /) -> int: ...

    def get_compute_task_launched_time(self, arg: int, /) -> int: ...

    def get_data_task_completed_time(self, arg: int, /) -> int: ...

    def get_compute_task_completed_time(self, arg: int, /) -> int: ...

    def is_eviction_task_virtual(self, arg: int, /) -> bool: ...

    def get_compute_task_duration(self, arg: int, /) -> int: ...

    def get_data_task_duration(self, arg: int, /) -> int: ...

    def get_eviction_task_source_device(self, arg: int, /) -> int: ...

    def get_eviction_task_launched_time(self, arg: int, /) -> int: ...

    def get_eviction_task_completed_time(self, arg: int, /) -> int: ...

class Data:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_blocks: int) -> None: ...

    def size(self) -> int: ...

    def create_block(self, id: int, size: int, location: int, name: str) -> None: ...

    def append_block(self, size: int, location: int, name: str) -> int: ...

    def set_size(self, id: int, size: int) -> None: ...

    def set_location(self, id: int, location: int) -> None: ...

    def set_name(self, id: int, name: str) -> None: ...

    def set_x_pos(self, id: int, x: float) -> None: ...

    def set_y_pos(self, id: int, y: float) -> None: ...

    def get_x_pos(self, id: int) -> float: ...

    def get_y_pos(self, id: int) -> float: ...

    def get_x_pos_vec(self, id: int) -> float: ...

    def get_y_pos_vec(self, id: int) -> float: ...

    def get_size(self, id: int) -> int: ...

    def get_location(self, id: int) -> int: ...

    def get_name(self, id: int) -> str: ...

    def set_type(self, id: int, type: int) -> None: ...

    def get_type(self, id: int) -> int: ...

    def set_tag(self, id: int, tag: int) -> None: ...

    def get_tag(self, id: int) -> int: ...

    def get_id(self, name: str) -> int: ...

class ValidEventArray:
    @property
    def size(self) -> int: ...

    @property
    def starts(self) -> Int64Vector: ...

    @property
    def stops(self) -> Int64Vector: ...

class TaskNoise:
    def __init__(self, tasks: StaticTaskInfo, seed: int = 0, pseed: int = 0) -> None: ...

    def set_seed(self, seed: int) -> None: ...

    def set_pseed(self, pseed: int) -> None: ...

    def get(self, task_id: int, arch: DeviceType) -> int: ...

    @overload
    def set(self, task_id: int, arch: DeviceType, noise: int) -> None: ...

    @overload
    def set(self, noise: Int64Vector) -> None: ...

    @overload
    def set_priority(self, task_id: int, p: int) -> None: ...

    @overload
    def set_priority(self, noise: Int32Vector) -> None: ...

    def get_priorities(self) -> "std::__1::span<int const, 18446744073709551615ul>": ...

    def get_durations(self) -> "std::__1::span<long long const, 18446744073709551615ul>": ...

    def get_priority_vector(self) -> Int32Vector: ...

    def set_priority_vector(self, priorities: Int32Vector) -> None: ...

    def get_duration_vector(self) -> Int64Vector: ...

    def set_duration_vector(self, durations: Int64Vector) -> None: ...

    def randomize_duration(self, arg: StaticTaskInfo, /) -> None: ...

    def randomize_priority(self, arg: StaticTaskInfo, /) -> None: ...

class LognormalTaskNoise(TaskNoise):
    def __init__(self, tasks: StaticTaskInfo, seed: int = 0, pseed: int = 1000, scale: float = 0.5) -> None: ...

class StaticLognormalTaskNoise(TaskNoise):
    def __init__(self, tasks: StaticTaskInfo, seed: int = 0, pseed: int = 0, stddev: float = 500) -> None: ...

class Topology:
    def __init__(self, num_devices: int) -> None: ...

    def set_bandwidth(self, src: int, dst: int, bandwidth: int) -> None: ...

    def set_max_connections(self, src: int, dst: int, max_links: int) -> None: ...

    def set_latency(self, src: int, dst: int, latency: int) -> None: ...

    def get_latency(self, src: int, dst: int) -> int: ...

    def get_bandwidth(self, src: int, dst: int) -> int: ...

    def get_max_connections(self, src: int, dst: int) -> int: ...

class SchedulerState:
    def get_global_time(self) -> int: ...

    def get_mapping_priority(self, task_id: int) -> int: ...

    def get_reserving_priority(self, task_id: int) -> int: ...

    def get_launching_priority(self, task_id: int) -> int: ...

    def get_task_runtime(self) -> RuntimeTaskInfo: ...

    def get_tasks(self) -> StaticTaskInfo: ...

class HysteresisTransitionConditions:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, open_in_flight: int, close_in_flight: int, starvation_threshold: int) -> None: ...

    @property
    def open_in_flight(self) -> int: ...

    @property
    def close_in_flight(self) -> int: ...

    @property
    def starvation_threshold(self) -> int: ...

    @property
    def last_window_opened(self) -> int: ...

    @property
    def window_open(self) -> bool: ...

class DeviceThresholdTransitionConditions:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, mapped_threshold: int, reserved_threshold: int) -> None: ...

    @property
    def mapped_threshold(self) -> int: ...

    @property
    def reserved_threshold(self) -> int: ...

class ActionVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: ActionVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[Action], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[Action]: ...

    @overload
    def __getitem__(self, arg: int, /) -> Action: ...

    @overload
    def __getitem__(self, arg: slice, /) -> ActionVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: Action, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: Action, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> Action:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: ActionVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: Action, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: ActionVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

class Action:
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: int, /) -> None: ...

    @property
    def pos(self) -> int: ...

    @pos.setter
    def pos(self, arg: int, /) -> None: ...

    @property
    def device(self) -> int: ...

    @device.setter
    def device(self, arg: int, /) -> None: ...

    @property
    def reservable_priority(self) -> int: ...

    @reservable_priority.setter
    def reservable_priority(self, arg: int, /) -> None: ...

    @property
    def launchable_priority(self) -> int: ...

    @launchable_priority.setter
    def launchable_priority(self, arg: int, /) -> None: ...

    def __str__(self) -> str: ...

class Mapper:
    def map_task(self, task_id: int, state: SchedulerState) -> Action: ...

    def map_tasks(self, tasks: Int32Vector, state: SchedulerState) -> ActionVector: ...

class RandomMapper(Mapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, other: RandomMapper) -> None: ...

    @overload
    def __init__(self, seed: int) -> None: ...

class RoundRobinMapper(Mapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, other: RoundRobinMapper) -> None: ...

class StaticMapper(Mapper):
    @overload
    def __init__(self, device_ids_: Int32Vector) -> None: ...

    @overload
    def __init__(self, device_ids: Int32Vector, reserving_priorities: Int32Vector, launching_priorities: Int32Vector) -> None: ...

    @overload
    def __init__(self, other: StaticMapper) -> None: ...

    def set_mapping(self, device_ids_: Int32Vector) -> None: ...

    def set_reserving_priorities(self, reserving_priorites_: Int32Vector) -> None: ...

    def set_launching_priorities(self, launching_priorites_: Int32Vector) -> None: ...

class StaticActionMapper(Mapper):
    @overload
    def __init__(self, actions: ActionVector) -> None: ...

    @overload
    def __init__(self, other: StaticActionMapper) -> None: ...

class DeviceTime:
    def __init__(self, device_id: int, time: int) -> None: ...

    @property
    def device_id(self) -> int: ...

    @property
    def time(self) -> int: ...

class EFTMapper(Mapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_tasks: int, num_devices: int) -> None: ...

    @overload
    def __init__(self, other: EFTMapper) -> None: ...

    def get_best_device(self, task_id: int, state: SchedulerState) -> DeviceTime: ...

    def get_dependency_finish_time(self, task_id: int, state: SchedulerState) -> int: ...

    def get_device_available_time(self, device_id: int, state: SchedulerState) -> int: ...

    def get_finish_time(self, task_id: int, device_id: int, start_t: int, state: SchedulerState) -> int: ...

    def time_for_transfer(self, task_id: int, device_id: int, state: SchedulerState) -> int: ...

class MemoryAwareEFTMapper(EFTMapper):
    alpha: float

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_tasks: int, num_devices: int, alpha: float = ...) -> None: ...

    @overload
    def __init__(self, other: MemoryAwareEFTMapper) -> None: ...

class DequeueEFTMapper(EFTMapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_tasks: int, num_devices: int) -> None: ...

    @overload
    def __init__(self, other: DequeueEFTMapper) -> None: ...

class DataAwareMapper(Mapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_tasks: int, num_devices: int) -> None: ...

    @overload
    def __init__(self, other: DataAwareMapper) -> None: ...

class KaHyParMapper(EFTMapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_tasks: int, num_devices: int) -> None: ...

    @overload
    def __init__(self, other: KaHyParMapper) -> None: ...

    @property
    def mapped_threshold(self) -> int: ...

    @mapped_threshold.setter
    def mapped_threshold(self, arg: int, /) -> None: ...

    @property
    def reserved_threshold(self) -> int: ...

    @reserved_threshold.setter
    def reserved_threshold(self, arg: int, /) -> None: ...

class DARTSMapper(Mapper):
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, num_tasks: int, num_devices: int) -> None: ...

    @overload
    def __init__(self, other: DARTSMapper) -> None: ...

    @property
    def mapped_threshold(self) -> int: ...

    @mapped_threshold.setter
    def mapped_threshold(self, arg: int, /) -> None: ...

    @property
    def reserved_threshold(self) -> int: ...

    @reserved_threshold.setter
    def reserved_threshold(self, arg: int, /) -> None: ...

class IFeatureVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: IFeatureVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[IFeature], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[IFeature]: ...

    @overload
    def __getitem__(self, arg: int, /) -> IFeature: ...

    @overload
    def __getitem__(self, arg: slice, /) -> IFeatureVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: IFeature, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: IFeature, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> IFeature:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: IFeatureVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: IFeature, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: IFeatureVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: IFeature, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: IFeature, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: IFeature, /) -> None:
        """Remove first occurrence of `arg`."""

class IEdgeFeatureVector:
    @overload
    def __init__(self) -> None:
        """Default constructor"""

    @overload
    def __init__(self, arg: IEdgeFeatureVector) -> None:
        """Copy constructor"""

    @overload
    def __init__(self, arg: Iterable[IEdgeFeature], /) -> None:
        """Construct from an iterable object"""

    def __len__(self) -> int: ...

    def __bool__(self) -> bool:
        """Check whether the vector is nonempty"""

    def __repr__(self) -> str: ...

    def __iter__(self) -> Iterator[IEdgeFeature]: ...

    @overload
    def __getitem__(self, arg: int, /) -> IEdgeFeature: ...

    @overload
    def __getitem__(self, arg: slice, /) -> IEdgeFeatureVector: ...

    def clear(self) -> None:
        """Remove all items from list."""

    def append(self, arg: IEdgeFeature, /) -> None:
        """Append `arg` to the end of the list."""

    def insert(self, arg0: int, arg1: IEdgeFeature, /) -> None:
        """Insert object `arg1` before index `arg0`."""

    def pop(self, index: int = -1) -> IEdgeFeature:
        """Remove and return item at `index` (default last)."""

    def extend(self, arg: IEdgeFeatureVector, /) -> None:
        """Extend `self` by appending elements from `arg`."""

    @overload
    def __setitem__(self, arg0: int, arg1: IEdgeFeature, /) -> None: ...

    @overload
    def __setitem__(self, arg0: slice, arg1: IEdgeFeatureVector, /) -> None: ...

    @overload
    def __delitem__(self, arg: int, /) -> None: ...

    @overload
    def __delitem__(self, arg: slice, /) -> None: ...

    def __eq__(self, arg: object, /) -> bool: ...

    def __ne__(self, arg: object, /) -> bool: ...

    @overload
    def __contains__(self, arg: IEdgeFeature, /) -> bool: ...

    @overload
    def __contains__(self, arg: object, /) -> bool: ...

    def count(self, arg: IEdgeFeature, /) -> int:
        """Return number of occurrences of `arg`."""

    def remove(self, arg: IEdgeFeature, /) -> None:
        """Remove first occurrence of `arg`."""

class EmptyTaskFeature:
    def __init__(self, arg0: SchedulerState, arg1: int, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, /) -> IFeature: ...

class InDegreeTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class OutDegreeTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class ReadDegreeTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskInputDegreesFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class PredecessorSizeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class PredecessorMappedDeviceFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DurationTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class OneHotMappedDeviceTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskStateFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class InputOutputTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DepthTaskFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskDeviceMappedTimeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskCoordinatesFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskReadCoordinateFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskDataMappedSizeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskDataMappedCoordinatesFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class CandidateVectorFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class PrevReadSizeFeature:
    def __init__(self, state: SchedulerState, width: int, length: int, add_current: bool = False, frames: int = 3) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, arg2: int, arg3: bool, arg4: int, /) -> IFeature: ...

class PrevMappedSizeFeature:
    def __init__(self, state: SchedulerState, width: int, length: int, add_current: bool = False, frames: int = 3) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, arg2: int, arg3: bool, arg4: int, /) -> IFeature: ...

class PrevMappedDeviceFeature:
    def __init__(self, state: SchedulerState, width: int, length: int, add_current: bool = False, frames: int = 3) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, arg2: int, arg3: bool, arg4: int, /) -> IFeature: ...

class ReadDataLocationFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class TaskMeanDurationFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class EmptyDataFeature:
    def __init__(self, arg0: SchedulerState, arg1: int, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, /) -> IFeature: ...

class DataMappedLocationsFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DataReservedLocationsFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DataLaunchedLocationsFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DataSizeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DataCoordinateFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class EmptyDeviceFeature:
    def __init__(self, arg0: SchedulerState, arg1: int, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, /) -> IFeature: ...

class DeviceMemoryFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DeviceTimeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class DeviceIDFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IFeature: ...

class EmptyTaskTaskFeature:
    def __init__(self, arg0: SchedulerState, arg1: int, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, /) -> IEdgeFeature: ...

class TaskTaskDefaultEdgeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class EmptyTaskDataFeature:
    def __init__(self, arg0: SchedulerState, arg1: int, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg0: SchedulerState, arg1: int, /) -> IEdgeFeature: ...

class TaskDataUsageFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class TaskDataDefaultEdgeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class TaskDataMappedFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class TaskDataMappedOneHotFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class TaskDataSizeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class TaskDeviceDefaultEdgeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class DataDeviceDefaultEdgeFeature:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    @staticmethod
    def create(arg: SchedulerState, /) -> IEdgeFeature: ...

class IFeature:
    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: "std::__1::span<float, 18446744073709551615ul>", /) -> None: ...

class IEdgeFeature:
    @property
    def feature_dim(self) -> int: ...

    def extract_feature(self, arg0: int, arg1: int, arg2: "std::__1::span<float, 18446744073709551615ul>", /) -> None: ...

class RuntimeFeatureExtractor:
    def __init__(self) -> None: ...

    def add_feature(self, feature: IFeature) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def get_features(self, arg0: int, arg1: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    def get_features_batch(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C')], /) -> None: ...

    @property
    def feature_type_names(self) -> StringVector: ...

class RuntimeEdgeFeatureExtractor:
    def __init__(self) -> None: ...

    def add_feature(self, feature: IEdgeFeature) -> None: ...

    @property
    def feature_dim(self) -> int: ...

    def get_features(self, arg0: int, arg1: int, arg2: Annotated[NDArray[numpy.float32], dict(device='cpu')], /) -> None: ...

    def get_features_batch(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg1: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C')], /) -> None: ...

    @property
    def feature_type_names(self) -> StringVector: ...

class GraphSpec:
    def __init__(self) -> None: ...

    @property
    def max_in_degree(self) -> int: ...

    @max_in_degree.setter
    def max_in_degree(self, arg: int, /) -> None: ...

    @property
    def max_out_degree(self) -> int: ...

    @max_out_degree.setter
    def max_out_degree(self, arg: int, /) -> None: ...

    @property
    def max_data_usage(self) -> int: ...

    @max_data_usage.setter
    def max_data_usage(self, arg: int, /) -> None: ...

    @property
    def max_candidates(self) -> int: ...

    @max_candidates.setter
    def max_candidates(self, arg: int, /) -> None: ...

    @property
    def max_edges_tasks_tasks(self) -> int: ...

    @max_edges_tasks_tasks.setter
    def max_edges_tasks_tasks(self, arg: int, /) -> None: ...

    @property
    def max_edges_tasks_data(self) -> int: ...

    @max_edges_tasks_data.setter
    def max_edges_tasks_data(self, arg: int, /) -> None: ...

    @property
    def max_edges_tasks_devices(self) -> int: ...

    @max_edges_tasks_devices.setter
    def max_edges_tasks_devices(self, arg: int, /) -> None: ...

    @property
    def max_edges_data_devices(self) -> int: ...

    @max_edges_data_devices.setter
    def max_edges_data_devices(self, arg: int, /) -> None: ...

    @property
    def max_tasks(self) -> int: ...

    @max_tasks.setter
    def max_tasks(self, arg: int, /) -> None: ...

    @property
    def max_data(self) -> int: ...

    @max_data.setter
    def max_data(self, arg: int, /) -> None: ...

    @property
    def max_devices(self) -> int: ...

    @max_devices.setter
    def max_devices(self, arg: int, /) -> None: ...

    def __str__(self) -> str: ...

class GraphExtractor:
    def __init__(self, arg: SchedulerState, /) -> None: ...

    def get_device_selection_mask(self, arg0: int, arg1: Annotated[NDArray[numpy.int8], dict(shape=(None,), order='C')], /) -> None: ...

    def get_k_hop_dependencies(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: int, arg2: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], /) -> int: ...

    def get_k_hop_dependents(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: int, arg2: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], /) -> int: ...

    def get_k_hop_bidirectional(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: int, arg2: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], /) -> int: ...

    def get_k_hop_neighborhood(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: int, arg2: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], /) -> int: ...

    def get_active_tasks(self) -> Int32Vector: ...

    def get_task_task_edges(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_task_shared_read_edges(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_task_edges_reverse(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_data_edges_all(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg3: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_data_edges_read(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg3: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_data_edges_write(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg3: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_data_edges_read_mapped(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg3: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_task_device_edges(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_data_device_edges(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], arg2: Annotated[NDArray[numpy.int64], dict(shape=(None, None), order='C')], /) -> int: ...

    def get_device_load(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C')], /) -> None: ...

    def get_device_memory(self, arg: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C')], /) -> None: ...

    def get_unique_data(self, arg0: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], arg1: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], /) -> int: ...

def start_logger() -> None: ...

class ExecutionState(enum.IntEnum):
    NONE = 0

    RUNNING = 1

    COMPLETE = 2

    BREAKPOINT = 3

    EXTERNAL_MAPPING = 4

    ERROR = 5

NONE: ExecutionState = ExecutionState.NONE

RUNNING: ExecutionState = ExecutionState.RUNNING

COMPLETE: ExecutionState = ExecutionState.COMPLETE

BREAKPOINT: ExecutionState = ExecutionState.BREAKPOINT

EXTERNAL_MAPPING: ExecutionState = ExecutionState.EXTERNAL_MAPPING

ERROR: ExecutionState = ExecutionState.ERROR

class EvictionPolicy(enum.IntEnum):
    LRU = 0

    LEAST_USED_MAPPED = 1

LRU: EvictionPolicy = EvictionPolicy.LRU

LEAST_USED_MAPPED: EvictionPolicy = EvictionPolicy.LEAST_USED_MAPPED

class SchedulerInput:
    @overload
    def __init__(self, arg0: Graph, arg1: StaticTaskInfo, arg2: Data, arg3: Devices, arg4: Topology, arg5: TaskNoise, arg6: TransitionConditions, /) -> None: ...

    @overload
    def __init__(self, graph: Graph, tasks: StaticTaskInfo, data: Data, devices: Devices, topology: Topology, task_noise: TaskNoise, conditions: TransitionConditions, top_k_candidates: int = 1, eviction_policy: EvictionPolicy = EvictionPolicy.LRU, transfer_aware_data_launch_order: bool = False, /) -> None: ...

    @overload
    def __init__(self, arg: SchedulerInput) -> None: ...

    @property
    def top_k_candidates(self) -> int: ...

    @property
    def eviction_policy(self) -> EvictionPolicy: ...

    @property
    def transfer_aware_data_launch_order(self) -> bool: ...

class Simulator:
    @overload
    def __init__(self, arg0: SchedulerInput, arg1: Mapper, /) -> None: ...

    @overload
    def __init__(self, other: Simulator) -> None: ...

    @property
    def initialized(self) -> bool: ...

    @property
    def use_python_mapper(self) -> bool: ...

    @property
    def last_execution_state(self) -> ExecutionState: ...

    @property
    def data_initialized(self) -> bool: ...

    @property
    def events_processed(self) -> int: ...

    def initialize(self, create_data_tasks: bool = True, initialize_data_manager: bool = False) -> None: ...

    def set_steps(self, steps: int) -> None: ...

    def set_mapper_boundary_steps(self, boundaries: int) -> None: ...

    def start_drain(self) -> None: ...

    def stop_drain(self) -> None: ...

    def initialize_data(self) -> None: ...

    def initialize_data_replicate(self, data_id: int, device_id: int) -> None: ...

    def enable_python_mapper(self) -> None: ...

    def disable_python_mapper(self) -> None: ...

    def skip_external_mapping(self, enqueue_mapping_event: bool = True) -> None: ...

    def set_mapper(self, arg: Mapper, /) -> None: ...

    def get_state(self) -> SchedulerState: ...

    def run(self) -> ExecutionState: ...

    def get_current_time(self) -> int: ...

    def get_evicted_memory_size(self) -> int: ...

    def get_max_memory_usage(self) -> int: ...

    def get_total_data_movement(self) -> Int64Vector: ...

    def get_eviction_data_movement(self) -> Int64Vector: ...

    @overload
    def get_mappable_candidates(self, arg: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C')], /) -> int: ...

    @overload
    def get_mappable_candidates(self, arg: "std::__1::span<long long, 18446744073709551615ul>", /) -> int: ...

    def map_tasks(self, arg: ActionVector, /) -> None: ...

    def add_task_breakpoint(self, arg0: EventType, arg1: int, /) -> None: ...

    def clear_breakpoints(self) -> None: ...

class ParMETIS_wrapper:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, arg: int, /) -> None: ...

    def print_info(self) -> None: ...

    def callParMETIS(self, vtxdist: NDArray[numpy.int32], xadj: NDArray[numpy.int32], adjncy: NDArray[numpy.int32], vwgt: NDArray[numpy.int32], vsize: NDArray[numpy.int32], adjwgt: NDArray[numpy.int32], wgtflag: int, numflag: int, ncon: int, tpwgts: NDArray[numpy.float32], ubvec: NDArray[numpy.float32], itr: float, part: NDArray[numpy.int32]) -> bool: ...
