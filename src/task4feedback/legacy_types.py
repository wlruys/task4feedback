from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional,
    Callable,
    Set,
    Type,
    Mapping,
    MutableMapping,
)
from dataclasses import dataclass, field, InitVar
from enum import IntEnum
from functools import total_ordering
from collections import defaultdict

from fractions import Fraction
from decimal import Decimal

from ast import literal_eval as make_tuple

from rich.text import Text
import numpy as np

#########################################
# Time Information
#########################################

time_units: List[str] = ["ns", "us", "ms", "s", "m", "h", "d"]
time_scale: List = [
    1_000_000_000,
    1_000_000,
    1_000,
    1,
    Fraction(1, 60),
    Fraction(1, 3600),
    Fraction(1, 86400),
]


@total_ordering
@dataclass(slots=True)
class Time:
    duration: int = 0
    unit: str = "us"
    display_unit: str = "us"

    def __post_init__(self):
        assert self.duration >= 0, f"Time cannot be negative: {self.duration}"

    def scale_between(self, target_unit: str) -> int | Fraction | Decimal:
        if target_unit not in time_units:
            raise ValueError(f"Invalid time unit: {target_unit}")
        if self.unit not in time_units:
            raise ValueError(f"Invalid time unit: {self.unit}")

        target_idx = time_units.index(target_unit)
        current_idx = time_units.index(self.unit)

        return Fraction(time_scale[target_idx], time_scale[current_idx])

    def scale_to(self, target_unit: str) -> int | Fraction | Decimal:
        value = self.scale_between(target_unit) * self.duration
        return value

    def print(self, unit: str | None = None) -> str:
        value = self.scale_to(unit or self.display_unit)
        value_str = str(float(value)) if isinstance(value, Fraction) else str(value)
        return f"{value_str}{unit or self.display_unit}"

    def __str__(self):
        return self.print()

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, Time) and self.unit == other.unit:
            return Time(self.duration + other.duration)
        elif isinstance(other, int):
            return Time(self.duration + other)
        else:
            raise TypeError(f"Invalid type for Time.__add__: {type(other)}:{other}")

    def __sub__(self, other):
        if isinstance(other, Time) and self.unit == other.unit:
            return Time(self.duration - other.duration)
        elif isinstance(other, int):
            return Time(self.duration - other)
        else:
            raise TypeError(f"Invalid type for Time.__sub__: {type(other)}:{other}")

    def __iadd__(self, other):
        if isinstance(other, Time) and self.unit == other.unit:
            self.duration += other.duration
        elif isinstance(other, int):
            self.duration += other
        else:
            raise TypeError(f"Invalid type for Time.__add__: {type(other)}:{other}")
        return self

    def __isub__(self, other):
        if isinstance(other, Time) and self.unit == other.unit:
            self.duration -= other.duration
        elif isinstance(other, int):
            self.duration -= other
        else:
            raise TypeError(f"Invalid type for Time.__sub__: {type(other)}:{other}")
        assert self.duration >= 0, f"Time cannot be negative: {self.duration}"
        return self

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Time) and self.unit == __value.unit:
            return self.duration == __value.duration
        elif isinstance(__value, int):
            return self.duration == __value
        else:
            return False

    def __lt__(self, __value: object) -> bool:
        if isinstance(__value, Time) and self.unit == __value.unit:
            return self.duration < __value.duration
        elif isinstance(__value, int):
            return self.duration < __value
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.duration)

    def __bool__(self) -> bool:
        return self.duration > 0


#########################################
# Device Information
#########################################


class Architecture(IntEnum):
    """
    Used to specify the architecture of a device in a synthetic task graph.
    """

    ANY = -1
    CPU = 0
    GPU = 1

    def __str__(self):
        return self.name


@dataclass(frozen=True, slots=True)
class Device:
    """
    Identifies a device in a synthetic task graph.
    """

    # The architecture of the device
    architecture: Architecture = Architecture.CPU
    # The id of the device (-1 for any)
    device_id: int = 0

    def __str__(self):
        return f"{self.architecture.name}[{self.device_id}]"

    def __repr__(self):
        return str(self)


Devices = Device | Tuple[Device, ...]

#########################################
# Data Information
#########################################


class AccessType(IntEnum):
    """
    Used to specify the type of access for a data object in a synthetic task graph.
    """

    READ = 0
    WRITE = 1
    READ_WRITE = 2
    EVICT = 3


@dataclass(slots=True)
class DataID:
    idx: Tuple[int, ...] = (0,)

    def __init__(self, idx: Tuple[int, ...] | int):
        if isinstance(idx, int):
            idx = (idx,)
        self.idx = idx

    def __str__(self):
        return f"{self.idx}"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return hash(self.idx)

    def __eq__(self, other) -> bool:
        return self.idx == other.idx

    def __lt__(self, other) -> bool:
        return self.idx < other.idx


@dataclass(slots=True)
class DataInfo:
    """
    The collection of information for a data object in a synthetic task graph.

    @field id: The id of the data object
    @field size: The size of the data object
    @field location: The initial allocation location of the data object (device or tuple of devices)

    Distribution is assumed to be uniform partitioning along the first dimension across all devices.
    """

    id: DataID
    size: int
    location: Device | Tuple[Device, ...] | None

    def __init__(
        self,
        id: DataID | Tuple[int, ...] | int,
        size: int,
        location: Device | Tuple[Device, ...] | None,
    ):
        if not isinstance(id, DataID):
            id = DataID(id)

        self.id = id
        self.size = size
        self.location = location


@dataclass(slots=True)
class DataAccess:
    """
    The collection of information for a data access in a synthetic task graph.

    @field id: The id of the data object
    @field pattern: The access pattern for the data object (slice, list, int, or None)

    Only access patterns along the first dimension are supported.
    """

    id: DataID
    pattern: slice | list[int] | int | None = None
    device: int = 0

    def __init__(
        self,
        id: DataID | Tuple[int, ...] | int,
        pattern: slice | list[int] | int | None = None,
        device: int = 0,
    ):
        if not isinstance(id, DataID):
            id = DataID(id)

        self.id = id
        self.pattern = pattern
        self.device = device

    def __hash__(self) -> int:
        return hash((id, self.pattern, self.device))

    def __eq__(self, __value: object) -> bool:
        return (self.id == __value.id) and (self.device == __value.device)


@dataclass(slots=True)
class TaskDataInfo:
    """
    The data dependencies for a task in a synthetic task graph.

    @field read: The list of data objects that are read by the task
    @field write: The list of data objects that are written by the task (and not read). These don't really exist.
    @field read_write: The list of data objects that are read and written by the task
    """

    read: list[DataAccess] = field(default_factory=list)
    write: list[DataAccess] = field(default_factory=list)
    read_write: list[DataAccess] = field(default_factory=list)

    def __getitem__(self, access: AccessType):
        if access == AccessType.READ:
            return self.read
        elif access == AccessType.WRITE:
            return self.write
        elif access == AccessType.READ_WRITE:
            return self.read_write
        else:
            raise ValueError(f"Invalid access type: {access}")

    def __contains__(self, access: DataAccess) -> bool:
        if access in self.read:
            return True
        if access in self.write:
            return True
        if access in self.read_write:
            return True
        return False

    def __len__(self) -> int:
        return len(self.read) + len(self.write) + len(self.read_write)

    def all_accesses(self) -> list[DataAccess]:
        return self.read + self.write + self.read_write

    def all_ids(self) -> list[DataID]:
        return [x.id for x in self.all_accesses()]


DataMap = Dict[DataID, DataInfo]


#########################################
# Task Graph Information
#########################################


class TaskStatus(IntEnum):
    NONE = 0
    MAPPABLE = 1
    RESERVABLE = 2
    LAUNCHABLE = 3

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def matching_state(status: "TaskStatus") -> Optional["TaskState"]:
        if status == TaskStatus.MAPPABLE:
            return TaskState.MAPPED
        elif status == TaskStatus.RESERVABLE:
            return TaskState.RESERVED
        elif status == TaskStatus.LAUNCHABLE:
            return TaskState.COMPLETED
        else:
            return None

    @staticmethod
    def check_valid_transition(
        status_set: Set["TaskStatus"], new_state: "TaskState"
    ) -> bool:
        # ~very Parla specific checks
        if new_state == TaskState.MAPPED:
            return TaskStatus.MAPPABLE in status_set
        elif new_state == TaskState.RESERVED:
            return TaskStatus.RESERVABLE in status_set
        elif new_state == TaskState.LAUNCHED:
            return TaskStatus.LAUNCHABLE in status_set
        else:
            return True


class TaskState(IntEnum):
    NONE = 0
    SPAWNED = 1
    MAPPED = 2
    RESERVED = 3
    LAUNCHED = 4
    COMPLETED = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def matching_status(state: "TaskState") -> Optional[TaskStatus]:
        if state == TaskState.MAPPED:
            return TaskStatus.MAPPABLE
        elif state == TaskState.RESERVED:
            return TaskStatus.RESERVABLE
        elif state == TaskState.COMPLETED:
            return TaskStatus.LAUNCHABLE
        else:
            return None

    @staticmethod
    def check_valid_transition(old_state, new_state) -> bool:
        # ~very Parla specific checks
        if new_state == TaskState.MAPPED:
            return old_state == TaskState.SPAWNED
        elif new_state == TaskState.RESERVED:
            return old_state == TaskState.MAPPED
        elif new_state == TaskState.LAUNCHED:
            return old_state == TaskState.RESERVED
        else:
            # I am not sure how you got here, but I'm not going to stop you
            return True


class TaskType(IntEnum):
    BASE = -1
    COMPUTE = 0
    DATA = 1
    EVICTION = 2


class ResourceType(IntEnum):
    VCU = 0
    MEMORY = 1
    COPY = 2
    ALL = 3


@dataclass(frozen=True, slots=True)
class TaskID:
    """
    The identifier for a task in a synthetic task graph.
    """

    taskspace: str = "T"  # The task space the task belongs to
    task_idx: Tuple[int, ...] = (0,)  # The index of the task in the task space
    # How many times the task has been spawned (continuation number)
    instance: int = 0

    def __str__(self):
        return f"{self.taskspace}[{', '.join([str(x) for x in [*self.task_idx]])}]"

    def __repr__(self):
        return str(self)

    def base(self) -> "TaskID":
        return TaskID(self.taskspace, self.task_idx, 0)

    def __hash__(self) -> int:
        return hash((self.taskspace, self.task_idx, self.instance))


@dataclass(slots=True)
class TaskRuntimeInfo:
    """
    The collection of important runtime information / constraints for a task in a synthetic task graph.
    """

    task_time: int = 0
    device_fraction: Union[float, Fraction] = 0
    gil_accesses: int = 0
    gil_fraction: Union[float, Fraction] = 0
    memory: int = 0


TaskRuntimeSpec = (
    TaskRuntimeInfo | Mapping[Device, TaskRuntimeInfo] | List[TaskRuntimeInfo]
)
TaskRuntimeMap = MutableMapping[Devices, TaskRuntimeSpec]


@dataclass(slots=True)
class TaskPlacementInfo:
    locations: list[Device | Tuple[Device, ...]] = field(default_factory=list)
    info: TaskRuntimeMap = field(default_factory=dict)
    lookup: Mapping[int, Mapping[int, Dict[Device, TaskRuntimeInfo]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    def add(
        self,
        placement: Device | Tuple[Device, ...],
        runtime_info: (
            TaskRuntimeInfo | Dict[Device, TaskRuntimeInfo] | List[TaskRuntimeInfo]
        ),
    ):
        if isinstance(placement, Device):
            placement = (placement,)

        if isinstance(placement, tuple) and isinstance(runtime_info, TaskRuntimeInfo):
            for localidx, device in enumerate(placement):
                self.lookup[len(placement)][localidx][device] = runtime_info
        elif isinstance(placement, tuple) and isinstance(runtime_info, dict):
            for localidx, device in enumerate(placement):
                self.lookup[len(placement)][localidx][device] = runtime_info[device]
        elif isinstance(placement, tuple) and isinstance(runtime_info, list):
            if len(placement) != len(runtime_info):
                raise ValueError(
                    f"Invalid placement and runtime_info. {placement} and {runtime_info} must be the same length."
                )
            for localidx, device in enumerate(placement):
                details = runtime_info[localidx]
                self.lookup[len(placement)][localidx][device] = details
        else:
            raise ValueError(
                f"Invalid runtime_info type: {type(runtime_info)}. Expected TaskRuntimeInfo or Dict[Device, TaskRuntimeInfo]."
            )

        self.locations.append(placement)
        self.info[placement] = runtime_info

        return self

    def remove(self, placement: Device | Tuple[Device, ...]):
        if isinstance(placement, Device):
            placement = (placement,)

        if isinstance(placement, tuple):
            for localidx, device in enumerate(placement):
                del self.lookup[len(placement)][localidx][device]
        else:
            raise ValueError(
                f"Invalid placement: {placement} of {type(placement)}. Expected Device or Tuple[Device]"
            )

        del self.info[placement]
        self.locations.remove(placement)

        return self

    def update(self):
        """
        Convert self.info to self.lookup
        """
        for device, details in self.info.items():
            if isinstance(device, Device):
                device = (device,)

            if isinstance(details, dict):
                for localidx, d in enumerate(device):
                    self.lookup[len(device)][localidx][d] = details[d]
            elif isinstance(details, list):
                if len(device) != len(details):
                    raise ValueError(
                        f"Invalid placement and runtime_info. {device} and {details} must be the same length."
                    )
                for localidx, d in enumerate(device):
                    self.lookup[len(device)][localidx][d] = details[localidx]
            elif isinstance(details, TaskRuntimeInfo):
                for localidx, d in enumerate(device):
                    self.lookup[len(device)][localidx][d] = details

        self.locations = list(self.info.keys())

        return self

    def __len__(self):
        return len(self.info)

    def __repr__(self):
        return repr(self.info)

    def _get_any(self, device: Device, lookup: Dict[Device, TaskRuntimeInfo]):
        if device in lookup:
            return lookup[device]

        any_of_device = Device(device.architecture, -1)
        if any_of_device in lookup:
            return lookup[any_of_device]

        generic = Device(Architecture.ANY, -1)
        if generic in lookup:
            return lookup[generic]

        return None

    def __getitem__(
        self, placement: Device | Tuple[Device, ...]
    ) -> List[TaskRuntimeInfo]:
        if placement is None:
            raise KeyError("Placement query cannot be None.")

        if isinstance(placement, Device):
            placement = (placement,)

        if isinstance(placement, tuple):
            runtime_info_list = []
            for idx, device in enumerate(placement):
                runtime_info = self._get_any(device, self.lookup[len(placement)][idx])
                if runtime_info is not None:
                    runtime_info_list.append(runtime_info)
                else:
                    raise KeyError(f"RuntimeInfo not found for {placement}.")
        else:
            raise KeyError(
                f"Invalid placement: {placement} of {type(placement)}. Expected Device or Tuple[Device]"
            )

        return runtime_info_list

    def __contains__(self, placement: Device | Tuple[Device, ...]) -> bool:
        if placement is None:
            return False

        if isinstance(placement, Device):
            placement = (placement,)

        if isinstance(placement, tuple):
            for idx, device in enumerate(placement):
                runtime_info = self._get_any(device, self.lookup[len(placement)][idx])
                if runtime_info is None:
                    return False
        else:
            raise KeyError(
                f"Invalid placement: {placement} of {type(placement)}. Expected Device or Tuple[Device]"
            )

        return True


@dataclass(slots=True)
class TaskInfo:
    """
    The collection of important information for a task in a synthetic task graph.
    """

    id: TaskID
    runtime: TaskPlacementInfo
    dependencies: list[TaskID]
    data_dependencies: TaskDataInfo
    mapping: Device | Tuple[Device, ...] | None = None
    order: int = 0
    # Longest depth from root tasks
    depth: int = -1
    # User-defined task integer ID; in general, it is decided based
    # on function (or kernel) that this task calls
    func_id: int = 0
    heft_rank: int = -1
    heft_makespan: float = 0
    heft_allocation: int = -1
    z3_allocation: int = -1
    z3_order: int = -1


# Graph Type Aliases
TaskMap = Mapping[TaskID, TaskInfo]

#########################################
# Execution Records
#########################################


@dataclass(slots=True)
class TaskTime:
    """
    The parsed timing information from a task from an execution log.
    """

    assigned_t: float
    start_t: float
    end_t: float
    duration: float


@dataclass(slots=True)
class TimeSample:
    """
    A collection of timing information.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int


#########################################
# Generic Synthetic Graph Configurations
#########################################


class MovementType(IntEnum):
    """
    Used to specify the type of data movement to be used in a synthetic task graph execution.
    """

    NO_MOVEMENT = 0
    LAZY_MOVEMENT = 1
    EAGER_MOVEMENT = 2


class DataInitType(IntEnum):
    """
    Used to specify the data movement pattern and initialization in a synthetic task graph execution.
    """

    NO_DATA = 0
    INDEPENDENT_DATA = 1
    OVERLAPPED_DATA = 2


#########################################
# Synthetic Graph Execution Configurations
#########################################


@dataclass(slots=True)
class RunConfig:
    """
    Configuration object for executing a synthetic task graph.

    @field outer_iterations: Number of times to launch the Parla runtime and execute the task graph
    @field inner_iterations: Number of times to execute the task graph within the same Parla runtime
    @field inner_sync: Whether to synchronize after each kernel launch
    @field outer_sync: Whether to synchronize at the end of the task
    @field verbose: Whether to print the task graph to the console
    @field device_fraction: VCUs
    @field data_scale: Scaling factor to increase the size of the data objects
    @field threads: Number of threads to use for the Parla runtime
    @field task_time: Total time for all tasks (this overrides the time in the graphs)
    @field gil_fraction: Fraction of time spent in the GIL (this overrides the time in the graphs)
    @field gil_accesses: Number of kernel launches/GIL accesses per task (this overrides the time in the graphs)
    @field movement_type: The data movement pattern to use
    @field logfile: The log file location
    @field do_check: If this is true, validate configuration/execution
    @field num_gpus: The number of GPUs to use for the execution
    """

    outer_iterations: int = 1
    inner_iterations: int = 1
    inner_sync: bool = False
    outer_sync: bool = False
    verbose: bool = False
    device_fraction: Optional[float | Fraction] = None
    data_scale: float = 1.0
    threads: int = 1
    task_time: Optional[int] = None
    gil_fraction: Optional[float | Fraction] = None
    gil_accesses: Optional[int] = None
    movement_type: int = MovementType.NO_MOVEMENT
    logfile: str = "testing.blog"
    do_check: bool = False
    num_gpus: int = 4
    use_cpu_sleep: bool = True


class ExecutionMode(IntEnum):
    """
    Specify the current execution mode
    """

    TRAINING = 0
    TESTING = 1
    EVALUATION = 2

    def __str__(self):
        return self.name


class TaskOrderType(IntEnum):
    """
    Specify task spawning (mapping) order
    """

    DEFAULT = 0
    HEFT = 1
    RANDOM = 2
    REPLAY_LAST_ITER = 3
    REPLAY_FILE = 4
    OPTIMAL = 5


#########################################
# Utility Functions & Conversions
#########################################


def apply_mapping(
    mapping: Dict[TaskID, Device | Tuple[Device, ...]], tasks: TaskMap
) -> TaskMap:
    """
    Apply the mapping to the tasks
    """
    for task_id, device in mapping.items():
        tasks[task_id].mapping = device

    return tasks


def apply_order(order: Dict[TaskID, int], tasks: TaskMap) -> TaskMap:
    """
    Apply the order to the tasks
    """
    for task_id, v in order.items():
        tasks[task_id].order = v

    return tasks


def extract_mapping(tasks: TaskMap) -> Dict[TaskID, Device | Tuple[Device, ...]]:
    """
    Extract the mapping from the tasks
    """
    mapping = {}
    for task_id, task in tasks.items():
        mapping[task_id] = task.mapping

    return mapping


def extract_order(tasks: TaskMap) -> Dict[TaskID, int]:
    """
    Extract the order from the tasks
    """
    order = {}
    for task_id, task in tasks.items():
        order[task_id] = task.order

    return order


def get_base_task_id(task_id: TaskID) -> TaskID:
    """
    Get the base task id for a task id. This is the task id with instance=0.
    """
    return TaskID(taskspace=task_id.taskspace, task_idx=task_id.task_idx, instance=0)


def task_id_to_str(task_id: TaskID) -> str:
    """
    Convert a task id to a string
    """
    return f"{task_id.taskspace}[{task_id.task_idx}]"


def decimal_from_fraction(frac):
    return frac.numerator / Decimal(frac.denominator)


def numeric_from_str(string: str) -> int | Fraction:
    """
    Extracts string as decimal or int
    """
    if "." in string:
        return Fraction(string)
    else:
        return int(string)


def numeric_to_str(obj: Fraction | Decimal):
    """
    Convert other numeric types to strings of the form "0.00"
    """
    if isinstance(obj, Fraction):
        return f"{decimal_from_fraction(obj):0.2f}"
    elif isinstance(obj, Decimal):
        return f"{obj:0.2f}"
    else:
        raise ValueError(f"Unsupported numeric type {type(obj)} of value {obj}")


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def make_data_info(data_info: Dict) -> DataInfo:
    data_idx = make_tuple(data_info["id"])
    assert isinstance(data_idx, tuple)
    data_size = int(data_info["size"])
    data_location = device_from_string(data_info["location"])

    return DataInfo(DataID(data_idx), data_size, data_location)


def make_task_id_from_dict(task_id: Dict) -> TaskID:
    taskspace = task_id["taskspace"]
    task_idx = make_tuple(task_id["task_idx"])
    assert isinstance(task_idx, tuple)

    task_instance = int(task_id["instance"])
    return TaskID(taskspace, task_idx, task_instance)


def make_data_access_from_dict(data_access: Dict) -> DataAccess:
    data_idx = make_tuple(data_access["id"])
    assert isinstance(data_idx, tuple)

    if "pattern" in data_access:
        data_pattern = data_access["pattern"]
        if data_pattern is not None:
            raise NotImplementedError("Access patterns currently not supported.")

    return DataAccess(id=DataID(data_idx))


def make_data_dependencies_from_dict(data_dependencies: Dict) -> TaskDataInfo:
    read_data = [make_data_access_from_dict(x) for x in data_dependencies["read"]]
    write_data = [make_data_access_from_dict(x) for x in data_dependencies["write"]]
    read_write_data = [
        make_data_access_from_dict(x) for x in data_dependencies["read_write"]
    ]
    return TaskDataInfo(read_data, write_data, read_write_data)


def make_task_runtime_from_dict(task_runtime: Dict) -> TaskRuntimeInfo:
    task_time = int(task_runtime["task_time"])
    device_fraction = Fraction(task_runtime["device_fraction"])
    gil_accesses = int(task_runtime["gil_accesses"])
    gil_fraction = Fraction(task_runtime["gil_fraction"])
    memory = int(task_runtime["memory"])

    return TaskRuntimeInfo(
        task_time, device_fraction, gil_accesses, gil_fraction, memory
    )


def device_from_string(device_str: str) -> Optional[Devices]:
    """
    Convert a device string (or string of a device tuple) to a device set
    """
    if device_str is None:
        return None

    processed_str = device_str.strip()
    processed_str = processed_str.strip()
    processed_str = processed_str.strip("()")
    processed_str = processed_str.strip()
    processed_str = processed_str.split(",")
    processed_str = [d.strip() for d in processed_str]

    devices = []

    for d in processed_str:
        if d.isspace() or d == "":
            continue

        d = d.strip()
        d = d.strip("]")
        d = d.split("[")

        if d[0] == "CPU":
            devices.append(Device(Architecture.CPU, int(d[1])))
        elif d[0] == "GPU":
            devices.append(Device(Architecture.GPU, int(d[1])))
        elif d[0] == "ANY":
            devices.append(Device(Architecture.ANY, int(d[1])))
        else:
            raise ValueError(f"Unknown device type {d[0]} in {device_str}")

    if len(devices) == 1:
        return devices[0]
    else:
        return tuple(devices)


def make_task_placement_from_dict(
    task_runtime: Dict,
) -> TaskPlacementInfo:
    """
    Parse the device runtime from a dictionary
    """
    device_runtime = {}
    for device_str, runtime in task_runtime.items():
        device = device_from_string(device_str)

        if "task_time" in runtime:
            device_runtime[device] = make_task_runtime_from_dict(runtime)
        elif isinstance(runtime, list):
            device_runtime[device] = [make_task_runtime_from_dict(r) for r in runtime]
        elif isinstance(device, tuple):
            device_runtime[device] = make_task_runtime_from_dict(runtime)
        else:
            raise ValueError(
                f"Unknown device type {device} or Invalid runtime {runtime} configuration."
            )

    device_runtime = TaskPlacementInfo(info=device_runtime)
    device_runtime.update()

    return device_runtime
