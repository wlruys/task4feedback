from __future__ import annotations
from ..types import TaskID, TaskInfo, TaskState, TaskStatus, DataAccess, Time, TaskType

from ..types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap
from ..types import Architecture, Device, Devices
from ..types import DataInfo
from typing import (
    List,
    Dict,
    Set,
    Tuple,
    Optional,
    Sequence,
    Mapping,
    MutableMapping,
)

from .queue import PriorityQueue
from dataclasses import dataclass, field
from .resourceset import ResourceSet, FasterResourceSet
from .datapool import DataPool
from ..logging import logger
import numpy as np


@dataclass(slots=True)
class TaskTimes:
    duration: Time = field(default_factory=Time)
    completion_time: Time = field(default_factory=Time)
    state_times: Dict[TaskState, Time] = field(default_factory=dict)
    status_times: Dict[TaskStatus, Time] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        return TaskTimes(
            duration=self.duration,
            completion_time=self.completion_time,
            state_times={k: v for k, v in self.state_times.items()},
            status_times={k: v for k, v in self.status_times.items()},
        )

    def __post_init__(self):
        self.state_times = {}
        self.status_times = {}

        for state in TaskState:
            self.state_times[state] = Time(0)
        for status in TaskStatus:
            self.status_times[status] = Time(0)

    def __getitem__(self, state: TaskState | TaskStatus) -> Time:
        if isinstance(state, TaskState):
            return self.state_times[state]
        else:
            return self.status_times[state]

    def __setitem__(self, state: TaskState | TaskStatus, time: Time):
        if isinstance(state, TaskState):
            self.state_times[state] = time
        else:
            self.status_times[state] = time

    # def get_state(self, time: Time) -> TaskState:
    #     current = TaskState.SPAWNED
    #     for state in [TaskState.MAPPED, TaskState.LAUNCHED, TaskState.COMPLETED]:
    #         if self.state_times[state] <= time:
    #             current = state
    #     return current


@dataclass(slots=True)
class TaskCounters:
    n_deps: int | None = None
    remaining_deps_states: np.ndarray = None
    remaining_deps_status: np.ndarray = None

    def __deepcopy__(self, memo):
        # return TaskCounters(
        #     remaining_deps_states=self.remaining_deps_states,
        #     remaining_deps_status=self.remaining_deps_status,
        # )
        return TaskCounters(
            remaining_deps_states=self.remaining_deps_states.copy(),
            remaining_deps_status=self.remaining_deps_status.copy(),
        )
        # return TaskCounters(
        #     remaining_deps_states={k: v for k, v in self.remaining_deps_states.items()},
        #     remaining_deps_status={k: v for k, v in self.remaining_deps_status.items()},
        # )

    def __post_init__(self):
        if self.n_deps is not None:
            self.remaining_deps_states = np.zeros((len(TaskState),), dtype=np.int8)
            self.remaining_deps_status = np.zeros((len(TaskStatus),), dtype=np.int8)

            for state in TaskState:
                self.remaining_deps_states[state] = self.n_deps

            for state in TaskStatus:
                self.remaining_deps_status[state] = self.n_deps

    def __str__(self) -> str:
        return f"TaskCounters({self.remaining_deps_states})"

    def __repr__(self) -> str:
        return self.__str__()

    def check_count(self, key: TaskStatus | TaskState) -> bool:
        if isinstance(key, TaskState):
            value = self.remaining_deps_states[key]
            assert value >= 0
            return value == 0
        else:
            value = self.remaining_deps_status[key]
            assert value >= 0
            return value == 0

    def notified_state(self, new_state: TaskState) -> Optional[TaskStatus]:
        self.remaining_deps_states[new_state] -= 1
        if self.check_count(new_state):
            if new_status := TaskState.matching_status(new_state):
                return new_status

    def notified_status(self, new_status: TaskStatus) -> None:
        self.remaining_deps_status[new_status] -= 1


from copy import deepcopy


@dataclass(slots=True)
class SimulatedTask:
    name: TaskID
    info: TaskInfo
    state: TaskState = TaskState.SPAWNED
    status: Set[TaskStatus] = field(default_factory=set)
    # times: TaskTimes = field(default_factory=TaskTimes)
    counters: TaskCounters | None = None
    dependencies: List[TaskID] = field(default_factory=list)
    dependents: List[TaskID] = field(default_factory=list)
    resources: List[FasterResourceSet] = field(default_factory=list)
    depth: int = 0
    type: TaskType = TaskType.BASE
    parent: Optional[TaskID] = None
    data_tasks: Optional[List[TaskID]] = None
    # eviction_tasks: Optional[List[TaskID]] = None
    spawn_tasks: Optional[List[TaskID]] = None
    eviction_requested: bool = False
    duration: Time = field(default_factory=Time)
    completion_time: Time = field(default_factory=Time)
    # This is only used for online EFT-based schedulers
    est_completion_time: float = 0
    wait_time: Time = field(default_factory=Time)
    init: bool = True

    def __post_init__(self):
        if self.init:
            self.dependencies = [d for d in self.info.dependencies]
            self.counters = TaskCounters(len(self.info.dependencies))
            self.init = False

    def __deepcopy__(self, memo):

        state = self.state
        status = {s for s in self.status}

        # times = deepcopy(self.times)

        counters = deepcopy(self.counters)
        # eviction_tasks = deepcopy(self.eviction_tasks)
        eviction_requested = self.eviction_requested
        resources = [deepcopy(r) for r in self.resources]

        return SimulatedTask(
            name=self.name,
            info=self.info,
            state=state,
            status=status,
            # times=times,
            counters=counters,
            dependencies=[d for d in self.dependencies],
            dependents=[d for d in self.dependents],
            resources=resources,
            depth=self.depth,
            type=self.type,
            parent=self.parent,
            data_tasks=self.data_tasks,
            # eviction_tasks=eviction_tasks,
            spawn_tasks=self.spawn_tasks,
            eviction_requested=eviction_requested,
            init=self.init,
        )

    def set_status(self, new_status: TaskStatus, time: Time, verify: bool = True):
        # print(f"Setting {self.name} to {new_status}. Status: {self.status}")
        # self.times[new_status] = time
        self.status.add(new_status)

    def set_state(self, new_state: TaskState, time: Time, verify: bool = True):
        # print(f"Setting {self.name} to {new_state}. State: {self.state}")

        if verify:
            TaskStatus.check_valid_transition(self.status, new_state)
            TaskState.check_valid_transition(self.state, new_state)

        # self.times[new_state] = time
        self.state = new_state

    def set_states(self, new_states: List[TaskState], time: Time):
        for state in new_states:
            self.set_state(state, time)

    @property
    def read_accesses(self) -> List[DataAccess]:
        return self.info.data_dependencies.read

    @property
    def write_accesses(self) -> List[DataAccess]:
        return self.info.data_dependencies.write

    @property
    def read_write_accesses(self) -> List[DataAccess]:
        return self.info.data_dependencies.read_write

    @property
    def priority(self) -> int:
        return self.info.order

    @priority.setter
    def priority(self, priority: int):
        self.info.order = priority

    # @property
    # def completion_time(self) -> Time:
    #     return self.times.completion_time

    # @completion_time.setter
    # def completion_time(self, time: Time):
    #     self.times.completion_time = time

    @property
    def assigned_devices(self) -> Optional[Tuple[Device, ...]]:
        if isinstance(self.info.mapping, Device):
            return (self.info.mapping,)
        else:
            return self.info.mapping

    @assigned_devices.setter
    def assigned_devices(self, devices: Tuple[Device, ...]):
        self.info.mapping = devices

    @property
    def read_data_list(self) -> List[DataAccess]:
        return self.info.data_dependencies.read

    @property
    def write_data_list(self) -> List[DataAccess]:
        return self.info.data_dependencies.write

    def add_dependency(
        self,
        task: TaskID,
        states: List[TaskState] = [],
        statuses: List[TaskStatus] = [],
    ):
        self.dependencies.append(task)
        for state in states:
            self.counters.remaining_deps_states[state] += 1
        for status in statuses:
            self.counters.remaining_deps_status[status] += 1

    def add_dependent(self, task: TaskID):
        self.dependents.append(task)

    def notify_state(
        self,
        state: TaskState,
        taskmap: SimulatedTaskMap,
        time: Time,
        verbose: bool = False,
    ):
        # Notify only if changed
        if self.state == state:
            return

        logger.state.debug(
            "Notifying dependents of state change",
            extra=dict(task=self.name, state=state, time=time),
        )

        for taskid in self.dependents:
            task = taskmap[taskid]
            if new_status := task.counters.notified_state(state):
                task.notify_status(new_status, taskmap, time)

        self.set_state(state, time)

    def notify_status(self, status: TaskStatus, taskmap: SimulatedTaskMap, time: Time):
        logger.state.debug(
            "Notifying dependents of status change",
            extra=dict(task=self.name, status=status, time=time),
        )

        for taskid in self.dependents:
            task = taskmap[taskid]
            task.counters.notified_status(status)

        self.set_status(status, time)

    def check_status(
        self,
        status: TaskStatus,
        taskmap: SimulatedTaskMap,
        time: Time,
        verbose: bool = False,
    ) -> bool:
        if checked_state := TaskStatus.matching_state(status):
            if status not in self.status and self.counters.check_count(checked_state):
                self.notify_status(status, taskmap, time)
                return True
        return status in self.status

    def __str__(self) -> str:
        return f"Task({self.name}, {self.state})"

    def __rich_repr__(self):
        yield "name", self.name
        yield "state", self.state
        yield "status", self.status
        yield "duration", self.duration
        yield "func_id", self.info.func_id
        yield "dependencies", self.dependencies
        yield "assigned_devices", self.assigned_devices

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimulatedTask):
            return NotImplemented
        return self.name == other.name

    def get_runtime_info(self, device: Devices) -> List[TaskRuntimeInfo]:
        return self.info.runtime[device]

    def get_resources(
        self, devices: Devices, data_inputs: bool = False
    ) -> List[FasterResourceSet]:
        raise NotImplementedError

    def add_eviction_dependency(self, task: SimulatedTask):
        assert task.type == TaskType.EVICTION
        self.add_dependency(task.name, states=[TaskState.LAUNCHED, TaskState.COMPLETED])
        task.dependents.append(self.name)

        # if self.eviction_tasks is None:
        #     self.eviction_tasks = []
        # self.eviction_tasks.append(task.name)


@dataclass(slots=True)
class SimulatedComputeTask(SimulatedTask):
    type: TaskType = TaskType.COMPUTE

    def add_data_dependency(self, task: TaskID):
        self.add_dependency(task, states=[TaskState.LAUNCHED, TaskState.COMPLETED])

        if self.data_tasks is None:
            self.data_tasks = []
        self.data_tasks.append(task)

    def get_resources(self, devices: Devices) -> List[FasterResourceSet]:
        resources = []

        if isinstance(devices, Device):
            devices = (devices,)
        runtime_info_list = self.get_runtime_info(devices)
        for runtime_info in runtime_info_list:
            vcus = runtime_info.device_fraction
            memory = runtime_info.memory
            resources.append(FasterResourceSet(vcus=vcus, memory=memory, copy=0))

        return resources

    def __deepcopy__(self, memo):

        return SimulatedComputeTask(
            name=self.name,
            info=self.info,
            state=self.state,
            status={s for s in self.status},
            # times=deepcopy(self.times),
            counters=deepcopy(self.counters),
            dependencies=[t for t in self.dependencies],
            dependents=[t for t in self.dependents],
            resources=[deepcopy(r) for r in self.resources],
            depth=self.depth,
            type=self.type,
            parent=self.parent,
            data_tasks=self.data_tasks,
            # eviction_tasks=deepcopy(self.eviction_tasks),
            spawn_tasks=self.spawn_tasks,
            eviction_requested=deepcopy(self.eviction_requested),
            init=self.init,
        )


@dataclass(slots=True)
class SimulatedDataTask(SimulatedTask):
    type: TaskType = TaskType.DATA
    source: Optional[Device] = None
    local_index: int = 0
    real: bool = True

    def get_resources(self, devices: Devices) -> List[FasterResourceSet]:
        return [FasterResourceSet(vcus=0, memory=0, copy=1)]

    def __deepcopy__(self, memo):
        return SimulatedDataTask(
            name=self.name,
            info=self.info,
            state=self.state,
            status={s for s in self.status},
            source=self.source,
            local_index=self.local_index,
            real=self.real,
            # times=deepcopy(self.times),
            counters=deepcopy(self.counters),
            dependencies=[t for t in self.dependencies],
            dependents=[t for t in self.dependents],
            resources=[deepcopy(r) for r in self.resources],
            depth=self.depth,
            type=self.type,
            parent=self.parent,
            data_tasks=self.data_tasks,
            # eviction_tasks=deepcopy(self.eviction_tasks),
            spawn_tasks=self.spawn_tasks,
            eviction_requested=deepcopy(self.eviction_requested),
            init=self.init,
        )


@dataclass(slots=True)
class SimulatedEvictionTask(SimulatedDataTask):
    type: TaskType = TaskType.EVICTION

    def __deepcopy__(self, memo):
        return SimulatedEvictionTask(
            name=self.name,
            info=self.info,
            state=self.state,
            source=self.source,
            local_index=self.local_index,
            real=self.real,
            status={s for s in self.status},
            # times=deepcopy(self.times),
            counters=deepcopy(self.counters),
            dependencies=[t for t in self.dependencies],
            dependents=[t for t in self.dependents],
            resources=[deepcopy(r) for r in self.resources],
            depth=self.depth,
            type=self.type,
            parent=self.parent,
            data_tasks=self.data_tasks,
            spawn_tasks=self.spawn_tasks,
            eviction_requested=self.eviction_requested,
            init=self.init,
        )


SimulatedTaskMap = Mapping[
    TaskID, SimulatedTask | SimulatedComputeTask | SimulatedDataTask
]
SimulatedComputeTaskMap = MutableMapping[TaskID, SimulatedComputeTask]
SimulatedDataTaskMap = MutableMapping[TaskID, SimulatedDataTask]
