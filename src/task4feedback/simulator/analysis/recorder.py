from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Optional, Sequence, Set, Tuple, Self, Type
from fractions import Fraction

from ..events import *
from ..queue import EventPair
from ..data import SimulatedData
from ..device import SimulatedDevice, ResourceSet, ResourceType
from ..schedulers import SchedulerArchitecture, SystemState
from ..task import *
from ...types import (
    DataID,
    TaskID,
    TaskState,
    Time,
    Device,
    Architecture,
    TaskType,
    Devices,
)

from copy import copy, deepcopy


@dataclass(slots=True)
class Recorder:
    def __getitem__(
        self, time: Time
    ) -> Tuple[List[SchedulerArchitecture], List[SystemState]]:
        raise NotImplementedError()

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        raise NotImplementedError()


@dataclass(slots=True)
class RecorderList:
    recorder_types: List[Type[Recorder]] = field(default_factory=list)
    recorders: List[Recorder] = field(init=False)

    def __post_init__(self):
        self.recorders = []
        for recorder_type in self.recorder_types:
            self.recorders.append(recorder_type())

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        for recorder in self.recorders:
            recorder.save(time, arch_state, system_state, current_event, new_events)


@dataclass(slots=True)
class IdleTime(Recorder):
    idle_time: Dict[Time, Dict[Device, List[Time]]] = field(default_factory=dict)

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        if time not in self.idle_time:
            self.idle_time[time] = {}

        for device in system_state.topology.devices:
            if device.name not in self.idle_time[time]:
                self.idle_time[time][device.name] = []

            self.idle_time[time][device.name].append(device.stats.idle_time_compute)


@dataclass(slots=True)
class ResourceUsage(Recorder):
    memory_usage: Dict[Time, Dict[Device, List[int]]] = field(default_factory=dict)
    vcu_usage: Dict[Time, Dict[Device, List[Fraction]]] = field(default_factory=dict)
    copy_usage: Dict[Time, Dict[Device, List[int]]] = field(default_factory=dict)

    phase: TaskState = TaskState.LAUNCHED

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        resource_pool = system_state.resource_pool

        if time not in self.memory_usage:
            self.memory_usage[time] = {}
        if time not in self.vcu_usage:
            self.vcu_usage[time] = {}
        if time not in self.copy_usage:
            self.copy_usage[time] = {}

        for device in resource_pool.pool:
            if device not in self.memory_usage[time]:
                self.memory_usage[time][device] = []
            if device not in self.vcu_usage[time]:
                self.vcu_usage[time][device] = []
            if device not in self.copy_usage[time]:
                self.copy_usage[time][device] = []

            resources = resource_pool.pool[device][self.phase]

            self.memory_usage[time][device] = resources[ResourceType.MEMORY]
            self.vcu_usage[time][device] = resources[ResourceType.VCU]
            self.copy_usage[time][device] = resources[ResourceType.COPY]


@dataclass(slots=True)
class EventRecorder(Recorder):
    completed_events: Dict[Time, List[Event]] = field(default_factory=dict)
    created_events: Dict[Time, List[Event]] = field(default_factory=dict)

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        if time not in self.completed_events:
            self.completed_events[time] = []
        if time not in self.created_events:
            self.created_events[time] = []

        self.completed_events[time].append(current_event)
        self.created_events[time].extend([e[1] for e in new_events])


@dataclass(slots=True)
class TaskRecord:
    name: TaskID
    type: TaskType = TaskType.COMPUTE
    start_time: int = 0
    end_time: int = 0
    devices: Optional[Devices] = None


@dataclass(slots=True)
class ComputeTaskRecord:
    pass


@dataclass(slots=True)
class DataTaskRecord:
    name: TaskID
    type: TaskType = TaskType.DATA
    start_time: int = 0
    end_time: int = 0
    devices: Optional[Devices] = None
    source: Optional[Device] = None
    data: Optional[DataID] = None
    data_size: Optional[int] = None


@dataclass(slots=True)
class ComputeTaskRecorder(Recorder):
    tasks: Dict[TaskID, TaskRecord] = field(default_factory=dict)

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        if isinstance(current_event, TaskCompleted):
            name = current_event.task
            task = system_state.objects.get_task(name)
            current_time = system_state.time

            if isinstance(task, SimulatedComputeTask):
                if name not in self.tasks:
                    self.tasks[name] = TaskRecord(
                        name,
                        end_time=current_time.duration,
                        devices=task.assigned_devices,
                    )
                else:
                    self.tasks[name].end_time = current_time.duration

        for event_pair in new_events:
            event = event_pair[1]
            if isinstance(event, TaskCompleted):
                name = event.task
                task = system_state.objects.get_task(name)
                if isinstance(task, SimulatedComputeTask):
                    current_time = system_state.time

                    if name not in self.tasks:
                        self.tasks[name] = TaskRecord(
                            name,
                            start_time=current_time.duration,
                            devices=task.assigned_devices,
                        )
                    else:
                        self.tasks[name].start_time = current_time.duration


@dataclass(slots=True)
class DataTaskRecorder(Recorder):
    tasks: Dict[TaskID, DataTaskRecord] = field(default_factory=dict)

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        if isinstance(current_event, TaskCompleted):
            name = current_event.task
            task = system_state.objects.get_task(name)
            current_time = system_state.time

            if isinstance(task, SimulatedDataTask):
                if name not in self.tasks:
                    data_id = task.read_accesses[0].id
                    data = system_state.objects.get_data(data_id)
                    data_size = data.size

                    self.tasks[name] = DataTaskRecord(
                        name,
                        end_time=current_time.duration,
                        devices=task.assigned_devices,
                        source=task.source,
                    )
                else:
                    self.tasks[name].end_time = current_time.duration

        for event_pair in new_events:
            event = event_pair[1]
            if isinstance(event, TaskCompleted):
                name = event.task
                task = system_state.objects.get_task(name)
                if isinstance(task, SimulatedDataTask):
                    current_time = system_state.time

                    if name not in self.tasks:
                        data_id = task.read_accesses[0].id
                        data = system_state.objects.get_data(data_id)
                        data_size = data.size

                        self.tasks[name] = DataTaskRecord(
                            name,
                            start_time=current_time.duration,
                            devices=task.assigned_devices,
                            source=task.source,
                            data=data_id,
                            data_size=data_size,
                        )
                    else:
                        self.tasks[name].start_time = current_time.duration
