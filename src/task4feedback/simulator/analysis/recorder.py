from dataclasses import dataclass, field, InitVar
from typing import Dict, List, Optional, Sequence, Set, Tuple, Self, Type
from fractions import Fraction

from ..events import *
from ..queue import EventPair
from ..data import SimulatedData, DataState
from ..device import SimulatedDevice, ResourceType
from ..resourceset import FasterResourceSet
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

    def finalize(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
    ):
        pass


@dataclass(slots=True)
class RecorderList:
    recorder_dict: Dict[Type[Recorder], Recorder] = field(default_factory=dict)
    recorder_types: List[Type[Recorder]] = field(default_factory=list)
    recorders: List[Recorder] = field(init=False)

    def __post_init__(self):
        self.recorders = []
        for recorder_type in self.recorder_types:
            new_recorder = recorder_type()
            self.recorders.append(new_recorder)
            self.recorder_dict[recorder_type] = new_recorder

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

    def finalize(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
    ):
        for recorder in self.recorders:
            recorder.finalize(time, arch_state, system_state)

    def get(self, type: Type[Recorder]) -> Recorder:
        if type not in self.recorder_dict:
            raise KeyError(f"Recorder {type} not in list of recorders.")

        return self.recorder_dict[type]


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
class ResourceUsageRecorder(Recorder):
    memory_usage: Dict[Device, Dict[Time, int]] = field(default_factory=dict)
    vcu_usage: Dict[Device, Dict[Time, Fraction]] = field(default_factory=dict)
    copy_usage: Dict[Device, Dict[Time, int]] = field(default_factory=dict)
    max_resources: Dict[Device, FasterResourceSet] = field(default_factory=dict)

    phase: TaskState = TaskState.RESERVED

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        resource_pool = system_state.resource_pool

        if len(self.max_resources) == 0:
            for device in system_state.topology.devices:
                self.max_resources[device.name] = device.resources

        for device in resource_pool.pool:
            if device not in self.memory_usage:
                self.memory_usage[device] = {}
            if device not in self.vcu_usage:
                self.vcu_usage[device] = {}
            if device not in self.copy_usage:
                self.copy_usage[device] = {}

            resources = resource_pool.pool[device][self.phase]

            if time not in self.memory_usage[device]:
                self.memory_usage[device][time] = 0
            if time not in self.vcu_usage[device]:
                self.vcu_usage[device][time] = Fraction(0)
            if time not in self.copy_usage[device]:
                self.copy_usage[device][time] = 0

            self.memory_usage[device][time] = max(
                resources.memory, self.memory_usage[device][time]
            )
            self.vcu_usage[device][time] = max(
                resources.vcus, self.vcu_usage[device][time]
            )
            self.copy_usage[device][time] = max(
                resources.copy, self.copy_usage[device][time]
            )


@dataclass(slots=True)
class MappedResourceUsageRecorder(ResourceUsageRecorder):
    phase: TaskState = TaskState.MAPPED


@dataclass(slots=True)
class ReservedResourceUsageRecorder(ResourceUsageRecorder):
    phase: TaskState = TaskState.RESERVED


@dataclass(slots=True)
class LaunchedResourceUsageRecorder(ResourceUsageRecorder):
    phase: TaskState = TaskState.LAUNCHED


@dataclass(slots=True)
class ResourceUsageListRecorder(Recorder):
    memory_usage: Dict[Device, Dict[Time, List[int]]] = field(default_factory=dict)
    vcu_usage: Dict[Device, Dict[Time, List[Fraction]]] = field(default_factory=dict)
    copy_usage: Dict[Device, Dict[Time, List[int]]] = field(default_factory=dict)
    max_resources: Dict[Device, FasterResourceSet] = field(default_factory=dict)

    phase: TaskState = TaskState.RESERVED

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        resource_pool = system_state.resource_pool

        if len(self.max_resources) == 0:
            for device in system_state.topology.devices:
                self.max_resources[device.name] = device.resources

        for device in resource_pool.pool:
            if device not in self.memory_usage:
                self.memory_usage[device] = {}
            if device not in self.vcu_usage:
                self.vcu_usage[device] = {}
            if device not in self.copy_usage:
                self.copy_usage[device] = {}

            resources = resource_pool.pool[device][self.phase]

            if time not in self.memory_usage[device]:
                self.memory_usage[device][time] = []
            if time not in self.vcu_usage[device]:
                self.vcu_usage[device][time] = []
            if time not in self.copy_usage[device]:
                self.copy_usage[device][time] = []

            self.memory_usage[device][time].append(resources.memory)

            self.vcu_usage[device][time].append(resources.vcus)

            self.copy_usage[device][time].append(resources.copy)


@dataclass(slots=True)
class MappedResourceUsageListRecorder(ResourceUsageListRecorder):
    phase: TaskState = TaskState.MAPPED


@dataclass(slots=True)
class ReservedResourceUsageListRecorder(ResourceUsageListRecorder):
    phase: TaskState = TaskState.RESERVED


@dataclass(slots=True)
class LaunchedResourceUsageListRecorder(ResourceUsageListRecorder):
    phase: TaskState = TaskState.LAUNCHED


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
    start_time: Time = Time(0)
    end_time: Time = Time(0)
    devices: Optional[Devices] = None
    read_data: List[DataID] = field(default_factory=list)
    write_data: List[DataID] = field(default_factory=list)
    read_write_data: List[DataID] = field(default_factory=list)


@dataclass(slots=True)
class ComputeTaskRecord(TaskRecord):
    map_time: Time = Time(0)
    reserve_time: Time = Time(0)


@dataclass(slots=True)
class DataTaskRecord(TaskRecord):
    name: TaskID
    type: TaskType = TaskType.DATA
    start_time: Time = Time(0)
    end_time: Time = Time(0)
    devices: Optional[Devices] = None
    source: Optional[Device] = None
    data: Optional[DataID] = None
    data_size: Optional[int] = None
    reserve_time: Time = Time(0)
    communication_energy: float = 0


@dataclass(slots=True)
class ComputeTaskRecorder(Recorder):
    tasks: Dict[TaskID, ComputeTaskRecord] = field(default_factory=dict)

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
            current_time = Time(system_state.time.duration)

            if isinstance(task, SimulatedComputeTask):
                if name not in self.tasks:
                    self.tasks[name] = ComputeTaskRecord(
                        name=name,
                        end_time=current_time,
                        devices=task.assigned_devices,
                        read_data=[d.id for d in task.read_accesses],
                        write_data=[d.id for d in task.write_accesses],
                        read_write_data=[d.id for d in task.read_write_accesses],
                        reserve_time=task.times.state_times[TaskState.RESERVED],
                        map_time=task.times.state_times[TaskState.MAPPED],
                    )
                else:
                    self.tasks[name].end_time = current_time

        for event_pair in new_events:
            event = event_pair[1]
            if isinstance(event, TaskCompleted):
                name = event.task
                task = system_state.objects.get_task(name)
                if isinstance(task, SimulatedComputeTask):
                    current_time = Time(system_state.time.duration)

                    if name not in self.tasks:
                        self.tasks[name] = ComputeTaskRecord(
                            name,
                            start_time=current_time,
                            devices=task.assigned_devices,
                            read_data=[d.id for d in task.read_accesses],
                            write_data=[d.id for d in task.write_accesses],
                            read_write_data=[d.id for d in task.read_write_accesses],
                            reserve_time=task.times.state_times[TaskState.RESERVED],
                            map_time=task.times.state_times[TaskState.MAPPED],
                        )
                    else:
                        self.tasks[name].start_time = current_time


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
            current_time = Time(system_state.time.duration)

            if isinstance(task, SimulatedDataTask):
                if name not in self.tasks:
                    data_id = task.read_accesses[0].id
                    data = system_state.objects.get_data(data_id)
                    data_size = data.size
                    devices = task.assigned_devices
                    communication_energy = 0
                    print("In recorder end")
                    if(task.real): # check if there is any data movement
                        print("In if end")
                        for device in devices:
                            communication_energy += device.energy * data_size
                            print(communication_energy)
                    self.tasks[name] = DataTaskRecord(
                        name,
                        end_time=current_time,
                        devices=task.assigned_devices,
                        source=task.source,
                        reserve_time=task.times.state_times[TaskState.RESERVED],
                        communication_energy=communication_energy,
                    )
                else:
                    self.tasks[name].end_time = current_time

        for event_pair in new_events:
            event = event_pair[1]
            if isinstance(event, TaskCompleted):
                name = event.task
                task = system_state.objects.get_task(name)
                if isinstance(task, SimulatedDataTask):
                    current_time = Time(system_state.time.duration)

                    if name not in self.tasks:
                        data_id = task.read_accesses[0].id
                        data = system_state.objects.get_data(data_id)
                        data_size = data.size
                        devices = task.assigned_devices
                        communication_energy = 0
                        #print("In recorder")
                        if(task.real): # check if there is any data movement
                            #print("In if")
                            for device in devices:
                                communication_energy += device.energy * data_size
                                #print(communication_energy)

                        self.tasks[name] = DataTaskRecord(
                            name,
                            start_time=current_time,
                            devices=task.assigned_devices,
                            source=task.source,
                            data=data_id,
                            data_size=data_size,
                            reserve_time=task.times.state_times[TaskState.RESERVED],
                            communication_energy=communication_energy,
                        )
                    else:
                        self.tasks[name].start_time = current_time


@dataclass(slots=True)
class ValidInterval:
    name: DataID
    start_time: Optional[Time]
    end_time: Optional[Time]


@dataclass(slots=True)
class FasterDataValidRecorder(Recorder):
    valid: Dict[DataID, Dict[Device, bool]] = field(default_factory=dict)
    current_interval: Dict[DataID, Dict[Device, Optional[ValidInterval]]] = field(
        default_factory=dict
    )
    intervals: Dict[DataID, Dict[Device, List[ValidInterval]]] = field(
        default_factory=dict
    )

    def _update_valid(self, time: Time, valid_locations: Set[Device], data_id: DataID):
        for device, old_value in self.valid[data_id].items():
            if device in valid_locations:
                if old_value is False:
                    # Start a new interval
                    current_interval = ValidInterval(
                        name=data_id, start_time=time, end_time=None
                    )
                    self.current_interval[data_id][device] = current_interval

                self.valid[data_id][device] = True
            else:
                if old_value is True:
                    # End the current interval
                    current_interval = self.current_interval[data_id][device]

                    assert (
                        current_interval is not None
                    ), f"Cannot end an interval that does not exist. {time} {data_id} {device}"

                    current_interval.end_time = time
                    self.intervals[data_id][device].append(current_interval)
                    self.current_interval[data_id][device] = None

                self.valid[data_id][device] = False

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        time = Time(system_state.time.duration)

        if len(self.valid) == 0:
            # Record the initial state of the data
            for data in system_state.objects.datamap.values():
                data_id = data.name

                self.valid[data_id] = {}
                self.intervals[data_id] = {}
                self.current_interval[data_id] = {}

                valid_sources_ids = data.get_device_set_from_states(
                    TaskState.LAUNCHED, DataState.VALID
                )

                for device in system_state.topology.devices:
                    device_id = device.name
                    self.intervals[data_id][device_id] = []

                    if device_id not in self.valid[data_id]:
                        self.valid[data_id][device_id] = False
                        self.current_interval[data_id][device_id] = None

                    else:
                        self.valid[data_id][device.name] = True
                        current_interval = ValidInterval(
                            name=data_id, start_time=time, end_time=None
                        )
                        self.current_interval[data_id][device.name] = current_interval
        else:
            data_set = set()
            if isinstance(current_event, TaskCompleted):
                taskid = current_event.task
                task = system_state.objects.get_task(taskid)
                data_set.update(data_access.id for data_access in task.read_accesses)
                data_set.update(data_access.id for data_access in task.write_accesses)
                data_set.update(
                    data_access.id for data_access in task.read_write_accesses
                )
            elif isinstance(current_event, Launcher):
                for completion_time, new_event in new_events:
                    if isinstance(new_event, TaskCompleted):
                        task = system_state.objects.get_task(new_event.task)
                        data_set.update(
                            data_access.id for data_access in task.read_accesses
                        )
                        data_set.update(
                            data_access.id for data_access in task.write_accesses
                        )
                        data_set.update(
                            data_access.id for data_access in task.read_write_accesses
                        )

            for data_id in data_set:
                data = system_state.objects.get_data(data_id)
                valid_sources_ids = data.get_device_set_from_states(
                    TaskState.LAUNCHED, DataState.VALID
                )
                self._update_valid(time, valid_sources_ids, data_id)

    def finalize(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
    ):
        # End all current intervals
        for data_id in self.current_interval:
            for device in self.current_interval[data_id]:
                current_interval = self.current_interval[data_id][device]
                if current_interval is not None:
                    current_interval.end_time = time
                    self.intervals[data_id][device].append(current_interval)


@dataclass(slots=True)
class DataValidRecorder(Recorder):
    valid: Dict[DataID, Dict[Device, bool]] = field(default_factory=dict)
    current_interval: Dict[DataID, Dict[Device, ValidInterval]] = field(
        default_factory=dict
    )
    intervals: Dict[DataID, Dict[Device, List[ValidInterval]]] = field(
        default_factory=dict
    )
    phase: TaskState = TaskState.LAUNCHED

    def __post_init__(self):
        self.valid = {}

    def save(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
        current_event: Event,
        new_events: Sequence[EventPair],
    ):
        time = Time(system_state.time.duration)

        for data in system_state.objects.datamap.values():
            data_id = data.name

            if data_id not in self.valid:
                self.valid[data_id] = {}

            if data_id not in self.intervals:
                self.intervals[data_id] = {}

            if data_id not in self.current_interval:
                self.current_interval[data_id] = {}

            valid_sources_ids = data.get_devices_from_states(
                [TaskState.LAUNCHED], [DataState.VALID]
            )

            for device in system_state.topology.devices:
                device = device.name
                if device not in self.valid[data_id]:
                    self.valid[data_id][device] = False

                if device not in self.intervals[data_id]:
                    self.intervals[data_id][device] = []

                if device in valid_sources_ids:
                    old = self.valid[data_id][device]
                    self.valid[data_id][device] = True

                    if old is False:
                        if data_id in self.current_interval:
                            if device in self.current_interval[data_id]:
                                current_interval = self.current_interval[data_id][
                                    device
                                ]
                                self.intervals[data_id][device].append(current_interval)

                        self.current_interval[data_id][device] = ValidInterval(
                            name=data_id, start_time=time, end_time=None
                        )

                else:
                    old = self.valid[data_id][device]
                    self.valid[data_id][device] = False
                    create_flag = False
                    if old is True:
                        # print("Ending interval", data_id, device)
                        if data_id in self.current_interval:
                            if device in self.current_interval[data_id]:
                                current_interval = self.current_interval[data_id][
                                    device
                                ]
                                if current_interval.start_time is not None:
                                    current_interval.end_time = time
                                # print(
                                #     "Internal Ending interval",
                                #     data_id,
                                #     device,
                                #     current_interval,
                                # )
                            else:
                                create_flag = True
                        else:
                            create_flag = True

                        if create_flag:
                            # print("Creating new interval", data_id, device)
                            current_interval = ValidInterval(
                                name=data_id, start_time=Time(0), end_time=time
                            )
                            self.current_interval[data_id][device] = current_interval

    def finalize(
        self,
        time: Time,
        arch_state: SchedulerArchitecture,
        system_state: SystemState,
    ):
        # print("Before finalize")
        # print(self)
        for data_id in self.current_interval:
            for device in self.current_interval[data_id]:
                current_interval = self.current_interval[data_id][device]
                if current_interval.end_time is None:
                    current_interval.end_time = time
                self.intervals[data_id][device].append(current_interval)
