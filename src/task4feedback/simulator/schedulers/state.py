from ..task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from ..data import *
from ..device import *
from ..queue import *
from ..events import *
from ..resources import *
from ..task import *
from ..topology import *

from ...types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ...types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap

from typing import List, Dict, Set, Tuple, Optional, Callable, Type, Sequence
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict
from copy import copy, deepcopy

from rich import print


@dataclass(slots=True)
class ObjectRegistry:
    # Object References (Hashable Name -> Object)
    devicemap: Dict[Device, SimulatedDevice] = field(default_factory=dict)
    taskmap: SimulatedTaskMap = field(default_factory=dict)
    datamap: Dict[DataID, SimulatedData] = field(default_factory=dict)

    def add_task(self, task: SimulatedTask):
        self.taskmap[task.name] = task

    def get_task(self, task_id: Optional[TaskID]) -> SimulatedTask:
        assert task_id is not None
        assert self.taskmap is not None

        if task_id not in self.taskmap:
            raise ValueError(
                f"System state does not have a reference to task: {task_id}."
            )

        task = self.taskmap[task_id]

        if task is None:
            raise ValueError(
                f"System state has a reference to task {task_id} but it is None."
            )

        return task

    def add_data(self, data: SimulatedData):
        self.datamap[data.name] = data

    def get_data(self, data_id: Optional[DataID]) -> SimulatedData:
        assert data_id is not None
        assert self.datamap is not None

        if data_id not in self.datamap:
            raise ValueError(
                f"System state does not have a reference to data: {data_id}."
            )

        data = self.datamap[data_id]

        if data is None:
            raise ValueError(
                f"System state has a reference to data {data_id} but it is None."
            )

        return data

    def add_device(self, device: SimulatedDevice):
        self.devicemap[device.name] = device

    def get_device(self, device_id: Optional[Device]) -> SimulatedDevice:
        assert device_id is not None
        assert self.devicemap is not None

        if device_id not in self.devicemap:
            raise ValueError(
                f"System state does not have a reference to device: {device_id}."
            )

        device = self.devicemap[device_id]

        if device is None:
            raise ValueError(
                f"System state has a reference to device {device_id} but it is None."
            )

        return device


@dataclass(slots=True)
class SystemState:
    topology: SimulatedTopology
    data_pool: DataPool = field(init=False)
    resource_pool: FasterResourcePool = field(init=False)
    objects: ObjectRegistry = field(init=False)
    time: Time = field(default_factory=Time)

    def __post_init__(self):
        assert self.topology is not None

        self.objects = ObjectRegistry()

        for device in self.topology.devices:
            self.objects.add_device(device)

        self.resource_pool = FasterResourcePool(devices=self.topology.devices)

    def register_tasks(self, taskmap: SimulatedTaskMap, copy: bool = False):
        if copy:
            self.objects.taskmap = deepcopy(taskmap)
        else:
            self.objects.taskmap = taskmap

    def register_data(self, datamap: Dict[DataID, SimulatedData], copy: bool = False):
        if copy:
            self.objects.datamap = deepcopy(datamap)
        else:
            self.objects.datamap = datamap

    def register_devices(
        self, devicemap: Dict[Device, SimulatedDevice], copy: bool = False
    ):
        if copy:
            self.objects.devicemap = deepcopy(devicemap)
        else:
            self.objects.devicemap = devicemap

    def check_resources(
        self, phase: TaskState, taskid: TaskID, verbose: bool = False
    ) -> bool:
        # Check that the resources are available
        raise NotImplementedError()

    def acquire_resources(
        self, phase: TaskState, taskid: TaskID, verbose: bool = False
    ):
        # Reserve the resources
        raise NotImplementedError()

    def release_resources(
        self, phase: TaskState, taskid: TaskID, verbose: bool = False
    ):
        # Release the resources
        raise NotImplementedError()

    def use_data(self, phase: TaskState, taskid: TaskID, verbose: bool = False):
        # Update data tracking
        raise NotImplementedError()

    def release_data(self, phase: TaskState, taskid: TaskID, verbose: bool = False):
        # Update data tracking
        raise NotImplementedError()

    def get_task_duration(
        self, task: SimulatedTask, devices: Devices, verbose: bool = False
    ):
        # Get the duration of a task
        raise NotImplementedError()

    def check_task_status(
        self, task: SimulatedTask, status: TaskStatus, verbose: bool = False
    ):
        # Check the status of a task
        raise NotImplementedError()

    def finalize_stats(self):
        raise NotImplementedError()

    def launch_stats(self, task: SimulatedTask):
        raise NotImplementedError()

    def completion_stats(self, task: SimulatedTask):
        raise NotImplementedError()
