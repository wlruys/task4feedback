from ..task import SimulatedTask, SimulatedDataTask, SimulatedComputeTask
from ..data import *
from ..device import *
from ..queue import *
from ..events import *
from ..resources import *
from ..task import *
from ..topology import *
from ..utility import *
from ..randomizer import Randomizer

from ...types import Architecture, Device, TaskID, TaskState, TaskType, Time
from ...types import TaskRuntimeInfo, TaskPlacementInfo, TaskMap, ExecutionMode
from ...types import TaskOrderType

from ..rl.models.model import *
from ..rl.models.env import *

from typing import List, Dict, Set, Tuple, Optional, Callable, Type, Sequence
from dataclasses import dataclass, InitVar
from collections import defaultdict as DefaultDict
from copy import copy, deepcopy

import os

# from rich import print


@dataclass(slots=True)
class ObjectRegistry:
    # Object References (Hashable Name -> Object)
    devicemap: Dict[Device, SimulatedDevice] = field(default_factory=dict)
    taskmap: SimulatedTaskMap = field(default_factory=dict)
    datamap: Dict[DataID, SimulatedData] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        s = clock()
        devicemap = {k: deepcopy(v) for k, v in self.devicemap.items()}
        # print(f"Time to deepcopy devicemap: {clock() - s}")

        s = clock()
        taskmap = {k: deepcopy(v) for k, v in self.taskmap.items()}
        # print(f"Time to deepcopy taskmap: {clock() - s}")

        s = clock()
        datamap = {k: deepcopy(v) for k, v in self.datamap.items()}
        # print(f"Time to deepcopy datamap: {clock() - s}")
        return ObjectRegistry(devicemap=devicemap, taskmap=taskmap, datamap=datamap)

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


from time import perf_counter as clock


@dataclass(slots=True)
class SystemState:
    randomizer: Randomizer
    topology: SimulatedTopology
    task_order_mode: TaskOrderType
    data_pool: DataPool | None = None
    resource_pool: FasterResourcePool | None = None
    objects: ObjectRegistry | None = None
    time: Time = field(default_factory=Time)
    init: bool = True
    use_eviction: bool = True
    use_duration_noise: bool = False
    noise_scale: float = 0
    save_task_order: bool = False
    load_task_order: bool = False
    save_task_noise: bool = False
    load_task_noise: bool = False
    loaded_task_noises: Dict[str, int] | None = None
    wait_time_accum: Time = field(default_factory=Time)
    num_tasks: float = 0

    init: bool = True

    # RL environment providing RL state and performing auxiliary operations
    # TODO(hc): make these specific to RLState
    rl_env: RLBaseEnvironment = None
    rl_mapper: RLModel = None

    def __deepcopy__(self, memo):
        s = clock()

        topology = deepcopy(self.topology)
        # print(f"Time to deepcopy topology: {clock() - s}")

        s = clock()
        data_pool = deepcopy(self.data_pool)
        # print(f"Time to deepcopy data_pool: {clock() - s}")

        s = clock()
        resource_pool = deepcopy(self.resource_pool)
        # print(f"Time to deepcopy resource_pool: {clock() - s}")

        s = clock()
        objects = deepcopy(self.objects)
        # print(f"Time to deepcopy objects: {clock() - s}")

        s = clock()
        time = deepcopy(self.time)
        # print(f"Time to deepcopy time: {clock() - s}")

        s = clock()
        loaded_task_noises = deepcopy(self.loaded_task_noises)

        s = clock()
        rl_env = deepcopy(self.rl_env)

        s = clock()
        rl_mapper = deepcopy(self.rl_mapper)

        return SystemState(
            randomizer=self.randomizer,
            topology=topology,
            task_order_mode=self.task_order_mode,
            data_pool=data_pool,
            resource_pool=resource_pool,
            objects=objects,
            time=time,
            init=self.init,
            use_eviction=self.use_eviction,
            use_duration_noise=self.use_duration_noise,
            noise_scale=self.noise_scale,
            save_task_order=self.save_task_order,
            load_task_order=self.load_task_order,
            save_task_noise=self.save_task_noise,
            load_task_noise=self.load_task_noise,
            loaded_task_noises=loaded_task_noises,
            wait_time_accum=self.wait_time_accum,
            num_tasks=self.num_tasks,
            rl_env=rl_env,
            rl_mapper=rl_mapper,
        )

    def __post_init__(self):
        assert self.topology is not None

        if self.init:
            if self.objects is None:
                self.objects = ObjectRegistry()

                for device in self.topology.devices:
                    self.objects.add_device(device)

            if self.resource_pool is None:
                self.resource_pool = FasterResourcePool(devices=self.topology.devices)
            self.init = False

        if self.save_task_order:
            if os.path.exists("replay.order"):
                print("replay.order is removed..")
                os.remove("replay.order")

        if self.save_task_noise:
            if os.path.exists("replay.noise"):
                print("replay.noise is removed..")
                os.remove("replay.noise")

        if self.load_task_noise:
            self.loaded_task_noises = load_task_noise()
            if self.loaded_task_noises is None:
                self.load_task_noise = False
                self.use_duration_noise = False
                self.save_task_noise = False

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

    def get_task_duration_completion(
        self, task: SimulatedTask, devices: Devices, verbose: bool = False
    ):
        # Get the duration of a task
        raise NotImplementedError()

    def check_task_status(
        self, task: SimulatedTask, status: TaskStatus, verbose: bool = False
    ):
        # Check the status of a task
        raise NotImplementedError()

    def initialize(self, task_ids: List[TaskID], task_objects: List[SimulatedTask]):
        raise NotImplementedError()

    def complete(self):
        raise NotImplementedError()
