from typing import Optional, Self
from task4feedback.fastsim.simulator import (
    PyTasks,
    PyData,
    PyDevices,
    PyTopology,
    PyDeviceType,
    PySimulator,
    PyAction,
    PyExecutionState,
    PyStaticMapper,
    PyEventType,
    PyTaskNoise,
    PyExternalTaskNoise,
    PyLognormalTaskNoise,
    PyCommunicationNoise,
    PySchedulerInput,
    start_logger,
)
from task4feedback.types import (
    DataID,
    TaskID,
    TaskInfo,
    DataInfo,
    Device,
    Architecture,
    TaskMap,
    DataMap,
)
from task4feedback.graphs import *
from dataclasses import dataclass, field
import numpy as np
from enum import IntEnum, Enum


class Phase(Enum):
    MAP = 0
    RESERVE = 1
    LAUNCH = 2
    COMPLETE = 3


MAX_VCU = 1000


@dataclass(slots=True)
class Devices:
    devices: dict[Device, tuple[int, int]] = field(default_factory=dict)
    connections: dict[(tuple[Device, Device], tuple[int, int])] = field(
        default_factory=dict
    )

    def add_device(self, device: Device, mem: int, vcu: int = MAX_VCU):
        self.devices[device] = (vcu, mem)

    def add_connection(
        self, device1: Device, device2: Device, latency: int, bandwidth: int
    ):
        self.connections[(device1, device2)] = (latency, bandwidth)


@dataclass(slots=True)
class DeviceHandle:
    devices: Devices
    devices_to_ids: dict[Device, int] = field(default_factory=dict)
    ids_to_devices: dict[int, Device] = field(default_factory=dict)
    cdevices: PyDevices = field(init=False)
    ctopology: PyTopology = field(init=False)

    def __post_init__(self):
        self.cdevices = PyDevices(len(self.devices.devices))
        for i, (device, (vcu, mem)) in enumerate(self.devices.devices.items()):
            name = str(device)
            arch = device.architecture.value
            self.cdevices.create_device(i, name, arch, vcu, mem)
            self.devices_to_ids[device] = i
            self.ids_to_devices[i] = device

        self.ctopology = PyTopology(len(self.devices.devices))

        for device1, device2 in self.devices.connections:
            device1_id = self.devices_to_ids[device1]
            device2_id = self.devices_to_ids[device2]
            latency, bandwidth = self.devices.connections[(device1, device2)]
            self.ctopology.set_bandwidth(device1_id, device2_id, bandwidth)
            self.ctopology.set_latency(device1_id, device2_id, latency)
            self.ctopology.set_max_connections(device1_id, device2_id, 1)

    def __len__(self) -> int:
        return len(self.devices)

    def __getitem__(self, device_id: int) -> Device:
        return self.ids_to_devices[device_id]

    def __iter__(self):
        return iter(self.devices)

    def __contains__(self, device: Device) -> bool:
        return device in self.devices

    def get_device_id(self, device: Device) -> int:
        return self.devices_to_ids[device]

    def get_device_index(self, device_id: int) -> Device:
        return self.ids_to_devices[device_id]

    def get_connection(self, device1: Device, device2: Device) -> tuple[int, int, int]:
        return self.connections[(device1, device2)]


@dataclass(slots=True)
class DataHandle:
    device_handle: DeviceHandle
    datamap: DataMap
    cdata: PyData = field(init=False)
    ids_to_data: dict[int, DataID] = field(default_factory=dict)
    data_to_ids: dict[DataID, int] = field(default_factory=dict)

    def __post_init__(self):
        self.ids_to_data = {}
        self.data_to_ids = {}

        for i, data_id in enumerate(self.datamap):
            self.ids_to_data[i] = data_id
            self.data_to_ids[data_id] = i

        self.cdata = PyData(len(self.datamap))

        for i, data_id in enumerate(self.datamap):
            name = str(data_id)
            size = self.datamap[data_id].size
            location = self.datamap[data_id].location

            if not isinstance(location, Device):
                location = location[0]

            location_id = self.device_handle.get_device_id(location)
            self.cdata.create_block(i, size, location_id, name)

    def get_data_id(self, data_id: int) -> DataID:
        return self.ids_to_data[data_id]

    def get_data_index(self, data_id: DataID) -> int:
        return self.data_to_ids[data_id]

    def __len__(self) -> int:
        return len(self.datamap)

    def __getitem__(self, data_id: DataID) -> DataInfo:
        return self.datamap[data_id]

    def __iter__(self):
        return iter(self.datamap)

    def __contains__(self, data_id: DataID) -> bool:
        return data_id in self.datamap


@dataclass(slots=True)
class TaskHandle:
    data: DataHandle
    taskmap: TaskMap
    ctask: PyTasks = field(init=False)
    ids_to_task: dict[int, TaskID] = field(default_factory=dict)
    task_to_ids: dict[TaskID, int] = field(default_factory=dict)

    def __post_init__(self):
        self.task_to_ids = {}
        self.ids_to_task = {}

        for i, task_id in enumerate(self.taskmap):
            self.ids_to_task[i] = task_id
            self.task_to_ids[task_id] = i

        self.ctask = PyTasks(len(self.taskmap))

        for i, task_id in enumerate(self.taskmap):
            task = self.taskmap[task_id]
            dependencies = [self.task_to_ids[dep] for dep in task.dependencies]
            self.ctask.create_task(i, str(task_id), dependencies)

            read = [d.id for d in task.data_dependencies.read]
            write = [d.id for d in task.data_dependencies.write]
            read_write = [d.id for d in task.data_dependencies.read_write]

            read_set = set(read).union(set(read_write))
            write_set = set(write).union(set(read_write))

            read_ids = [self.data.get_data_index(d) for d in read_set]
            write_ids = [self.data.get_data_index(d) for d in write_set]

            self.ctask.add_read_set(i, read_ids)
            self.ctask.add_write_set(i, write_ids)

            if Device(Architecture.CPU, 0) in task.runtime:
                placement_info = task.runtime[Device(Architecture.CPU, 0)][0]
                vcu = int(placement_info.device_fraction * MAX_VCU)
                self.ctask.add_variant(
                    i,
                    PyDeviceType.CPU,
                    vcu,
                    placement_info.memory,
                    placement_info.task_time,
                )

            if Device(Architecture.GPU, -1) in task.runtime:
                placement_info = task.runtime[Device(Architecture.GPU, -1)][0]
                vcu = int(placement_info.device_fraction * MAX_VCU)
                self.ctask.add_variant(
                    i,
                    PyDeviceType.GPU,
                    vcu,
                    placement_info.memory,
                    placement_info.task_time,
                )

    def get_task_id(self, task_id: int) -> TaskID:
        return self.ids_to_task[task_id]

    def get_task_index(self, task_id: TaskID) -> int:
        return self.task_to_ids[task_id]

    def __len__(self) -> int:
        return len(self.taskmap)

    def __getitem__(self, task_id: TaskID) -> TaskInfo:
        return self.taskmap[task_id]

    def __iter__(self):
        return iter(self.taskmap)

    def __contains__(self, task_id: TaskID) -> bool:
        return task_id in self.taskmap

    def get_name(self, task_id: TaskID) -> str:
        return str(task_id)


class TaskNoise:
    def __init__(self, tasks: TaskHandle, seed: int = 0):
        self.tasks = tasks
        self.noise = PyTaskNoise(tasks.ctask, seed)

    def generate(self):
        return self.noise.generate()

    def dump(self, path: str):
        self.noise.dump_to_binary(path)

    def load(self, path: str):
        self.noise.load_from_binary(path)


class CNumbaTaskNoise(TaskNoise):
    def __init__(self, tasks: TaskHandle, seed: int):
        self.tasks = tasks
        self.noise = PyExternalTaskNoise(tasks.ctask, seed)

    def set_cfunc(self, cfunc):
        self.noise.set_function(cfunc)


class LognormalTaskNoise(TaskNoise):
    def __init__(self, tasks: TaskHandle, seed: int = 0):
        self.tasks = tasks
        self.noise = PyLognormalTaskNoise(tasks.ctask, seed)


class CommunicationNoise:
    def __init__(self, devices: DeviceHandle, seed: int = 0):
        self.devices = devices
        self.noise = PyCommunicationNoise(devices.ctopology, seed)

    def dump(self, path: str):
        self.noise.dump_to_binary(path)

    def load(self, path: str):
        self.noise.load_from_binary(path)


class CMapper:
    pass


class StaticCMapper:

    def __init__(self):
        self.mapper = PyStaticMapper()

    def set_mapping(self, mapping: np.ndarray[np.uint64]):
        self.mapper.set_mapping(mapping)

    def set_launching_priorities(self, priorities: np.ndarray[np.uint64]):
        self.mapper.set_launching_priorities(priorities)

    def set_mapping_priorities(self, priorities: np.ndarray[np.uint64]):
        self.mapper.set_mapping_priorities(priorities)


class Action:
    def __init__(
        self,
        taskid: int,
        index: int,
        device: int,
        reserve_priority: int,
        launch_priority: int,
    ):
        self.taskid = taskid
        self.index = index
        self.device = device
        self.reserve_priority = reserve_priority
        self.launch_priority = launch_priority

        self.caction = PyAction(
            taskid, index, device, reserve_priority, launch_priority
        )


def to_c_action_list(action_list: list[Action]) -> list[PyAction]:
    if not action_list:
        return []
    return [action.caction for action in action_list]


class PythonMapper:
    pass


class StaticPythonMapper:

    def __init__(self):
        self.mapping = None

    def set_mapping(self, mapping: np.ndarray[np.uint64]):
        self.mapping = mapping

    def map_tasks(self, candidates: list[int]) -> list[Action]:
        action_list = []
        for i, candidate in enumerate(candidates):
            device = self.mapping[candidate]
            action_list.append(Action(candidate, i, device, 0, 0))
        return action_list


class RoundRobinPythonMapper(PythonMapper):

    def __init__(self, n_devices: int):
        self.n_devices = n_devices

    def map_tasks(self, candidates: list[int]) -> list[Action]:
        action_list = []
        for i, candidate in enumerate(candidates):
            device = candidate % self.n_devices
            action_list.append(Action(candidate, i, device, 0, 0))
            return action_list


@dataclass
class Simulator:
    tasks: TaskMap
    data: DataMap
    devices: Devices
    cmapper: CMapper = field(default_factory=StaticCMapper)
    pymapper: PythonMapper = field(default_factory=StaticPythonMapper)
    task_handle: Optional[TaskHandle] = None
    task_noise: Optional[TaskNoise] = None
    comm_noise: Optional[CommunicationNoise] = None
    data_handle: Optional[DataHandle] = None
    device_handle: Optional[DeviceHandle] = None
    input: Optional[PySchedulerInput] = None
    simulator: Optional[PySimulator] = None

    def __post_init__(self):
        if self.device_handle is None:
            self.device_handle = DeviceHandle(self.devices)

        if self.data_handle is None:
            self.data_handle = DataHandle(
                self.device_handle, self.data, self.data_handle
            )

        if self.task_handle is None:
            self.task_handle = TaskHandle(self.data_handle, self.tasks)

        if self.task_noise is None:
            self.task_noise = TaskNoise(self.task_handle)

        if self.comm_noise is None:
            self.comm_noise = CommunicationNoise(self.device_handle)

        if self.input is None:
            self.input = PySchedulerInput(
                self.task_handle.ctask,
                self.data_handle.cdata,
                self.device_handle.cdevices,
                self.device_handle.ctopology,
                self.cmapper.mapper,
                self.task_noise.noise,
                self.comm_noise.noise,
            )

        if self.simulator is None:
            self.simulator = PySimulator(self.input)

        self.initialized = False

    def use_python_mapper(self, use: bool):
        if use:
            self.simulator.use_python_mapper(True)
        else:
            self.simulator.use_python_mapper(False)

    def set_python_mapper(self, pymapper: PythonMapper):
        self.pymapper = pymapper

    def set_c_mapper(self, cmapper: CMapper):
        self.cmapper = cmapper
        self.simulator.set_mapper(cmapper.mapper)

    def initialize(self, use_data: bool = True):
        self.simulator.initialize(use_data)
        self.initialized = True

    def copy(self):
        sim = Simulator(
            self.tasks,
            self.data,
            self.devices,
            self.cmapper,
            self.pymapper,
            self.task_handle,
            self.task_noise,
            self.comm_noise,
            self.data_handle,
            self.device_handle,
            self.input,
            self.simulator.copy(),
        )
        sim.initialized = True
        return sim

    def copy_bare(self):
        return self.simulator.copy()

    def run(self) -> PyExecutionState:
        sim_state = PyExecutionState.RUNNING
        while sim_state == PyExecutionState.RUNNING:
            sim_state = self.simulator.run()

            if sim_state == PyExecutionState.BREAKPOINT:
                return sim_state

            if sim_state == PyExecutionState.ERROR:
                return sim_state

            if sim_state == PyExecutionState.PYTHON_MAPPING:
                candidates = self.simulator.get_mappable_candidates()
                action_list = self.pymapper.map_tasks(candidates)
                c_action_list = to_c_action_list(action_list)
                self.simulator.map_tasks(c_action_list)
                sim_state = PyExecutionState.RUNNING

        return sim_state

    def get_current_time(self):
        return self.simulator.get_current_time()

    def add_time_breakpoint(self, time: int):
        self.simulator.add_time_breakpoint(time)

    def add_task_breakpoint(self, event: Phase, task_id: int):
        if event == Phase.MAP:
            etype = PyEventType.MAPPER
        elif event == Phase.RESERVE:
            etype = PyEventType.RESERVER
        elif event == Phase.LAUNCH:
            etype = PyEventType.LAUNCHER
        elif event == Phase.COMPLETE:
            etype = PyEventType.COMPLETER

        self.simulator.add_task_breakpoint(etype, task_id)


def uniform_connected_devices(n_gpus: int, mem: int, latency: int, bandwidth: int):
    devices = Devices()
    cpu = Device(Architecture.CPU, 0)
    devices.add_device(cpu, mem)
    for i in range(n_gpus):
        device = Device(Architecture.GPU, i)
        devices.add_connection(cpu, device, latency, bandwidth)
        devices.add_connection(device, cpu, latency, bandwidth)

    for i in range(n_gpus):
        device = Device(Architecture.GPU, i)
        devices.add_device(device, mem)

    for i in range(n_gpus):
        for j in range(n_gpus):
            device1 = Device(Architecture.GPU, i)
            device2 = Device(Architecture.GPU, j)
            devices.add_connection(device1, device2, latency, bandwidth)
            devices.add_connection(device2, device1, latency, bandwidth)

    return devices


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", type=int, default=3)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--vcus", type=int, default=1)
args = parser.parse_args()


def task_config(task_id: TaskID) -> TaskPlacementInfo:
    placement_info = TaskPlacementInfo()
    placement_info.add(
        (Device(Architecture.GPU, -1),),
        TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
    )
    placement_info.add(
        (Device(Architecture.CPU, -1),),
        TaskRuntimeInfo(task_time=1000, device_fraction=args.vcus),
    )
    return placement_info


data_config = CholeskyDataGraphConfig(data_size=10000)
config = CholeskyConfig(blocks=args.blocks, task_config=task_config)
tasks, data = make_graph(config, data_config=data_config)

# 16GB in bytes
mem = 16 * 1024 * 1024 * 1024
bandwidth = 100
latency = 1
n_devices = args.devices
devices = uniform_connected_devices(n_devices, mem, latency, bandwidth)


# start_logger()
sim = Simulator(tasks, data, devices)
sim.initialize(True)
sim.set_python_mapper(RoundRobinPythonMapper(n_devices))
sim.cmapper.set_mapping(
    np.array([i % n_devices for i in range(len(tasks))], dtype=np.uint64)
)
sim.use_python_mapper(False)

sim2 = sim.copy()

state = sim2.run()
print(state)
print(sim2.get_current_time())
print(sim.get_current_time())

state = sim.run()
print(state)
print(sim.get_current_time())
