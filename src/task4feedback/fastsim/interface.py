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
    PyEFTMapper,
    PyDequeueEFTMapper,
    PyEventType,
    PyTaskNoise,
    PyExternalTaskNoise,
    PyLognormalTaskNoise,
    PyCommunicationNoise,
    PySchedulerInput,
    PyObserver,
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
import torch

import torch_geometric as geom
import gymnasium as gym


class Phase(Enum):
    MAP = 0
    RESERVE = 1
    LAUNCH = 2
    COMPLETE = 3


class TNoiseType:
    NONE = 0
    LOGNORMAL = 1
    USER = 2


class CMapperType:
    STATIC = 0
    ROUND_ROBIN = 1
    EFT_RESERVED = 2
    EFT_DEQUEUE = 3


MAX_VCU = 1000


class ExecutionState:
    RUNNING = PyExecutionState.RUNNING
    BREAKPOINT = PyExecutionState.BREAKPOINT
    ERROR = PyExecutionState.ERROR
    EXTERNAL_MAPPING = PyExecutionState.EXTERNAL_MAPPING

    def __init__(self, state: PyExecutionState):
        self.state = state

    def __int__(self):
        return self.state

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return str(self.state)


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
            self.cdevices.create_device(i, name, 0, vcu, mem)
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

            # location_id = self.device_handle.get_device_id(location)
            # print(location, location_id)
            self.cdata.create_block(i, size, 0, name)

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

    def sample_durations(self):
        return self.noise.generate_durations()

    def sample_priorities(self):
        return self.noise.generate_priorities()

    def dump_durations(self, path: str):
        self.noise.dump_to_binary(path)

    def load_durations(self, path: str):
        self.noise.load_from_binary(path)

    def dump_priorities(self, path: str):
        self.noise.dump_priorities_to_binary(path)

    def load_priorities(self, path: str):
        self.noise.load_priorities_from_binary(path)


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


class EFTCMapper(CMapper):
    def __init__(self, tasks: TaskHandle, devices: DeviceHandle):
        self.mapper = PyEFTMapper(tasks.ctask, devices.cdevices)


class DequeueCMapper(EFTCMapper):
    def __init__(self, tasks: TaskHandle, devices: DeviceHandle):
        self.mapper = PyDequeueEFTMapper(tasks.ctask, devices.cdevices)


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

    def map_tasks(self, candidates: list[int], simulator) -> list[Action]:
        action_list = []
        for i, candidate in enumerate(candidates):
            device = self.mapping[candidate]
            action_list.append(
                Action(
                    candidate,
                    i,
                    device,
                    0,
                    0,
                )
            )
        return action_list


class RoundRobinPythonMapper(PythonMapper):
    def __init__(self, n_devices: int):
        self.n_devices = n_devices

    def map_tasks(self, candidates: np.ndarray[np.uint32], simulator) -> list[Action]:
        action_list = []

        for i, candidate in enumerate(candidates):
            device = candidate % self.n_devices
            action_list.append(
                Action(
                    candidate,
                    i,
                    device,
                    0,
                    0,
                )
            )
            return action_list


class Observer:
    observer: PyObserver

    def __init__(self, simulator: PySimulator):
        self.observer = PyObserver(simulator)
        self.observer.global_features()

    def get_active_tasks(self) -> np.ndarray[np.uint64]:
        return self.observer.get_active_tasks()

    def get_k_hop_dependents(
        self, task_list: np.ndarray[np.uint64], k: int
    ) -> np.ndarray[np.uint64]:
        return self.observer.get_k_hop_dependents(task_list, k)

    def get_k_hop_dependencies(
        self, task_list: np.ndarray[np.uint64], k: int
    ) -> np.ndarray[np.uint64]:
        return self.observer.get_k_hop_dependencies(task_list, k)

    def unique(self, task_list: np.ndarray[np.uint64]):
        return np.unique(task_list)

    def get_task_features(self, tasks: np.ndarray[np.uint64]) -> np.ndarray[np.float64]:
        """
        Get task features

        Args:
            tasks (list[int]): List of task ids

        Returns:
            np.ndarray[np.float64]: Task features
            - f[0]: normalized in_degree
            - f[1]: normalized out_degree
            - f[2]: is mapped
            - f[3]: is reserved
            - f[4]: is launched
            - f[5]: is completed
            - f[6]: is mapping candiate (to be set by user)
        """
        return self.observer.get_task_features(tasks)

    def get_data_features(self, data: np.ndarray[np.uint64]) -> np.ndarray[np.float64]:
        """Get data features

        Args:
            data (list[int]): List of data ids

        Returns:
            np.ndarray[np.float64]: data features
            - f[0]: normalized data size
            - f[1+device_id]: data located on device_id on mapping table
        """
        return self.observer.get_data_features(data)

    def get_device_features(
        self, devices: np.ndarray[np.uint64]
    ) -> np.ndarray[np.float64]:
        """
        Get device features

        Args:
            devices (list[int]): List of device ids

        Returns:
            np.ndarray[np.float64]: Device features
            - f[0]: is cpu
            - f[1]: is gpu
            - f[2]: mapped memory (normalized across devices)
            - f[3]: reserved memory (normalized across devices)
            - f[4]: launched memory (normalized across devices)
            - f[5]: mapped time (normalized across devices)
            - f[6]: reserved time (normalized across devices)
            - f[7]: launched time (normalized across devices)
        """
        return self.observer.get_device_features(devices)

    def get_task_task_edges(
        self, source_tasks: np.ndarray[np.uint64], target_tasks: np.ndarray[np.uint64]
    ) -> tuple[
        list[np.ndarray[np.uint64]],
        np.ndarray[np.float64],
    ]:
        """
        Get task to task edges (dependencies).

        Args:
            source_tasks (list[int]): List of task ids to search for dependencies of.
            target_tasks (list[int]): List of task ids to find dependencies in.

        Returns:
            tuple: A tuple containing:
                - Edges (list[np.ndarray[np.uint64]]): COO local ids.
                - Features (np.ndarray[np.float64]): Feature matrix.
                    - f[0]: Memory shared with target task / source task total memory.
        """
        return self.observer.get_task_task_edges(source_tasks, target_tasks)

    def get_task_data_edges(self, tasks: np.ndarray[np.uint64]) -> tuple[
        np.ndarray[np.uint64],
        list[np.uint64],
        np.ndarray[np.float64],
    ]:
        """Get task to data edges (data usage by task)

        Args:
            tasks (list[int]): List of task ids

        Returns:
            tuple: A tuple containing:
            - Data ID Mapping (np.ndarray[np.uint64]): Local to global data id mapping.
            - Edges (list[np.ndarray[np.uint64]]): COO local ids.
            - Features (np.ndarray[np.float64]): Feature matrix.
                - f[0]: data size / task total memory
                - f[1]: is read access
                - f[2]: is write access
        """
        return self.observer.get_task_data_edges(tasks)

    def get_task_device_edges(self, tasks: np.ndarray[np.uint64]) -> tuple[
        np.ndarray[np.uint64],
        list[np.ndarray[np.uint64]],
        np.ndarray[np.float64],
    ]:
        """
        Get task to device edges (task supports variant on)

        Args:
            tasks (list[int]): List of task ids

        Returns:
            tuple: A tuple containing:
            - Local to global device id mapping (np.ndarray[np.uint64]).
            - Edge list (list[np.ndarray[np.uint64]]): COO format edge list using local ids.
            - Edge features (np.ndarray[np.float64]): Edge features.
                - f[0]: Duration (normalized by global task average duration on architecture).
                - f[1]: Memory cost (normalized by global task average memory cost on architecture).
        """
        return self.observer.get_task_device_edges(tasks)

    def get_data_device_edges(self, tasks: np.ndarray[np.uint64]) -> tuple[
        np.ndarray[np.uint64],
        np.ndarray[np.uint64],
        list[np.ndarray[np.uint64]],
        np.ndarray[np.float64],
    ]:
        """
        Get data to device edges (data located on device).

        Args:
            tasks (list[int]): List of task ids.

        Returns:
            tuple: A tuple containing:
            - Local to global data id mapping (np.ndarray[np.uint64]).
            - Local to global device id mapping (np.ndarray[np.uint64]).
            - Edge list (list[np.ndarray[np.uint64]]): COO format edge list using local ids.
            - Edge features (np.ndarray[np.float64]): Edge features.
                - f[0]: 1 (constant, placeholder).
        """
        return self.observer.get_data_device_edges(tasks)

    def local_graph_features(
        self, candidate_tasks: np.ndarray[np.uint64], k_hop: int = 1
    ):

        # print(active_tasks, active_tasks.dtype)

        g = geom.data.HeteroData()
        if len(candidate_tasks) == 0:
            return g

        candidate_tasks = np.asarray(candidate_tasks, dtype=np.uint32)

        k_hop_dependents = self.get_k_hop_dependents(candidate_tasks, k_hop)
        k_hop_dependencies = self.get_k_hop_dependencies(candidate_tasks, k_hop)

        # print("k_hop_dependents", k_hop_dependents)
        # print("k_hop_dependencies", k_hop_dependencies)

        unique_k_hop = np.unique(np.concatenate([k_hop_dependents, k_hop_dependencies]))
        unique_k_hop = np.asarray(unique_k_hop, dtype=np.uint32)

        all_tasks = np.concatenate([candidate_tasks, unique_k_hop])

        # task_features = self.get_task_features(all_tasks_list)
        task_features = self.get_task_features(all_tasks)
        task_features[: len(candidate_tasks), -1] = 1

        g["tasks"].x = torch.from_numpy(task_features)

        dep_edges, dep_features = self.get_task_task_edges(all_tasks, all_tasks)

        g["tasks", "depends_on", "tasks"].edge_index = torch.from_numpy(dep_edges).to(
            torch.long
        )
        g["tasks", "depends_on", "tasks"].edge_attr = torch.from_numpy(dep_features)

        vdevice2id, task_device_edges, task_device_features = (
            self.get_task_device_edges(all_tasks)
        )

        device_features = self.get_device_features(vdevice2id)
        g["devices"].x = torch.from_numpy(device_features)

        g["devices", "variant", "tasks"].edge_index = (
            torch.from_numpy(task_device_edges).to(torch.long).flip(0)
        )
        g["devices", "variant", "tasks"].edge_attr = torch.from_numpy(
            task_device_features
        )

        data2id, task_data_edges, task_data_features = self.get_task_data_edges(
            all_tasks
        )
        data_features = self.get_data_features(data2id)

        g["data"].x = torch.from_numpy(data_features)
        g["data", "used_by", "tasks"].edge_index = (
            torch.from_numpy(task_data_edges).to(torch.long).flip(0)
        )
        g["data", "used_by", "tasks"].edge_attr = torch.from_numpy(task_data_features)

        g["candidate_list"] = torch.from_numpy(candidate_tasks).to(torch.long)
        g["unique_k_hop"] = torch.from_numpy(unique_k_hop).to(torch.long)

        return g


@dataclass
class Simulator:
    simulator: PySimulator
    noise: TaskNoise
    initialized: bool = False
    pymapper: PythonMapper = field(default_factory=StaticPythonMapper)
    cmapper: CMapper = field(default_factory=StaticCMapper)
    observer: Observer = field(init=False)

    def __post_init__(self):
        self.observer = Observer(self.simulator)

    def use_python_mapper(self, use: bool):
        if use:
            self.simulator.use_python_mapper(True)
        else:
            self.simulator.use_python_mapper(False)

    def set_python_mapper(self, pymapper: PythonMapper):
        self.pymapper = pymapper

    def enable_python_mapper(self):
        self.simulator.use_python_mapper(True)

    def disable_python_mapper(self):
        self.simulator.use_python_mapper(False)

    def set_c_mapper(self, cmapper: CMapper):
        self.cmapper = cmapper
        self.simulator.set_mapper(cmapper.mapper)

    def initialize(self, use_data: bool = True):
        self.simulator.initialize(use_data)
        self.initialized = True

    def step(self, action_list=None):
        if action_list is not None:
            c_action_list = to_c_action_list(action_list)
            self.simulator.map_tasks(c_action_list)
        info = ExecutionState(int(self.simulator.run()))
        obs = self.observer.local_graph_features(
            self.simulator.get_mappable_candidates()
        )
        done = info.state == PyExecutionState.COMPLETE
        terminated = False
        immediate_reward = 0

        return obs, immediate_reward, done, terminated, info

    def get_mapping_candidates(self):
        return self.simulator.get_mappable_candidates()

    def get_observation(self, task_list: list[int]):
        return self.observer.local_graph_features(task_list)

    def run(self) -> PyExecutionState:
        sim_state = PyExecutionState.RUNNING
        while sim_state == PyExecutionState.RUNNING:
            sim_state = self.simulator.run()

            if sim_state == PyExecutionState.BREAKPOINT:
                return sim_state

            if sim_state == PyExecutionState.ERROR:
                return sim_state

            if sim_state == PyExecutionState.EXTERNAL_MAPPING:
                candidates = self.simulator.get_mappable_candidates()
                action_list = self.pymapper.map_tasks(candidates, self)
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

    def randomize_durations(self):
        self.noise.sample_durations()

    def randomize_priorities(self):
        self.noise.sample_priorities()


@dataclass
class SimulatorHandler:
    tasks: TaskMap
    data: DataMap
    devices: Devices
    noise_type: InitVar[TNoiseType] = TNoiseType.NONE
    seed: InitVar[int] = 0
    cmapper_type: InitVar[CMapperType] = CMapperType.STATIC
    cmapper: Optional[CMapper] = None
    pymapper: Optional[PythonMapper] = None
    task_handle: Optional[TaskHandle] = None
    task_noise: Optional[TaskNoise] = None
    comm_noise: Optional[CommunicationNoise] = None
    data_handle: Optional[DataHandle] = None
    device_handle: Optional[DeviceHandle] = None
    input: Optional[PySchedulerInput] = None

    def __post_init__(
        self,
        noise_type: TNoiseType,
        seed: int,
        cmapper_type: CMapperType,
    ):
        if self.device_handle is None:
            self.device_handle = DeviceHandle(self.devices)

        if self.data_handle is None:
            self.data_handle = DataHandle(
                self.device_handle, self.data, self.data_handle
            )

        if self.task_handle is None:
            self.task_handle = TaskHandle(self.data_handle, self.tasks)

        if self.task_noise is None:
            if noise_type == TNoiseType.NONE:
                self.task_noise = TaskNoise(self.task_handle, seed)
            elif noise_type == TNoiseType.LOGNORMAL:
                self.task_noise = LognormalTaskNoise(self.task_handle, seed)
            elif noise_type == TNoiseType.USER:
                self.task_noise = CNumbaTaskNoise(self.task_handle, seed)
            else:
                raise ValueError("Invalid noise type")
        if self.cmapper is None:
            if cmapper_type == CMapperType.STATIC:
                self.cmapper = StaticCMapper()
            elif cmapper_type == CMapperType.ROUND_ROBIN:
                self.cmapper = RoundRobinPythonMapper(len(self.devices.devices))
            elif cmapper_type == CMapperType.EFT_RESERVED:
                self.cmapper = EFTCMapper(self.task_handle, self.device_handle)
            elif cmapper_type == CMapperType.EFT_DEQUEUE:
                self.cmapper = DequeueCMapper(self.task_handle, self.device_handle)
            else:
                raise ValueError("Invalid mapper type")

        self.cmapper_type = cmapper_type

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

    def get_new_c_mapper(self):
        if self.cmapper_type == CMapperType.STATIC:
            return StaticCMapper()
        elif self.cmapper_type == CMapperType.ROUND_ROBIN:
            return RoundRobinPythonMapper(len(self.devices.devices))
        elif self.cmapper_type == CMapperType.EFT_RESERVED:
            return EFTCMapper(self.task_handle, self.device_handle)
        elif self.cmapper_type == CMapperType.EFT_DEQUEUE:
            return DequeueCMapper(self.task_handle, self.device_handle)
        else:
            raise ValueError("Invalid mapper type")

    def set_noise(self, noise_type: TNoiseType, seed: int = 0):
        if noise_type == TNoiseType.NONE:
            self.task_noise = TaskNoise(self.task_handle)
        elif noise_type == TNoiseType.LOGNORMAL:
            self.task_noise = LognormalTaskNoise(self.task_handle)
        elif noise_type == TNoiseType.USER:
            self.task_noise = CNumbaTaskNoise(self.task_handle, seed)
        else:
            raise ValueError("Invalid noise type")

        self.input = PySchedulerInput(
            self.task_handle.ctask,
            self.data_handle.cdata,
            self.device_handle.cdevices,
            self.device_handle.ctopology,
            self.cmapper.mapper,
            self.task_noise.noise,
            self.comm_noise.noise,
        )

    def set_python_mapper(self, pymapper: PythonMapper):
        self.pymapper = pymapper

    def set_c_mapper(self, cmapper: CMapper):
        self.cmapper = cmapper
        self.input = PySchedulerInput(
            self.task_handle.ctask,
            self.data_handle.cdata,
            self.device_handle.cdevices,
            self.device_handle.ctopology,
            self.cmapper.mapper,
            self.task_noise.noise,
            self.comm_noise.noise,
        )

    def create_simulator(self, use_python_mapper=False) -> Simulator:
        internal_sim = PySimulator(self.input)
        sim_wrapper = Simulator(
            internal_sim, self.task_noise, False, self.pymapper, self.cmapper
        )

        if use_python_mapper:
            sim_wrapper.enable_python_mapper()
        else:
            sim_wrapper.disable_python_mapper()

        return sim_wrapper

    def copy(self, simulator: Simulator) -> Simulator:
        internal_sim = simulator.simulator.copy()
        sim_wrapper = Simulator(
            internal_sim,
            simulator.noise,
            simulator.initialized,
            simulator.pymapper,
            self.get_new_c_mapper(),
        )

        return sim_wrapper


def uniform_connected_devices(n_devices: int, mem: int, latency: int, bandwidth: int):
    devices = Devices()
    n_gpus = n_devices - 1
    cpu = Device(Architecture.CPU, 0)

    devices.add_device(cpu, mem)
    for i in range(n_gpus):
        device = Device(Architecture.GPU, i)
        devices.add_device(device, mem)

    for i in range(n_gpus):
        device = Device(Architecture.GPU, i)
        devices.add_connection(cpu, device, latency, bandwidth)
        devices.add_connection(device, cpu, latency, bandwidth)

    for i in range(n_gpus):
        for j in range(n_gpus):
            device1 = Device(Architecture.GPU, i)
            device2 = Device(Architecture.GPU, j)
            devices.add_connection(device1, device2, latency, bandwidth)
            devices.add_connection(device2, device1, latency, bandwidth)

    return devices
