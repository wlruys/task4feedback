"""!
@file graphs/utilities.py
@brief Provides the core functions for representing and generating synthetic task graphs.
"""

from functools import partial
from fractions import Fraction
import numpy as np
import re

from typing import NamedTuple, Union, List, Dict, Tuple, FrozenSet
from dataclasses import dataclass, field
import tempfile
import time
from enum import IntEnum
import random
from collections import defaultdict

from ..legacy_types import *
from ..load import *

graph_generators = []


def default_task_mapping(task_id: TaskID) -> Devices:
    """
    Default task mapping function for a synthetic task graph.
    """
    return None


def round_robin_task_mapping_gpu(
    task_id: TaskID, n_devices: int = 4, index: int = 0
) -> Devices:
    return Device(Architecture.GPU, task_id.task_idx[index] % n_devices)


def default_task_config(task_id: TaskID) -> TaskPlacementInfo:
    """
    Default task configuration function for a synthetic task graph.
    """
    placement = TaskPlacementInfo().add(
        (Device(Architecture.CPU, 0),), TaskRuntimeInfo(task_time=10000)
    )
    return placement


def default_func_id(task_id: TaskID) -> int:
    """
    Default user-defined ID that represents a function type
    that the task mainly calls.
    As a user-defined parameter, this does not have a strict standard.
    This is used in RL to give task type information to the agent.
    """
    return 0


@dataclass(slots=True)
class GraphConfig:
    """
    Configures information about generating the synthetic task graph.

    @field task_config: The runtime information for each task
    @field task_mapping: The device mapping for each task

    """

    n_devices: int = 4
    task_mapping: Callable[[TaskID], Devices] = default_task_mapping
    task_config: Callable[[TaskID], TaskPlacementInfo] = default_task_config
    func_id: Callable[[TaskID], int] = default_func_id


def default_data_initial_placement(data_id: DataID) -> Devices:
    """
    Default data initial placement function for a synthetic task graph.
    """
    return Device(Architecture.CPU, 0)


def round_robin_data_initial_placement_gpu(
    data_id: DataID, n_devices: int = 4, index: int = 0
) -> Devices:
    return Device(Architecture.GPU, data_id.idx[index] % n_devices)


def default_data_sizes(data_id: DataID) -> int:
    """
    Default data initial size function for a synthetic task graph.
    """
    return 0


def default_data_edges(task_id: TaskID) -> TaskDataInfo:
    """
    Default data edges function for a synthetic task graph.
    """
    return TaskDataInfo()


@dataclass(slots=True)
class DataGraphConfig:
    """
    Information about initial data placement for a synthetic task graph.
    """

    name: str = "DataGraph"
    initial_placement: Callable[[DataID], Devices] = default_data_initial_placement
    initial_sizes: Callable[[DataID], int] = default_data_sizes
    edges: Callable[[TaskID], TaskDataInfo] = default_data_edges


@dataclass(slots=True)
class NoDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern that uses no data
    """

    def __post_init__(self):
        self.name = "NoData"
        self.initial_placement = default_data_initial_placement
        self.initial_sizes = default_data_sizes
        self.edges = default_data_edges


def get_mapping(config: GraphConfig, task_idx: TaskID) -> Optional[Devices]:
    mapping_lambda = config.task_mapping
    assert mapping_lambda is not None
    task_mapping = mapping_lambda(task_idx)
    return task_mapping


def register_graph_generator(func):
    """
    Registers a graph generator function to be used for generating synthetic task graphs.
    """
    graph_generators.append(func)
    return func


def shuffle_tasks(tasks: Dict[TaskID, TaskInfo]) -> Dict[TaskID, TaskInfo]:
    """
    Shuffles the task graph
    """
    task_list = list(tasks.values())
    random.shuffle(task_list)
    return convert_to_dictionary(task_list)


def check_config(config: GraphConfig):
    """
    Raise warnings for invalid configuration specifications.
    """
    if config is None:
        raise ValueError(f"Graph Configuration file must be specified: {config}")

    if config.task_config is None:
        raise ValueError(f"Task Configuration file must be specified: {config}")


def get_data_dependencies(
    task_id: TaskID, data_dict: DataMap, data_config: DataGraphConfig
):
    data_dependencies = data_config.edges(task_id)
    for data_id in data_dependencies.all_ids():
        data_placement = data_config.initial_placement(data_id)
        data_size = data_config.initial_sizes(data_id)
        data_dict[data_id] = DataInfo(data_id, data_size, data_placement)

    return data_dependencies, data_dict


def make_graph(
    config: GraphConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    """
    Generates a synthetic task graph based on the provided configuration.
    """
    for func in graph_generators:
        if isinstance(config, func.__annotations__["config"]):
            return func(config, data_config=data_config)

    raise ValueError(f"Invalid graph configuration: {config}")


@dataclass(slots=True)
class DataPlacer:
    """
    Data placer class for placing data on devices.
    """

    cpu_size: int = 0
    """CPU memory size in Bytes"""

    gpu_size: int = 0
    """GPU memory size in Bytes"""

    num_gpus: int = 0
    """Number of GPUs"""

    data_size: Callable[[DataID], int] = default_data_sizes
    """Function to determine data size"""

    stencil_width: int = 1

    device_data_sizes: dict = field(default_factory=dict, init=False)
    device_data_limit: dict = field(default_factory=dict, init=False)
    data_id_to_device: Dict[DataID, Devices] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.device_data_limit[Device(Architecture.CPU, 0)] = self.cpu_size
        self.device_data_sizes[Device(Architecture.CPU, 0)] = 0
        for i in range(self.num_gpus):
            self.device_data_limit[Device(Architecture.GPU, i)] = self.gpu_size
            self.device_data_sizes[Device(Architecture.GPU, i)] = 0

    def rr_2d_gpu_placement(self, data_id: DataID) -> Devices:
        if data_id in self.data_id_to_device:
            return self.data_id_to_device[data_id]

        def factor(n):
            # Pick closest to even split
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return i, n // i
            return n, 1

        grid = data_id.idx[-2:]
        dim_1, dim_2 = factor(self.num_gpus)

        chosen = (grid[1] % dim_2) * dim_1 + (grid[0] % dim_1)

        if (
            self.device_data_sizes[Device(Architecture.GPU, chosen)]
            + self.data_size(data_id)
            >= self.device_data_limit[Device(Architecture.GPU, chosen)]
        ):
            self.data_id_to_device[data_id] = Device(Architecture.CPU, 0)
        else:
            self.device_data_sizes[Device(Architecture.GPU, chosen)] += self.data_size(
                data_id
            )
            self.data_id_to_device[data_id] = Device(Architecture.GPU, chosen)

        return self.data_id_to_device[data_id]

    def rr_gpu_placement(self, data_id: DataID) -> Devices:
        if data_id in self.data_id_to_device:
            return self.data_id_to_device[data_id]

        data_size = self.data_size(data_id)
        candidate_devices = sorted(
            [
                (device, size)
                for device, size in self.device_data_sizes.items()
                if device.architecture == Architecture.GPU
            ],
            key=lambda x: x[1],
        )

        for device, current_size in candidate_devices:
            if current_size + data_size <= self.device_data_limit[device]:
                self.device_data_sizes[device] += data_size
                self.data_id_to_device[data_id] = device
                print(
                    f"data ID: {data_id} size in MB: {data_size / 1024 / 1024} placed: {device}."
                )
                return device

        # If no GPU can accommodate the data, place it on a CPU
        cpu_device = Device(Architecture.CPU, 0)
        self.data_id_to_device[data_id] = cpu_device
        print(
            f"data ID: {data_id} size in MB: {data_size / 1024 / 1024} placed: {cpu_device}."
        )
        return cpu_device

    def random_gpu_placement(self, data_id: DataID) -> Devices:
        if data_id in self.data_id_to_device:
            return self.data_id_to_device[data_id]

        np.random.seed(None)
        data_size = self.data_size(data_id)
        chosen = np.random.randint(0, self.num_gpus)
        if (
            self.device_data_sizes[Device(Architecture.GPU, chosen)] + data_size
            >= self.device_data_limit[Device(Architecture.GPU, chosen)]
        ):
            self.data_id_to_device[data_id] = Device(Architecture.CPU, 0)
        else:
            self.device_data_sizes[Device(Architecture.GPU, chosen)] += data_size
            self.data_id_to_device[data_id] = Device(Architecture.GPU, chosen)

        print(
            f"data ID: {data_id} size in MB: {data_size / 1024 / 1024} placed: {self.data_id_to_device[data_id]}."
        )

        return self.data_id_to_device[data_id]

    def cpu_data_placement(self, data_id: DataID) -> Devices:
        print(
            f"data ID: {data_id} size in MB: {self.data_size(data_id) / 1024 / 1024} placed: {Device(Architecture.CPU, 0)}."
        )
        return Device(Architecture.CPU, 0)

    def optimal_placement(self, data_id: DataID) -> Devices:
        data_size = self.data_size(data_id)
        if self.num_gpus == 2:
            chosen = int(data_id.idx[-2] / (float(self.stencil_width) / 2))
            if (
                self.device_data_sizes[Device(Architecture.GPU, chosen)] + data_size
                >= self.device_data_limit[Device(Architecture.GPU, chosen)]
            ):
                assert "Data size exceeds GPU memory limit. Not allowed on stencil optimal placement"
        elif self.num_gpus == 4:  # Quadrant divide
            chosen = int(data_id.idx[-2] / (float(self.stencil_width) / 2))
            chosen += int(data_id.idx[-1] / (float(self.stencil_width) / 2)) * 2
            if (
                self.device_data_sizes[Device(Architecture.GPU, chosen)] + data_size
                >= self.device_data_limit[Device(Architecture.GPU, chosen)]
            ):
                assert "Data size exceeds GPU memory limit. Not allowed on stencil optimal placement"
        elif self.num_gpus == 8:
            chosen = int(data_id.idx[-2] / (float(self.stencil_width) / 8))
            if (
                self.device_data_sizes[Device(Architecture.GPU, chosen)] + data_size
                >= self.device_data_limit[Device(Architecture.GPU, chosen)]
            ):
                assert "Data size exceeds GPU memory limit. Not allowed on stencil optimal placement"

        return Device(Architecture.GPU, chosen)
