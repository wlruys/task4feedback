"""!
@file execute.py
@brief Provides mechanisms to launch and log synthetic task graphs.
"""

import functools
import threading
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass, field

from .threads import Propagate

from .graphs import *

import os
import tempfile
from enum import Enum
import time
import itertools

from parla import Parla, spawn, TaskSpace, parray
from parla import sleep_gil
from parla import sleep_nogil
from parla.common.array import clone_here
from parla.common.globals import (
    get_current_devices,
    get_current_stream,
    cupy,
    CUPY_ENABLED,
    get_current_context,
)
from parla.common.parray.from_data import asarray
from parla.cython.device_manager import cpu, gpu
from parla.cython.variants import specialize
from parla import gpu_sleep_nogil
from parla.cython.core import gpu_bsleep_nogil, gpu_bsleep_gil
import numpy as np

from fractions import Fraction

PArray = parray.core.PArray


def make_parrays(data_list):
    parray_list = list()
    for i, data in enumerate(data_list):
        parray_list.append(asarray(data, name="data" + str(i)))
    return parray_list


def estimate_frequency(n_samples=10, ticks=1900000000):
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    cycles = ticks
    device_id = 0

    print(f"Starting GPU Frequency benchmark.")
    times = np.zeros(n_samples)
    for i in range(n_samples):
        start = time.perf_counter()
        gpu_bsleep_nogil(device_id, int(ticks), stream)
        stream.synchronize()
        end = time.perf_counter()
        print(f"...collected frequency sample {i} ", end - start)

        times[i] = end - start

    times = times[2:]
    elapsed = np.mean(times)
    estimated_speed = cycles / np.mean(times)
    median_speed = cycles / np.median(times)

    print("Finished Benchmark.")
    print(
        "Estimated GPU Frequency: Mean: ",
        estimated_speed,
        ", Median: ",
        median_speed,
        flush=True,
    )

    return estimated_speed


class GPUInfo:
    # approximate average on frontera RTX
    # cycles_per_second = 1919820866.3481758
    # cycles_per_second = 867404498.3008006
    # cycles_per_second = 47994628114801.04
    cycles_per_second = 1949802881.4819772

    def update_cycles(self, cycles: float = None):
        if cycles is None:
            cycles = estimate_frequency()

        self.cycles_per_second = cycles

    def get_cycles(self) -> int:
        return self.cycles_per_second


_GPUInfo = GPUInfo()


@specialize
def free_sleep(duration: float, config: RunConfig = None):
    sleep_nogil(duration)


@free_sleep.variant(architecture=gpu)
def free_sleep_gpu(duration: float, config: RunConfig = None):
    """
    Assumes all GPUs on the system are the same.
    """
    device = get_current_devices()[0]
    stream = get_current_stream()

    cycles_per_second = _GPUInfo.get_cycles()
    ticks = int(cycles_per_second * duration)
    gpu_bsleep_nogil(device.gpu_id, ticks, stream)

    if config.inner_sync:
        stream.synchronize()


@specialize
def lock_sleep(duration: float, config: RunConfig = None):
    sleep_gil(duration)


@lock_sleep.variant(architecture=gpu)
def lock_sleep_gpu(duration: float, config: RunConfig = None):
    """
    Assumes all GPUs on the system are the same.
    """
    device = get_current_devices()[0]
    stream = get_current_stream()

    cycles_per_second = _GPUInfo.get_cycles()
    ticks = int(cycles_per_second * duration)
    gpu_bsleep_gil(device.id, ticks, stream)

    if config.inner_sync:
        stream.synchronize()


def write_data_tag(block: "np.ndarray | cupy.ndarray", id: DataID):
    n, d = block.shape
    n_idx = len(id.idx)

    if d < 1:
        raise ValueError(
            "Data block dimension must be at least 1. Current dimension of {id} is {d}."
        )

    if n < n_idx + 2:
        raise ValueError(
            "Data block length must be at least 2 + the maximum id size. Current length of {id} is {n}. Expected at least {n_idx}."
        )

    # Write starting generation of the data to first index
    generation_idx = [0 for _ in range(d)]
    block[generation_idx] = 0

    # Write Data ID to the next consecutive indicies
    # ID is terminated with "-1"
    for i in range(1, n_idx + 2):
        current_idx = [0 for _ in range(d)]
        current_idx[0] = i
        if i - 1 < n_idx:
            block[generation_idx] = id.idx[i - 1]
        else:
            block[generation_idx] = -1


def generate_array(
    data_info: DataInfo, data_scale: int = 1
) -> Dict[Device, "np.ndarray | cupy.ndarray"]:
    data_id = data_info.id
    data_location = data_info.location
    data_size = data_info.size

    # Write bytes to float
    n = data_size // 4

    location_to_block = dict()

    if not isinstance(data_location, tuple):
        data_location = (data_location,)

    for location in data_location:
        assert location is not None
        if location.architecture == Architecture.CPU:
            block = np.zeros([n, data_scale], dtype=np.float32)
            write_data_tag(block, data_id)
        elif location.architecture == Architecture.GPU:
            with cupy.cuda.Device(location.device_id) as device:
                block = cupy.zeros([n, data_scale], dtype=np.float32)
                write_data_tag(block, data_id)
                device.synchronize()
        else:
            raise NotImplementedError(
                "There is no valid block type specified for Architecture: {location.architecture}"
            )
        location_to_block[location] = block

    return location_to_block


def generate_parray(
    location_to_block: Dict[Device, "np.ndarray | cupy.ndarray"]
) -> PArray:
    parray: PArray | None = None
    for i, (location, block) in enumerate(location_to_block.items()):
        if i == 0:
            parray = asarray(block)
        else:
            assert parray is not None
            # NOTE: This assumes only a single CPU that is device -1
            if location.architecture == Architecture.CPU:
                target_id = -1
            elif location.architecture == Architecture.GPU:
                target_id = location.device_id
            else:
                raise NotImplementedError(
                    "Only CPU and GPU Architectures are supported by PArray."
                )

            # Perform read at destimation
            parray._auto_move(target_id, do_write=False)
            # Assumes the above call is blocking

    assert parray is not None
    return parray


def generate_data(
    data_config: Dict[DataID, DataInfo],
    data_scale: int = 1,
    movement_type: MovementType = MovementType.NO_MOVEMENT,
) -> Dict[DataID, PArray | Dict[Device, "np.ndarray | cupy.ndarray"]]:
    data_blocks = dict()

    if movement_type == MovementType.NO_MOVEMENT:
        return data_blocks

    for data_idx, data_info in data_config.items():
        value = generate_array(data_info, data_scale)
        if movement_type == MovementType.EAGER_MOVEMENT:
            value = generate_parray(value)
        data_blocks[data_idx] = value

    return data_blocks


def get_kernel_info(
    info: TaskRuntimeInfo, config: Optional[RunConfig] = None
) -> Tuple[Tuple[float, float], int]:
    task_time = info.task_time
    gil_fraction = info.gil_fraction
    gil_accesses = info.gil_accesses

    if config is not None:
        if config.task_time is not None:
            task_time = config.task_time
        if config.gil_accesses is not None:
            gil_accesses = config.gil_accesses
        if config.gil_fraction is not None:
            gil_fraction = config.gil_fraction

    kernel_time = task_time / max(gil_accesses, 1)
    free_time = kernel_time * (1 - gil_fraction)
    gil_time = kernel_time * gil_fraction

    return (free_time, gil_time), gil_accesses


def convert_context_to_devices(context):
    device_list = []
    for device in context.devices:
        if device.architecture.name == "CPU":
            dev = Device(Architecture.CPU, 0)
        elif device.architecture.name == "GPU":
            dev = Device(Architecture.GPU, device.gpu_id)
        else:
            raise ValueError(f"Invalid architecture: {device.architecture.name}")
        device_list.append(dev)
    return tuple(device_list)


def synthetic_kernel(runtime_info: TaskPlacementInfo, config: RunConfig):
    """
    A simple synthetic kernel that simulates a task that takes a given amount of time
    and accesses the GIL a given number of times. The GIL is accessed in a fraction of
    the total time given.
    """

    if config.verbose:
        task_internal_start_t = time.perf_counter()

    context = get_current_context()
    devices = convert_context_to_devices(context)
    details = runtime_info[devices]

    if len(devices) == 0:
        raise ValueError("No devices provided to busy sleep kernel.")
    if len(devices) != len(details):
        raise ValueError(
            "Not enough TaskRuntimeInfo provided to busy sleep kernel. Must be equal to number of devices."
        )

    info = []
    for idx, device in enumerate(devices):
        if isinstance(details, TaskRuntimeInfo):
            info.append(details)
        else:
            info.append(details[idx])
        if info is None:
            raise ValueError(
                f"TaskRuntimeInfo cannot be None for {device}. Please check the runtime info passed to the task."
            )

    waste_time(info, config)

    if config.verbose:
        task_internal_end_t = time.perf_counter()
        task_internal_duration = task_internal_end_t - task_internal_start_t
        return task_internal_duration

    return None


@specialize
def waste_time(info_list: List[TaskRuntimeInfo], config: RunConfig):
    if len(info_list) == 0:
        raise ValueError("No TaskRuntimeInfo provided to busy sleep kernel.")

    info = info_list[0]

    (free_time, gil_time), gil_accesses = get_kernel_info(info, config=config)

    if gil_accesses == 0:
        free_sleep(free_time)
        return

    else:
        for i in range(gil_accesses):
            free_sleep(free_time)
            lock_sleep(gil_time)


@waste_time.variant(architecture=gpu)
def waste_time_gpu(info_list: List[TaskRuntimeInfo], config: RunConfig):
    context = get_current_context()
    if len(info_list) < len(context.devices):
        raise ValueError(
            "Not enough TaskRuntimeInfo provided to busy sleep kernel. Must be equal to number of devices."
        )

    for idx, device in enumerate(context.loop()):
        print("Device: ", device)
        info = info_list[idx]
        (free_time, gil_time), gil_accesses = get_kernel_info(info, config=config)
        if gil_accesses == 0:
            free_sleep(free_time, config=config)
        else:
            for i in range(gil_accesses):
                free_sleep(free_time, config=config)
                lock_sleep(gil_time, config=config)

    if config.outer_sync:
        context.synchronize()


def build_parla_device(mapping: Device, runtime_info: TaskRuntimeInfo):
    if mapping.architecture == Architecture.CPU:
        arch = cpu
    elif mapping.architecture == Architecture.GPU:
        arch = gpu
    elif mapping.architecture == Architecture.ANY:
        raise NotImplementedError("ANY architecture not supported for Parla devices.")
    else:
        raise ValueError(f"Invalid architecture: {mapping.architecture}")

    device_memory = runtime_info.memory
    device_fraction = runtime_info.device_fraction

    # Instatiate the Parla device object (may require scheduler to be active)
    if mapping.device_id != -1:
        device = arch(mapping.device_id)[
            {"memory": device_memory, "vcus": device_fraction}
        ]
    else:
        device = arch[{"memory": device_memory, "vcus": device_fraction}]

    return device


def build_parla_device_tuple(
    mapping: Device | Tuple[Device, ...], runtime_info: TaskPlacementInfo
):
    if isinstance(mapping, Device):
        mapping = (mapping,)

    device_constraints = runtime_info[mapping]

    if device_constraints is None:
        raise ValueError(
            f"Device constraints cannot be None for {mapping}. Please check the runtime info passed to the task."
        )
    if len(device_constraints) != len(mapping):
        raise ValueError(
            f"Device constraints must be the same length as the mapping. Please check the runtime info passed to the task."
        )

    device_list = []

    for idx, device in enumerate(mapping):
        if device is None:
            raise ValueError(
                f"Device cannot be None in mapping. Please check the runtime info passed to the task."
            )

        parla_device = build_parla_device(device, device_constraints[idx])
        device_list.append(parla_device)

    return tuple(device_list)


def build_parla_placement(
    mapping: Device | Tuple[Device, ...] | None, task_placment_info: TaskPlacementInfo
):
    if mapping is None:
        mapping_list = task_placment_info.locations
        mapping_list = [
            build_parla_device_tuple(mapping, task_placment_info)
            for mapping in mapping_list
        ]
        return mapping_list

    return [build_parla_device_tuple(mapping, task_placment_info)]


def parse_task_info(
    task: TaskInfo,
    taskspaces: Dict[str, TaskSpace],
    config: RunConfig,
    data_dict: Dict[DataID, "np.ndarray | cupy.ndarray | PArray"] | None = None,
):
    """
    Parse a tasks configuration into Parla objects to launch the task.
    """

    # Task ID
    task_idx = task.id.task_idx
    taskspace = taskspaces[task.id.taskspace]
    task_name = task.id

    # Dependency Info (List of Parla Tasks)
    dependencies = [
        taskspaces[dep.taskspace][dep.task_idx] for dep in task.dependencies
    ]

    # Valid Placement Set
    placement_info = task.runtime
    placement_list = build_parla_placement(task.mapping, placement_info)

    # Data information
    data_information = task.data_dependencies

    # Extract DataID lists
    read_data_list = data_information.read
    write_data_list = data_information.write
    rw_data_list = data_information.read_write

    if config.movement_type == MovementType.NO_MOVEMENT:
        INOUT = []
        IN = []
        OUT = []
    else:
        assert data_dict is not None

        # Remove duplicated data blocks between in/out and inout
        if len(read_data_list) > 0 and len(rw_data_list) > 0:
            read_data_list = list(set(read_data_list).difference(set(rw_data_list)))
        if len(write_data_list) > 0 and len(rw_data_list) > 0:
            write_data_list = list(set(write_data_list).difference(set(rw_data_list)))

        # Construct data blocks.
        INOUT = (
            [] if len(rw_data_list) == 0 else [data_dict[d.id] for d in rw_data_list]
        )
        IN = (
            []
            if len(read_data_list) == 0
            else [data_dict[d.id] for d in read_data_list]
        )

    return (
        task_name,
        (task_idx, taskspace, dependencies, placement_list),
        (IN, INOUT),
        placement_info,
    )


def create_task(task_name, task_info, data_info, runtime_info, config: RunConfig):
    try:
        task_idx, T, dependencies, placement_set = task_info
        IN, INOUT = data_info

        if config.verbose:
            print(
                f"Creating Task {task_name} with dependencies {dependencies} on placement {placement_set}",
                flush=True,
            )

        @spawn(
            T[task_idx],
            dependencies=dependencies,
            placement=placement_set,
            input=IN,
            inout=INOUT,
        )
        async def task_func():
            if config.verbose:
                print(f"+{task_name} Running", flush=True)

            elapsed = synthetic_kernel(runtime_info, config=config)

            if config.verbose:
                print(f"-{task_name} Finished: {elapsed} seconds", flush=True)

    except Exception as e:
        print(f"Failed creating Task {task_name}: {e}", flush=True)
    finally:
        return


def execute_tasks(
    taskspaces, tasks: Dict[TaskID, TaskInfo], run_config: RunConfig, data_list=None
):
    spawn_start_t = time.perf_counter()

    # Spawn tasks
    for task, details in tasks.items():
        task_name, task_info, data_info, runtime_info = parse_task_info(
            details, taskspaces, run_config, data_list
        )
        create_task(task_name, task_info, data_info, runtime_info, run_config)

    spawn_end_t = time.perf_counter()

    return taskspaces


def execute_graph(
    tasks: Dict[TaskID, TaskInfo],
    data_config: Dict[DataID, DataInfo],
    run_config: RunConfig,
    timing: List[TimeSample],
):
    @spawn(vcus=0, placement=cpu)
    async def main_task():
        graph_times = []

        for i in range(run_config.inner_iterations):
            data_list = generate_data(
                data_config, run_config.data_scale, run_config.movement_type
            )

            # Initialize task spaces
            taskspaces = {}

            for task, details in tasks.items():
                space_name = details.id.taskspace
                if space_name not in taskspaces:
                    taskspaces[space_name] = TaskSpace(space_name)

            graph_start_t = time.perf_counter()

            execute_tasks(taskspaces, tasks, run_config, data_list=data_list)

            for taskspace in taskspaces.values():
                await taskspace

            graph_end_t = time.perf_counter()

            graph_elapsed = graph_end_t - graph_start_t
            graph_times.append(graph_elapsed)

        graph_times = np.asarray(graph_times)
        graph_t = TimeSample(
            np.mean(graph_times),
            np.median(graph_times),
            np.std(graph_times),
            np.min(graph_times),
            np.max(graph_times),
            len(graph_times),
        )

        timing.append(graph_t)


def run(
    tasks: Dict[TaskID, TaskInfo],
    data_config: Optional[Dict[DataID, DataInfo]] = None,
    run_config: Optional[RunConfig] = None,
) -> TimeSample:
    if run_config is None:
        run_config = RunConfig(
            outer_iterations=1,
            inner_iterations=1,
            verbose=False,
            threads=1,
            data_scale=1,
        )

    if data_config is None:
        data_config = {}

    timing = []

    for outer in range(run_config.outer_iterations):
        outer_start_t = time.perf_counter()

        with Parla(logfile=run_config.logfile):
            internal_start_t = time.perf_counter()
            execute_graph(tasks, data_config, run_config, timing)
            internal_end_t = time.perf_counter()

        outer_end_t = time.perf_counter()

        parla_total_elapsed = outer_end_t - outer_start_t
        parla_internal_elapsed = internal_end_t - internal_start_t

    return timing[0]


def verify_order(
    log_times: Dict[TaskID, TaskTime], truth_graph: Dict[TaskID, List[TaskID]]
) -> bool:
    """
    Verify that all tasks have run in the correct order in the log graph.
    """

    for task in truth_graph:
        details = truth_graph[task]

        for dependency in details.dependencies:
            if log_times[task].start_t < log_times[dependency].end_t:
                print("Task {task} started before dependency {dependency}")
                return False

    return True


def verify_dependencies(
    log_graph: Dict[TaskID, List[TaskID]], truth_graph: Dict[TaskID, List[TaskID]]
):
    """
    Verify that all dependencies in the truth graph have completed execution in the log graph.
    """

    for task in truth_graph:
        details = truth_graph[task]

        for dependency in details.dependencies:
            if dependency not in log_graph:
                print(f"Dependency {dependency} of task {task} not in log graph")
                return False

    return True


def verify_complete(
    log_graph: Dict[TaskID, List[TaskID]], truth_graph: Dict[TaskID, List[TaskID]]
) -> bool:
    """
    Verify that all tasks in the truth graph have completed exceution in the log graph.
    """

    for task in truth_graph:
        if task not in log_graph:
            print(f"Task {task} not in log graph")
            return False

    return True


def verify_time(
    log_times: Dict[TaskID, TaskTime],
    truth_graph: Dict[TaskID, List[TaskID]],
    factor: float = 2.0,
) -> bool:
    """
    Verify that all tasks execute near their expected time.
    """

    for task in truth_graph:
        details = truth_graph[task]

        # TODO: This needs to be fixed for device support
        device_idx = (-1,)  # CPU
        expected_time = details.runtime[device_idx].task_time
        observed_time = log_times[task].duration / 1000

        if observed_time > expected_time * factor:
            print(
                f"Task {task} took too long to execute. Expected {expected_time} us, took {observed_time} us"
            )
            return False

    return True


def verify_ntasks(
    log_times: Dict[TaskID, TaskTime], truth_graph: Dict[TaskID, List[TaskID]]
):
    """
    Verify that the number of tasks in the log graph is the same as the number of tasks in the truth graph.
    """

    if len(log_times) != len(truth_graph) + 1:
        print(
            f"Number of tasks in log graph ({len(log_times)}) does not match number of tasks in truth graph ({len(truth_graph)})"
        )
        return False

    return True


def verify_states(log_states) -> bool:
    """
    Verify that all tasks have visited all states in a valid order.
    """

    for task in log_states:
        states = log_states[task]
        instance = task.instance

        # if ('SPAWNED' not in states):
        #    print(f"Task {task} did not spawn", flush=True)
        #    return False
        if "MAPPED" not in states:
            print(f"Task {task} was not mapped.", states, flush=True)
            return False
        if "RESERVED" not in states:
            print(f"Task {task} was not reserved.", states, flush=True)
            return False
        # if ('RUNNING' not in states):
        #    print(f"Task {task} did not run.", states, flush=True)
        #    return False
        if "RUNAHEAD" not in states:
            print(f"Task {task} was not runahead", states, flush=True)
            return False

    return True


def timeout(seconds_before_timeout):
    """
    Decorator that raises an exception if the function takes longer than seconds_before_timeout to execute.
    https://stackoverflow.com/questions/21827874/timeout-a-function-windows
    """

    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [
                Exception(
                    "function [%s] timeout [%s seconds] exceeded!"
                    % (func.__name__, seconds_before_timeout)
                )
            ]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Propagate(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
                r = t.value
            except Exception as e:
                print("Unhandled exception in Propagate wrapper", flush=True)
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


class GraphContext(object):
    def __init__(self, config: GraphConfig, name: str, graph_path=None):
        self.config = config
        self.graph = None
        self.data_config = None

        self.name = name
        self.graph_function = None

        if isinstance(config, SerialConfig):
            self.graph_function = generate_serial_graph
        elif isinstance(config, IndependentConfig):
            self.graph_function = generate_independent_graph
        elif isinstance(config, ReductionConfig):
            self.graph_function = generate_reduction_graph
        elif isinstance(config, ReductionScatterConfig):
            self.graph_function = generate_reduction_scatter_graph

        if graph_path is not None:
            self.tmpfilepath = graph_path
        else:
            self.tmpfilepath = None

    def __enter__(self):
        self.diro = tempfile.TemporaryDirectory()
        self.dir = self.diro.__enter__()

        if self.tmpfilepath is None:
            self.tmpfilepath = os.path.join(
                self.dir, "test_" + str(self.name) + ".graph"
            )
        self.tmplogpath = os.path.join(self.dir, "test_" + str(self.name) + "_.blog")

        print("Graph Path:", self.tmpfilepath)
        with open(self.tmpfilepath, "w") as tmpfile:
            graph = self.graph_function(self.config)
            # print(graph)
            tmpfile.write(graph)

        self.data_config, self.graph = read_pgraph(self.tmpfilepath)

        return self

    def run(self, run_config: RunConfig, max_time: int = 100):
        @timeout(max_time)
        def run_with_timeout():
            return run(self.graph, self.data_config, run_config)

        return run_with_timeout()

    def __exit__(self, type, value, traceback):
        self.diro.__exit__(type, value, traceback)


__all__ = [
    run,
    verify_order,
    verify_dependencies,
    verify_complete,
    verify_time,
    timeout,
    GraphContext,
]
