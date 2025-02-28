from dataclasses import dataclass
from typing import Optional, Type, Self
from .types import (
    DeviceTuple,
    TaskTuple,
    DataBlockTuple,
    VariantTuple,
    ConnectionTuple,
    _bytes_to_readable,
)
from .lambdas import VariantBuilder, TaskLabeler, DataBlockTransformer

import numpy as np
import task4feedback.fastsim2 as fastsim
from task4feedback.fastsim2 import (
    Devices,
    Topology,
    Tasks,
    TaskNoise,
    CommunicationNoise,
    Data,
    GraphTemplate,
    DeviceType,
    SchedulerInput,
    RangeTransitionConditions,
    DefaultTransitionConditions,
    SchedulerState,
    Simulator,
)
from task4feedback.fastsim2 import ExecutionState, start_logger


class Graph:
    def __init__(self):
        self.graph = GraphTemplate()
        self.ctasks = None

    def add_task(self, name, tag):
        self.graph.add_task(name, tag)

    def get_task(self, task, convert=False):
        if convert and isinstance(task, str):
            task = self.graph.get_id(task)

        id = task
        name = self.graph.get_name(id)
        tag = self.graph.get_tag(id)
        dependencies = self.graph.get_dependencies(id)
        read = self.graph.get_read_data(id)
        write = self.graph.get_write_data(id)
        type = self.graph.get_type(id)

        return TaskTuple(id, name, tag, dependencies, read, write, type)

    def convert_list_to_ids(self, tasklist):
        return [
            self.graph.get_id(task) if isinstance(task, str) else task
            for task in tasklist
        ]

    def convert_ids_to_names(self, tasklist):
        return [
            self.graph.get_name(task) if isinstance(task, int) else task
            for task in tasklist
        ]

    def get_id(self, name):
        return self.graph.get_id(name)

    def add_dependencies(self, task, dependencies, convert=False):
        if convert and isinstance(task, str):
            task = self.graph.get_id(task)
        self.graph.add_dependencies(task, dependencies)

    def add_dependency(self, task, dependency, convert=False):
        if convert and isinstance(task, str):
            task = self.graph.get_id(task)
        if convert and isinstance(dependency, str):
            dependency = self.graph.get_id(dependency)
        self.graph.add_dependency(task, dependency)

    def add_read_data(self, task, dataidlist, convert=False):
        if convert and isinstance(task, str):
            task = self.graph.get_id(task)
        self.graph.add_read_data(task, dataidlist)

    def add_write_data(self, task, dataidlist, convert=False):
        if convert and isinstance(task, str):
            task = self.graph.get_id(task)
        self.graph.add_write_data(task, dataidlist)

    def apply_variant(self, variant_builder: VariantBuilder):
        for i in range(self.graph.size()):
            task = self.get_task(i)
            for arch in DeviceType:
                if arch == DeviceType.NONE:
                    continue

                variant = variant_builder.build_variant(arch, task)

                if variant is None:
                    continue
                self.graph.add_variant(
                    i,
                    arch,
                    variant.memory_usage,
                    variant.vcu_usage,
                    variant.expected_time,
                )

    def apply_tag(self, tag_builder: TaskLabeler):
        for i in range(self.graph.size()):
            task = self.get_task(i)
            tag = tag_builder.label(task)
            self.graph.set_tag(i, tag)

    def apply_type(self, type_builder: TaskLabeler):
        for i in range(self.graph.size()):
            task = self.get_task(i)
            type = type_builder.label(task)
            self.graph.set_type(i, type)

    def finalize_tasks(
        self, create_data_tasks=True, add_missing_dependencies=True, verbose=False
    ):
        if verbose:
            print(f"..finalizing GraphWrapper with {self.graph.size()} tasks.")
        self.ctasks = self.graph.to_tasks()
        fastsim.GraphManager.finalize(
            self.ctasks, create_data_tasks, add_missing_dependencies
        )
        if verbose:
            print(f"..created {self.ctasks.data_size()} data tasks.")

    def get_c_tasks(self):
        assert self.ctasks is not None
        return self.ctasks

    @staticmethod
    def create_from_legacy_graph(graph, datamap):
        """
        Convert the older (2023-2024) python graph format to the new c++ stored graph format.
        """
        from task4feedback.legacy_types import Device, Architecture

        ids_to_tasks = {}
        tasks_to_ids = {}

        for i, task_id in enumerate(graph):
            ids_to_tasks[i] = task_id
            tasks_to_ids[task_id] = i

        ids_to_data = {}
        data_to_ids = {}

        for i, data_id in enumerate(datamap):
            ids_to_data[i] = data_id
            data_to_ids[data_id] = i

        g = Graph()
        for i, task_id in enumerate(graph):
            task = graph[task_id]
            dependencies = [tasks_to_ids[dep] for dep in task.dependencies]
            read = [d.id for d in task.data_dependencies.read]
            write = [d.id for d in task.data_dependencies.write]
            read_write = [d.id for d in task.data_dependencies.read_write]

            read_set = set(read).union(set(read_write))
            write_set = set(write).union(set(read_write))

            read_ids = [data_to_ids[d] for d in read_set]
            write_ids = [data_to_ids[d] for d in write_set]

            name = str(task_id)

            g.add_task(name, 0)
            g.add_dependencies(i, dependencies)
            g.add_read_data(i, read_ids)
            g.add_write_data(i, write_ids)

            any_cpu_flag = Device(Architecture.CPU, -1) in task.runtime
            cpu_0_flag = Device(Architecture.CPU, 0) in task.runtime

            if any_cpu_flag:
                placement_info = task.runtime[Device(Architecture.CPU, -1)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                g.graph.add_variant(i, DeviceType.CPU, 0, vcu, placement_info.task_time)
            elif cpu_0_flag:
                placement_info = task.runtime[Device(Architecture.CPU, 0)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                g.graph.add_variant(i, DeviceType.CPU, 0, vcu, placement_info.task_time)

            any_gpu_flag = Device(Architecture.GPU, -1) in task.runtime
            gpu_0_flag = Device(Architecture.GPU, 0) in task.runtime

            if any_gpu_flag:
                placement_info = task.runtime[Device(Architecture.GPU, -1)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                g.graph.add_variant(i, DeviceType.GPU, 0, vcu, placement_info.task_time)
            elif gpu_0_flag:
                placement_info = task.runtime[Device(Architecture.GPU, 0)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                g.graph.add_variant(i, DeviceType.GPU, 0, vcu, placement_info.task_time)

        return g

    def __str__(self):
        result = []

        task_count = self.graph.size()
        result.append(f"GraphWrapper with {task_count} tasks:")

        for i in range(task_count):
            task = self.get_task(i)

            if task.dependencies:
                dep_names = []
                for dep_id in task.dependencies:
                    dep_name = self.graph.get_name(dep_id)
                    dep_names.append(f"{dep_name}({dep_id})")
                deps_str = ", ".join(dep_names)
            else:
                deps_str = "None"

            read_str = ", ".join(map(str, task.read)) if task.read else "None"
            write_str = ", ".join(map(str, task.write)) if task.write else "None"

            result.append(
                f"  Task {task.id}: {task.name} (Tag: {task.tag}, Type: {task.type})"
            )
            result.append(f"    Dependencies: {deps_str}")
            result.append(f"    Reads: {read_str}")
            result.append(f"    Writes: {write_str}")

        return "\n".join(result)


class DataBlocks:
    def __init__(self, initial_size=0):
        if initial_size > 0:
            self.data = Data(initial_size)
        else:
            self.data = Data()

    def add_block(self, name, size, location=0, id=None):
        if id is None:
            id = self.data.append_block(size, location, name)
        else:
            self.data.create_block(id, size, location, name)
        return DataBlockTuple(id, name, size, location)

    def set_location(self, block, location, convert=False):
        if convert and isinstance(block, str):
            block = self.data.get_id(block)
        self.data.set_location(block, location)

    def get_block(self, block, convert=False):
        if convert and isinstance(block, str):
            block = self.data.get_id(block)
        id = block
        name = self.data.get_name(id)
        size = self.data.get_size(id)
        location = self.data.get_location(id)
        tag = self.data.get_tag(id)
        block_type = self.data.get_type(id)
        return DataBlockTuple(id, name, size, location, tag, block_type)

    def get_id(self, name):
        return self.data.get_id(name)

    def convert_list_to_ids(self, blocklist):
        return [
            self.data.get_id(block) if isinstance(block, str) else block
            for block in blocklist
        ]

    def convert_ids_to_names(self, blocklist):
        return [
            self.data.get_name(block) if isinstance(block, int) else block
            for block in blocklist
        ]

    def apply(self, transformer):
        for i in range(self.data.size()):
            block = self.get_block(i)
            if block is None:
                continue
            transformed_block = transformer.transform(block)
            self.data.set_name(i, transformed_block.name)
            self.data.set_size(i, transformed_block.size)
            self.data.set_location(i, transformed_block.location)
            self.data.set_tag(i, transformed_block.tag)
            self.data.set_type(i, transformed_block.type)

    @staticmethod
    def create_from_legacy_data(data, system):
        from task4feedback.legacy_types import Device

        ids_to_data = {}
        data_to_ids = {}

        for i, data_id in enumerate(data):
            ids_to_data[i] = data_id
            data_to_ids[data_id] = i

        d = DataBlocks()
        for i, data_id in enumerate(data):
            name = str(data_id)
            size = data[data_id].size
            location = data[data_id].location

            if not isinstance(location, Device):
                location = location[0]

            location_id = system.get_global_id_from_legacy(location)

            d.add_block(name, size, location_id)

        return d

    def __str__(self):
        result = []

        block_count = self.data.size()
        result.append(f"DataWrapper with {block_count} blocks:")

        for i in range(block_count):
            block = self.get_block(i)
            result.append(
                f"  Block {block.id}: {block.name} (Size: {_bytes_to_readable(block.size)}, Location: {block.location}, Tag: {block.tag}, Type: {block.type})"
            )

        return "\n".join(result)


class System:
    def __init__(self):
        self.devices = Devices()
        self.topology = None

    def create_device(self, name, arch, memory, vcu, id=None):
        MAX_VCUS = fastsim.MAX_VCUS
        vcu = int(vcu * MAX_VCUS)
        if id is None:
            id = self.devices.append_device(name, arch, vcu, memory)
        else:
            self.devices.create_device(id, name, arch, vcu, memory)

        return DeviceTuple(name, id, self.devices.get_local_id(id), arch, memory, vcu)

    def finalize_devices(self):
        self.topology = Topology(self.devices.size())

    @staticmethod
    def convert_legacy_architecture(arch):
        from task4feedback.legacy_types import Architecture

        if arch == Architecture.CPU:
            return DeviceType.CPU
        if arch == Architecture.GPU:
            return DeviceType.GPU
        return DeviceType.NONE

    def get_global_id_from_legacy(self, device):
        from task4feedback.legacy_types import Device

        assert isinstance(device, Device)
        device_type = System.convert_legacy_architecture(device.architecture)
        return self.get_global_id(device_type, device.device_id)

    def get_global_id(self, architecture: DeviceType, local_id: int):
        return self.devices.get_global_id(architecture, local_id)

    def get_local_id(self, global_id: int):
        return self.devices.get_local_id(global_id)

    def get_type(self, global_id: int):
        return self.devices.get_type(global_id)

    def get_device(self, global_id: int):
        local_id = self.get_local_id(global_id)
        name = self.devices.get_name(global_id)
        arch = self.devices.get_type(global_id)
        dev = self.devices.get_device(global_id)
        vcu = dev.get_vcu()
        memory = dev.get_mem()
        return DeviceTuple(name, global_id, local_id, arch, memory, vcu)

    def add_connection(self, s_gid, d_gid, bandwidth, latency, max_connections=2):
        if self.topology is None:
            self.finalize_devices()
            raise Warning(
                "Devices must be finalized before adding connections. Calling finalize_devices() first."
            )

        self.topology.set_bandwidth(s_gid, d_gid, bandwidth)
        self.topology.set_latency(s_gid, d_gid, latency)
        self.topology.set_max_connections(s_gid, d_gid, max_connections)

    def __str__(self):
        result = []

        device_count = self.devices.size()
        result.append(f"SystemWrapper with {device_count} devices:")

        for i in range(device_count):
            device = self.get_device(i)
            result.append(
                f"  Device {device.global_id}: {device.name} (Type: {device.arch}, Memory: {_bytes_to_readable(device.memory)}, VCU: {device.vcu})"
            )

        return "\n".join(result)

    def connection_table(self):
        if self.topology is None:
            return "No topology defined. Call finalize_devices() first."

        result = []
        result.append("-" * 80)
        result.append(f"{'Source':<20} {'Dest':<20} {'BW':<15} {'L':<15} {'C':<15}")
        result.append("-" * 80)

        device_count = self.devices.size()

        for src in range(device_count):
            src_device = self.get_device(src)
            for dst in range(device_count):
                if src == dst:
                    continue

                if self.topology.get_bandwidth(src, dst) > 0:
                    dst_device = self.get_device(dst)
                    bandwidth = self.topology.get_bandwidth(src, dst)
                    latency = self.topology.get_latency(src, dst)
                    max_conn = self.topology.get_max_connections(src, dst)

                    bandwidth_str = (
                        _bytes_to_readable(bandwidth) + "/s" if bandwidth > 0 else "N/A"
                    )

                    result.append(
                        f"{src_device.name} (ID: {src})".ljust(20)
                        + f"{dst_device.name} (ID: {dst})".ljust(20)
                        + f"{bandwidth_str}".ljust(15)
                        + f"{latency} ms".ljust(15)
                        + f"{max_conn}".ljust(15)
                    )

        if len(result) == 3:  # Only header rows present
            result.append("No connections found between devices.")

        return "\n".join(result)


@dataclass
class NoiseConfig:
    task_noise: TaskNoise
    comm_noise: CommunicationNoise

    def __init__(
        self,
        graph: Graph,
        system: System,
        duration_seed: int = 0,
        priority_seed: int = 0,
        comm_seed: int = 0,
    ):
        self.task_noise = TaskNoise(graph.get_c_tasks(), duration_seed, priority_seed)
        self.comm_noise = CommunicationNoise(system.topology, comm_seed)


class ExternalMapper:
    def __init__(self, mapper: Optional[Self] = None):
        pass

    def map_tasks(
        self, candidate_tasks: list, simulator: "SimulatorDriver"
    ) -> list[fastsim.Action]:
        # print(candidate_tasks)
        global_task_id = candidate_tasks[0]
        local_id = 0
        device = 0
        state = simulator.simulator.get_state()
        mapping_priority = state.get_mapping_priority(global_task_id)
        return [
            fastsim.Action(
                candidate_tasks[0], local_id, device, mapping_priority, mapping_priority
            )
        ]


@dataclass
class SimulatorInput:
    graph: Graph
    data: DataBlocks
    system: System
    noise: NoiseConfig
    transition_conditions: fastsim.TransitionConditions

    def __init__(
        self,
        graph: Graph,
        data: DataBlocks,
        system: System,
        noise: Optional[NoiseConfig] = None,
        transition_conditions: Optional[fastsim.TransitionConditions] = None,
    ):
        if transition_conditions is None:
            transition_conditions = fastsim.RangeTransitionConditions(5, 5, 8)
        if noise is None:
            noise = NoiseConfig(graph, system)
        self.noise = noise
        self.graph = graph
        self.data = data
        self.system = system
        self.transition_conditions = transition_conditions

    def to_input(self):
        return SchedulerInput(
            self.graph.get_c_tasks(),
            self.data.data,
            self.system.devices,
            self.system.topology,
            self.noise.task_noise,
            self.noise.comm_noise,
            self.transition_conditions,
        )


@dataclass
class ExternalObserver:
    task_feature_extractor: fastsim.RuntimeFeatureExtractor
    data_feature_extractor: fastsim.RuntimeFeatureExtractor
    device_feature_extractor: fastsim.RuntimeFeatureExtractor
    task_task_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
    task_data_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
    task_device_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
    data_device_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
    graph_extractor: fastsim.GraphExtractor

    @dataclass
    class ExternalObserver:
        task_feature_extractor: fastsim.RuntimeFeatureExtractor
        data_feature_extractor: fastsim.RuntimeFeatureExtractor
        device_feature_extractor: fastsim.RuntimeFeatureExtractor
        task_task_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
        task_data_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
        task_device_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
        data_device_feature_extractor: fastsim.RuntimeEdgeFeatureExtractor
        graph_extractor: fastsim.GraphExtractor

        def __init__(
            self,
            state: fastsim.SchedulerState,
            task_feature_types: Optional[list],
            data_feature_types: Optional[list],
            device_feature_types: Optional[list],
            task_task_feature_types: Optional[list],
        ):
            self.task_feature_extractor = fastsim.RuntimeFeatureExtractor(
                state, task_feature_types
            )
            self.data_feature_extractor = fastsim.RuntimeFeatureExtractor(
                state, data_feature_types
            )
            self.device_feature_extractor = fastsim.RuntimeFeatureExtractor(
                state, device_feature_types
            )
            self.task_task_feature_extractor = fastsim.RuntimeEdgeFeatureExtractor(
                state, task_task_feature_types
            )
            self.task_data_feature_extractor = fastsim.RuntimeEdgeFeatureExtractor(
                state, task_task_feature_types
            )
            self.task_device_feature_extractor = fastsim.RuntimeEdgeFeatureExtractor(
                state, task_task_feature_types
            )
            self.data_device_feature_extractor = fastsim.RuntimeEdgeFeatureExtractor(
                state, task_task_feature_types
            )
            self.graph_extractor = fastsim.GraphExtractor(state)


@dataclass
class SimulatorDriver:
    input: SimulatorInput
    internal_mapper: fastsim.Mapper
    external_mapper: ExternalMapper
    simulator: fastsim.Simulator
    observer: ExternalObserver

    def __init__(
        self,
        input: SimulatorInput,
        internal_mapper: fastsim.Mapper
        | Type[fastsim.Mapper] = fastsim.DequeueEFTMapper,
        external_mapper: ExternalMapper | Type[ExternalMapper] = ExternalMapper,
        observer: Type[ExternalObserver] = ExternalObserver,
        simulator: Optional[fastsim.Simulator] = None,
    ):
        """
        Initializes the wrapper with the provided input, mappers, observer, and simulator.

        Args:
            input (SimulatorInput): The input data for the simulator.
            internal_mapper (fastsim.Mapper | Type[fastsim.Mapper], optional): The internal mapper instance or class. Defaults to fastsim.DequeueEFTMapper.
            external_mapper (ExternalMapper | Type[ExternalMapper], optional): The external mapper instance or class. Defaults to ExternalMapper.
            observer (Type[ExternalObserver], optional): The observer class. Defaults to ExternalObserver.
            simulator (Optional[fastsim.Simulator], optional): An optional simulator instance. If not provided, a new simulator will be created using the input and internal mapper.

        Attributes:
            input (SimulatorInput): The input data for the simulator.
            internal_mapper (fastsim.Mapper): The internal mapper instance.
            external_mapper (ExternalMapper): The external mapper instance.
            observer (ExternalObserver): The observer instance.
            simulator (fastsim.Simulator): The simulator instance.
        """
        self.input = input
        if isinstance(internal_mapper, type):
            internal_mapper = internal_mapper()

        if isinstance(external_mapper, type):
            external_mapper = external_mapper()

        self.internal_mapper = internal_mapper
        self.external_mapper = external_mapper

        self.observer = observer()

        if simulator is None:
            self.simulator = fastsim.Simulator(input.to_input(), self.internal_mapper)
        else:
            self.simulator = simulator
            self.simulator.set_mapper(self.internal_mapper)

    def initialize(self):
        """
        Initialize the simulator (creates workspaces for current tasks, state, etc).
        The GRAPH input SHOULD NOT be modified after this is called.
        The NOISE input SHOULD NOT be modified after this is called.
        THE SYSTEM input SHOULD NOT be modified after this is called.
        """
        self.simulator.initialize()

    def initialize_data(self):
        """
        Initialize the simulator data manager.
        This finalizes the starting locations of all data blocks and their initial memory usage.
        The DATA input SHOULD NOT be modified after this is called.
        """
        self.simulator.initialize_data()

    def enable_external_mapper(self):
        """
        Use external mapper for mapping tasks (run Python callback).
        """
        self.simulator.enable_python_mapper()

    def disable_external_mapper(self):
        """
        Use internal mapper for mapping tasks (do not run Python callback).
        """
        self.simulator.disable_python_mapper()

    def fresh_copy(self) -> "SimulatorDriver":
        """
        Initialize a fresh (uninitialized) copy of the simulator driver with the same initial input and configuration.
        """
        internal_mapper_t = type(self.internal_mapper)
        external_mapper_t = type(self.external_mapper)
        observer_t = type(self.observer)

        internal_mapper_copy = internal_mapper_t()
        external_mapper_copy = external_mapper_t()

        return SimulatorDriver(
            self.input, internal_mapper_copy, external_mapper_copy, observer_t
        )

    def time(self) -> int:
        """
        Returns the current time (in microseconds) of the simulator state.
        """
        return self.simulator.get_current_time()

    def copy(self) -> "SimulatorDriver":
        """
        Initialize a copy of the simulator driver at the current state (may be initialized if the source simulator is).
        Mappers and their internal state (if any) are copied as well.
        """
        internal_mapper_t = type(self.internal_mapper)
        external_mapper_t = type(self.external_mapper)
        observer_t = type(self.observer)

        internal_mapper_copy = internal_mapper_t(self.internal_mapper)
        external_mapper_copy = external_mapper_t(self.external_mapper)

        simulator_copy = fastsim.Simulator(self.simulator)
        return SimulatorDriver(
            self.input,
            internal_mapper_copy,
            external_mapper_copy,
            observer_t,
            simulator_copy,
        )

    def run(self) -> ExecutionState:
        """
        Run the simulator until a breakpoint, error, or completion is reached.
        This function will return the current state of the simulator at the exitpoint.
        """
        sim_state = ExecutionState.RUNNING
        while sim_state == ExecutionState.RUNNING:
            sim_state = self.simulator.run()

            if sim_state == ExecutionState.BREAKPOINT:
                return sim_state

            if sim_state == ExecutionState.ERROR:
                return sim_state

            if sim_state == ExecutionState.EXTERNAL_MAPPING:
                print("External Mapping")
                candidates = self.simulator.get_mappable_candidates()
                actions = self.external_mapper.map_tasks(candidates, self)
                self.simulator.map_tasks(actions)
                sim_state = ExecutionState.RUNNING
        return sim_state


def uniform_connected_devices(n_devices: int, mem: int, latency: int, bandwidth: int):
    """
    Creates a system with a uniform connection of devices including one CPU and multiple GPUs.
    Parameters:
    n_devices (int): Total number of devices including one CPU and multiple GPUs. Must be greater than 1.
    mem (int): Memory allocated to each device.
    latency (int): Latency of the connections between devices.
    bandwidth (int): Bandwidth of the connections between devices.
    Returns:
    System: A system object with the specified devices and connections.
    Raises:
    AssertionError: If n_devices is not greater than 1.
    """
    assert n_devices > 1

    s = System()
    n_gpus = n_devices - 1

    s.create_device("CPU:0", DeviceType.CPU, mem, 1)
    for i in range(n_gpus):
        s.create_device(f"GPU:{i}", DeviceType.GPU, mem, 1)

    s.finalize_devices()

    for i in range(n_gpus):
        s.add_connection(0, i + 1, bandwidth, latency)
        s.add_connection(i + 1, 0, bandwidth, latency)

    for i in range(n_gpus):
        for j in range(n_gpus):
            s.add_connection(i + 1, j + 1, bandwidth, latency)
            s.add_connection(j + 1, i + 1, bandwidth, latency)

    return s
