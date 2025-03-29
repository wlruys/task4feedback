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
from rich import print
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
import torch
from tensordict.tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch


class Graph:
    def __init__(self):
        self.graph = GraphTemplate()
        self.ctasks = None

    def add_task(self, name, tag):
        return self.graph.add_task(name, tag)

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

    def __len__(self):
        return self.graph.size()

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

    def get_name(self, task):
        return self.graph.get_name(task)

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

    def apply_variant(self, variant_builder: type[VariantBuilder]):
        for i in range(self.graph.size()):
            task = self.get_task(i)
            for arch in DeviceType:
                if arch == DeviceType.NONE:
                    continue

                variant = variant_builder.build_variant(arch, task)

                if variant is None:
                    continue

                vcu_usage = int(variant.vcu_usage * fastsim.MAX_VCUS)
                self.graph.add_variant(
                    i,
                    arch,
                    vcu_usage,
                    variant.memory_usage,
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
        self, create_data_tasks=True, add_missing_dependencies=False, verbose=False
    ):
        if verbose:
            print(f"..finalizing GraphWrapper with {self.graph.size()} tasks.")
        self.ctasks = self.graph.to_tasks()
        fastsim.GraphManager.finalize(
            self.ctasks, create_data_tasks, add_missing_dependencies
        )
        if verbose:
            print(f"..created {self.ctasks.data_size()} data tasks.")

    def fill_data_flow_dependencies(self):
        self.graph.fill_dependencies_from_data_usage()

    def get_c_tasks(self):
        assert self.ctasks is not None
        return self.ctasks

    def __iter__(self):
        for i in range(self.graph.size()):
            yield self.get_task(i)

    def to_networkx(self):
        import networkx as nx

        G = nx.DiGraph()

        for task in self:
            G.add_node(task.id, label=task.name)

            for dep_id in task.dependencies:
                G.add_edge(dep_id, task.id)

        return G

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
                mem = placement_info.memory
                g.graph.add_variant(
                    i, DeviceType.CPU, vcu, mem, placement_info.task_time
                )
            elif cpu_0_flag:
                placement_info = task.runtime[Device(Architecture.CPU, 0)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                mem = placement_info.memory
                g.graph.add_variant(
                    i, DeviceType.CPU, vcu, mem, placement_info.task_time
                )

            any_gpu_flag = Device(Architecture.GPU, -1) in task.runtime
            gpu_0_flag = Device(Architecture.GPU, 0) in task.runtime

            if any_gpu_flag:
                placement_info = task.runtime[Device(Architecture.GPU, -1)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                mem = placement_info.memory
                g.graph.add_variant(
                    i, DeviceType.GPU, vcu, mem, placement_info.task_time
                )
            elif gpu_0_flag:
                placement_info = task.runtime[Device(Architecture.GPU, 0)][0]
                vcu = int(placement_info.device_fraction * fastsim.MAX_VCUS)
                mem = placement_info.memory
                g.graph.add_variant(
                    i, DeviceType.GPU, vcu, mem, placement_info.task_time
                )

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

    def get_location(self, block):
        if isinstance(block, str):
            block = self.data.get_id(block)
        return self.data.get_location(block)

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

    def apply(self, transformer: DataBlockTransformer):
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

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        # print(candidate_tasks)
        candidates = torch.zeros((1), dtype=torch.int64)
        simulator.simulator.get_mappable_candidates(candidates)
        global_task_id = candidates[0]
        local_id = 0
        device = 0
        state = simulator.simulator.get_state()
        mapping_priority = state.get_mapping_priority(global_task_id)
        return [fastsim.Action(local_id, device, mapping_priority, mapping_priority)]


class StaticExternalMapper:
    def __init__(
        self, mapper: Optional[Self] = None, mapping_dict: Optional[dict] = None
    ):
        if mapper is not None:
            self.mapping_dict = mapper.mapping_dict

        elif mapping_dict is not None:
            self.mapping_dict = mapping_dict
        else:
            self.mapping_dict = {}

    def set_mapping_dict(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        candidates = torch.zeros((1), dtype=torch.int64)
        simulator.simulator.get_mappable_candidates(candidates)
        global_task_id = candidates[0].item()
        local_id = 0
        device = self.mapping_dict[global_task_id]
        state = simulator.simulator.get_state()
        mapping_priority = state.get_mapping_priority(global_task_id)
        return [fastsim.Action(local_id, device, mapping_priority, mapping_priority)]


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
            # transition_conditions = fastsim.RangeTransitionConditions(5, 5, 16)
            transition_conditions = DefaultTransitionConditions()
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


class FeatureExtractorFactory:
    def __init__(
        self,
        feature_list: Optional[list] = None,
        options: Optional[dict[Type, tuple]] = None,
    ):
        if feature_list is None:
            feature_list = []
        if options is None:
            options = {}

        self.options = options
        self.feature_list = feature_list

    def create(self, state: fastsim.SchedulerState):
        feature_extractor = fastsim.RuntimeFeatureExtractor()
        for feature_t in self.feature_list:
            args = self.options.get(feature_t, tuple())
            feature_extractor.add_feature(feature_t.create(state, *args))
        return feature_extractor

    def add(self, feature_t: Type, *args):
        self.feature_list.append(feature_t)

        if args:
            self.options[feature_t] = args


class EdgeFeatureExtractorFactory:
    def __init__(
        self,
        feature_list: Optional[list] = None,
        options: Optional[dict[Type, tuple]] = None,
    ):
        if feature_list is None:
            feature_list = []
        if options is None:
            options = {}

        self.options = options
        self.feature_list = feature_list

    def create(self, state: fastsim.SchedulerState):
        feature_extractor = fastsim.RuntimeEdgeFeatureExtractor()
        for feature_t in self.feature_list:
            args = self.options.get(feature_t, tuple())
            feature_extractor.add_feature(feature_t.create(state, *args))
        return feature_extractor

    def add(self, feature_t: Type, *args):
        self.feature_list.append(feature_t)

        if args:
            self.options[feature_t] = args


@dataclass
class ExternalObserverFactory:
    graph_spec: fastsim.GraphSpec
    graph_extractor_t: Type[fastsim.GraphExtractor]
    task_feature_factory: FeatureExtractorFactory
    data_feature_factory: FeatureExtractorFactory
    device_feature_factory: FeatureExtractorFactory
    task_task_feature_factory: EdgeFeatureExtractorFactory
    task_data_feature_factory: EdgeFeatureExtractorFactory
    task_device_feature_factory: Optional[EdgeFeatureExtractorFactory]
    data_device_feature_factory: Optional[EdgeFeatureExtractorFactory]

    def create(self, simulator: Simulator):
        state = simulator.get_state()
        graph_spec = self.graph_spec
        graph_extractor = self.graph_extractor_t(state)
        task_feature_extractor = self.task_feature_factory.create(state)
        data_feature_extractor = self.data_feature_factory.create(state)
        device_feature_extractor = self.device_feature_factory.create(state)
        task_task_feature_extractor = self.task_task_feature_factory.create(state)
        task_data_feature_extractor = self.task_data_feature_factory.create(state)
        task_device_feature_extractor = (
            self.task_device_feature_factory.create(state)
            if self.task_device_feature_factory is not None
            else None
        )
        data_device_feature_extractor = (
            self.data_device_feature_factory.create(state)
            if self.data_device_feature_factory is not None
            else None
        )

        return ExternalObserver(
            simulator,
            graph_spec,
            graph_extractor,
            task_feature_extractor,
            data_feature_extractor,
            device_feature_extractor,
            task_task_feature_extractor,
            task_data_feature_extractor,
            task_device_feature_extractor,
            data_device_feature_extractor,
        )


class CompiledDefaultObserverFactory:
    def __init__(self, spec: fastsim.GraphSpec):
        self.spec = spec
        self.graph_extractor_t = fastsim.GraphExtractor
        self.task_feature_factory = fastsim.TaskFeatureExtractor
        self.data_feature_factory = fastsim.DataFeatureExtractor
        self.device_feature_factory = fastsim.DeviceFeatureExtractor
        self.task_task_feature_factory = fastsim.TaskTaskFeatureExtractor
        self.task_data_feature_factory = fastsim.TaskDataFeatureExtractor
        self.task_device_feature_factory = None
        self.data_device_feature_factory = None

    def create(self, simulator: Simulator):
        state = simulator.get_state()
        graph_spec = self.spec
        graph_extractor = self.graph_extractor_t(state)
        task_feature_extractor = self.task_feature_factory(
            fastsim.InDegreeTaskFeature(state),
            fastsim.OutDegreeTaskFeature(state),
            fastsim.OneHotMappedDeviceTaskFeature(state),
        )
        data_feature_extractor = self.data_feature_factory(
            fastsim.DataSizeFeature(state),
            fastsim.DataMappedLocationsFeature(state),
        )

        device_feature_extractor = self.device_feature_factory(
            fastsim.DeviceArchitectureFeature(state),
            fastsim.DeviceIDFeature(state),
            fastsim.DeviceMemoryFeature(state),
            fastsim.DeviceTimeFeature(state),
        )

        task_task_feature_extractor = self.task_task_feature_factory(
            fastsim.TaskTaskSharedDataFeature(state)
        )

        task_data_feature_extractor = self.task_data_feature_factory(
            fastsim.TaskDataRelativeSizeFeature(state),
            fastsim.TaskDataUsageFeature(state),
        )

        task_device_feature_extractor = None
        data_device_feature_extractor = None

        return ExternalObserver(
            simulator,
            graph_spec,
            graph_extractor,
            task_feature_extractor,
            data_feature_extractor,
            device_feature_extractor,
            task_task_feature_extractor,
            task_data_feature_extractor,
            task_device_feature_extractor,
            data_device_feature_extractor,
        )


class DefaultObserverFactory(ExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec):
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        task_feature_factory.add(fastsim.InDegreeTaskFeature)
        task_feature_factory.add(fastsim.OutDegreeTaskFeature)
        task_feature_factory.add(fastsim.TaskStateFeature)
        task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(fastsim.EmptyTaskFeature, 1)

        data_feature_factory = FeatureExtractorFactory()
        # data_feature_factory.add(fastsim.DataSizeFeature)
        data_feature_factory.add(fastsim.DataMappedLocationsFeature)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(fastsim.DeviceArchitectureFeature)
        device_feature_factory.add(fastsim.DeviceIDFeature)
        device_feature_factory.add(fastsim.DeviceMemoryFeature)
        device_feature_factory.add(fastsim.DeviceTimeFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.TaskTaskSharedDataFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.TaskDataRelativeSizeFeature)
        task_data_feature_factory.add(fastsim.TaskDataUsageFeature)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = None

        super().__init__(
            spec,
            graph_extractor_t,
            task_feature_factory,
            data_feature_factory,
            device_feature_factory,
            task_task_feature_factory,
            task_data_feature_factory,
            task_device_feature_factory,
            data_device_feature_factory,
        )


def observation_to_heterodata_truncate(
    observation: TensorDict, idx: int = 0, device="cpu", actions=None
) -> HeteroData:
    hetero_data = HeteroData()

    if actions is not None:
        hetero_data["actions"].x = actions

    for node_type, node_data in observation["nodes"].items():
        count = node_data["count"][0]
        hetero_data[f"{node_type}"].x = node_data["attr"][:count]

    for edge_key, edge_data in observation["edges"].items():
        target, source = edge_key.split("_")
        count = edge_data["count"][0]
        hetero_data[target, "to", source].edge_index = edge_data["idx"][:, :count]
        hetero_data[target, "to", source].edge_attr = edge_data["attr"][:count]

        if source != target:
            hetero_data[source, "to", target].edge_index = hetero_data[
                target, "to", source
            ].edge_index.flip(0)
            hetero_data[source, "to", target].edge_attr = hetero_data[
                target, "to", source
            ].edge_attr

    return hetero_data.to(device)


def observation_to_heterodata(
    observation: TensorDict, idx: int = 0, device="cpu", actions=None
) -> HeteroData:
    hetero_data = HeteroData()

    if actions is not None:
        # print("setting actions", actions.shape)
        hetero_data["actions"].x = actions

    for node_type, node_data in observation["nodes"].items():
        count = node_data["count"]
        hetero_data[f"{node_type}"].x = node_data["attr"]

    for edge_key, edge_data in observation["edges"].items():
        target, source = edge_key.split("_")
        count = edge_data["count"]
        hetero_data[target, "to", source].edge_index = edge_data["idx"]
        hetero_data[target, "to", source].edge_attr = edge_data["attr"]

        if source != target:
            hetero_data[source, "to", target].edge_index = hetero_data[
                target, "to", source
            ].edge_index.flip(0)
            hetero_data[source, "to", target].edge_attr = hetero_data[
                target, "to", source
            ].edge_attr

    return hetero_data.to(device)


@dataclass
class ExternalObserver:
    simulator: Simulator
    graph_spec: fastsim.GraphSpec
    graph_extractor: fastsim.GraphExtractor
    task_features: fastsim.RuntimeFeatureExtractor
    data_features: fastsim.RuntimeFeatureExtractor
    device_features: fastsim.RuntimeFeatureExtractor
    task_task_features: fastsim.RuntimeEdgeFeatureExtractor
    task_data_features: fastsim.RuntimeEdgeFeatureExtractor
    task_device_features: Optional[fastsim.RuntimeEdgeFeatureExtractor]
    data_device_features: Optional[fastsim.RuntimeEdgeFeatureExtractor]
    truncate: bool = True

    @property
    def task_feature_dim(self):
        if self.task_features is None:
            return 0
        return self.task_features.feature_dim

    @property
    def data_feature_dim(self):
        if self.data_features is None:
            return 0
        return self.data_features.feature_dim

    @property
    def device_feature_dim(self):
        if self.device_features is None:
            return 0
        return self.device_features.feature_dim

    @property
    def task_data_edge_dim(self):
        if self.task_data_features is None:
            return
        return self.task_data_features.feature_dim

    @property
    def task_device_edge_dim(self):
        if self.task_device_features is None:
            return 0
        return self.task_device_features.feature_dim

    @property
    def task_task_edge_dim(self):
        if self.task_task_features is None:
            return 0
        return self.task_task_features.feature_dim

    def get_task_features(self, task_ids, workspace):
        length = self.task_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_data_features(self, data_ids, workspace):
        length = self.data_features.get_features_batch(data_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_device_features(self, device_ids, workspace):
        length = self.device_features.get_features_batch(device_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_task_features(self, task_ids, workspace):
        length = self.task_task_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_data_features(self, task_ids, workspace):
        length = self.task_data_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_device_features(self, task_ids, workspace):
        length = self.task_device_features.get_features_batch(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_bidirectional_neighborhood(self, task_ids, workspace):
        length = self.graph_extractor.get_k_hop_bidirectional(task_ids, 1, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_used_data(self, task_ids, workspace):
        length = self.graph_extractor.get_unique_data(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_task_task_edges(self, task_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_task_edges(
            task_ids, workspace, global_workspace
        )

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_data_edges(self, task_ids, data_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_data_edges(
            task_ids, data_ids, workspace, global_workspace
        )

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_device_edges(self, task_ids, device_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_device_edges(
            task_ids, workspace, global_workspace
        )

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def _local_to_global(self, global_ids, local_ids, workspace=None):
        if workspace is not None:
            workspace[: len(local_ids)] = global_ids[local_ids]
            return workspace
        else:
            return global_ids[local_ids]

    def _local_to_global2D(self, g1, g2, l, workspace=None):
        if workspace is not None:
            size = len(l[0, :])
            workspace[0, :size] = g1[l[0, :]][:size]
            workspace[1, :size] = g2[l[1, :]][:size]
            return workspace
        else:
            id1 = g1[l[0, :]]
            id2 = g2[l[1, :]]
            return torch.stack((id1, id2), dim=0)

    def _local_to_global2D_same(self, g1, l, workspace=None):
        if workspace is not None:
            size = len(l[0, :])
            workspace[:, :size] = g1[l][:size]
            return workspace
        else:
            return g1[l]

    def new_observation_buffer(self, spec: Optional[fastsim.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec

        def _make_node_tensor(nodes, dim):
            return TensorDict(
                {
                    "glb": torch.zeros((nodes), dtype=torch.int64),
                    "attr": torch.zeros((nodes, dim), dtype=torch.float32),
                    "count": torch.zeros((1), dtype=torch.int64),
                    # "count": torch.tensor([0], dtype=torch.int64),
                }
            )

        def _make_edge_tensor(edges, dim):
            return TensorDict(
                {
                    "glb": torch.zeros((2, edges), dtype=torch.int64),
                    "idx": torch.zeros((2, edges), dtype=torch.int64),
                    "attr": torch.zeros((edges, dim), dtype=torch.float32),
                    "count": torch.zeros((1), dtype=torch.int64),
                    # "count": torch.tensor([0], dtype=torch.int64),
                }
            )

        def _make_index_tensor(n):
            return TensorDict(
                {
                    "idx": torch.zeros((n), dtype=torch.int64),
                    "count": torch.zeros((1), dtype=torch.int64),
                    # "count": torch.tensor([0], dtype=torch.int64),
                }
            )

        node_tensor = TensorDict(
            {
                "tasks": _make_node_tensor(
                    spec.max_tasks, self.task_features.feature_dim
                ),
                "data": _make_node_tensor(
                    spec.max_data, self.data_features.feature_dim
                ),
                "devices": _make_node_tensor(
                    spec.max_devices, self.device_features.feature_dim
                ),
            }
        )

        edge_tensor = TensorDict(
            {
                "tasks_tasks": _make_edge_tensor(
                    spec.max_edges_tasks_tasks, self.task_task_features.feature_dim
                ),
                "tasks_data": _make_edge_tensor(
                    spec.max_edges_tasks_data, self.task_data_features.feature_dim
                ),
                "tasks_devices": _make_edge_tensor(
                    spec.max_edges_tasks_devices, self.task_device_features.feature_dim
                ),
            }
        )

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                # "time": torch.tensor([0], dtype=torch.int64),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": node_tensor,
                "edges": edge_tensor,
                "aux": aux_tensor,
            }
        )

        return obs_tensor

    def task_observation(
        self, output: TensorDict, task_ids: Optional[torch.Tensor] = None
    ):
        if task_ids is None:
            n_candidates = output["aux"]["candidates"]["count"][0]
            task_ids = output["aux"]["candidates"]["idx"][:n_candidates]
            output["nodes"]["tasks"]["attr"][:n_candidates, -1] = 1

        _, count = self.get_bidirectional_neighborhood(
            task_ids, output["nodes"]["tasks"]["glb"]
        )
        output["nodes"]["tasks"]["count"][0] = count
        self.get_task_features(
            output["nodes"]["tasks"]["glb"][:count], output["nodes"]["tasks"]["attr"]
        )

    def data_observation(self, output: TensorDict):
        ntasks = output["nodes"]["tasks"]["count"][0]
        _, count = self.get_used_data(
            output["nodes"]["tasks"]["glb"][:ntasks], output["nodes"]["data"]["glb"]
        )
        output["nodes"]["data"]["count"][0] = count
        self.get_data_features(
            output["nodes"]["data"]["glb"][:count], output["nodes"]["data"]["attr"]
        )

    def device_observation(self, output: TensorDict):
        count = output["nodes"]["devices"]["glb"].shape[0]
        output["nodes"]["devices"]["count"][0] = count
        output["nodes"]["devices"]["glb"][:count] = torch.arange(
            count, dtype=torch.int64
        )
        self.get_device_features(
            output["nodes"]["devices"]["glb"][:count],
            output["nodes"]["devices"]["attr"],
        )

    def task_task_observation(self, output: TensorDict):
        ntasks = output["nodes"]["tasks"]["count"][0]

        _, count = self.get_task_task_edges(
            output["nodes"]["tasks"]["glb"][:ntasks],
            output["edges"]["tasks_tasks"]["idx"],
            output["edges"]["tasks_tasks"]["glb"],
        )
        output["edges"]["tasks_tasks"]["count"][0] = count

        self.get_task_task_features(
            output["edges"]["tasks_tasks"]["glb"][:, :count],
            output["edges"]["tasks_tasks"]["attr"],
        )

    def task_data_observation(self, output: TensorDict):
        ntasks = output["nodes"]["tasks"]["count"][0]
        ndata = output["nodes"]["data"]["count"][0]
        _, count = self.get_task_data_edges(
            output["nodes"]["tasks"]["glb"][:ntasks],
            output["nodes"]["data"]["glb"][:ndata],
            output["edges"]["tasks_data"]["idx"],
            output["edges"]["tasks_data"]["glb"],
        )
        output["edges"]["tasks_data"]["count"][0] = count

        self.get_task_data_features(
            output["edges"]["tasks_data"]["glb"][:, :count],
            output["edges"]["tasks_data"]["attr"],
        )

    def task_device_observation(self, output: TensorDict, use_all_tasks=True):
        if not use_all_tasks:
            ncandidates = output["aux"]["candidates"]["count"][0]
            task_ids = output["aux"]["candidates"]["idx"][:ncandidates]
        else:
            ntasks = output["nodes"]["tasks"]["count"][0]
            task_ids = output["nodes"]["tasks"]["glb"][:ntasks]

        ndevices = output["nodes"]["devices"]["count"][0]
        device_ids = output["nodes"]["devices"]["glb"][:ndevices]

        _, count = self.get_task_device_edges(
            task_ids,
            device_ids,
            output["edges"]["tasks_devices"]["idx"],
            output["edges"]["tasks_devices"]["glb"],
        )

        output["edges"]["tasks_devices"]["count"][0] = count

        self.get_task_device_features(
            output["edges"]["tasks_devices"]["glb"][:, :count],
            output["edges"]["tasks_devices"]["attr"],
        )

    def candidate_observation(self, output: TensorDict):
        count = self.simulator.get_mappable_candidates(
            output["aux"]["candidates"]["idx"]
        )
        output["aux"]["candidates"]["count"][0] = count

    def get_observation(self, output: Optional[TensorDict] = None):
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)

        # Get mappable candidates
        self.candidate_observation(output)

        # Node observations (all nodes must be processed before edges)
        self.task_observation(output)
        self.data_observation(output)
        self.device_observation(output)

        # Edge observations (edges depend on ids collected during node observation)
        self.task_task_observation(output)
        self.task_data_observation(output)
        self.task_device_observation(output)

        # Auxiliary observations
        output["aux"]["time"][0] = self.simulator.get_current_time()
        output["aux"]["improvement"][0] = -2.0
        return output


@dataclass
class SimulatorDriver:
    input: SimulatorInput
    internal_mapper: fastsim.Mapper
    external_mapper: ExternalMapper
    simulator: fastsim.Simulator
    observer_factory: Optional[ExternalObserverFactory]
    observer: Optional[ExternalObserver]
    use_external_mapper: bool = False

    def __init__(
        self,
        input: SimulatorInput,
        internal_mapper: (
            fastsim.Mapper | Type[fastsim.Mapper]
        ) = fastsim.DequeueEFTMapper,
        external_mapper: ExternalMapper | Type[ExternalMapper] = ExternalMapper,
        observer_factory: Optional[ExternalObserverFactory] = None,
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

        if simulator is None:
            self.simulator = fastsim.Simulator(input.to_input(), self.internal_mapper)
        else:
            self.simulator = simulator
            self.simulator.set_mapper(self.internal_mapper)

        if observer_factory is not None:
            self.observer_factory = observer_factory
            self.observer = observer_factory.create(self.simulator)

    def get_state(self):
        return self.simulator.get_state()

    @property
    def state(self):
        return self.simulator.get_state()

    @property
    def status(self):
        return self.simulator.last_execution_state

    def get_mappable_candidates(self, candidates: torch.Tensor):
        """
        Get the mappable candidates from the simulator.
        """
        return self.simulator.get_mappable_candidates(candidates)

    def get_mapping_priority(self, task_id: int):
        """
        Get the mapping priority for a task.
        """
        return self.state.get_mapping_priority(task_id)

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

    @property
    def mapper(self):
        if self.use_external_mapper:
            return self.external_mapper
        return self.internal_mapper

    def enable_external_mapper(
        self, external_mapper: Optional[ExternalMapper | Type[ExternalMapper]] = None
    ):
        """
        Use external mapper for mapping tasks (run Python callback).
        """
        if external_mapper is not None:
            if isinstance(external_mapper, type):
                external_mapper = external_mapper()

            self.external_mapper = external_mapper

        self.use_external_mapper = True
        self.simulator.enable_python_mapper()

    def disable_external_mapper(self):
        """
        Use internal mapper for mapping tasks (do not run Python callback).
        """
        self.use_external_mapper = False
        self.simulator.disable_python_mapper()

        if self.simulator.last_execution_state == ExecutionState.EXTERNAL_MAPPING:
            self.simulator.skip_external_mapping()

    def fresh_copy(self) -> "SimulatorDriver":
        """
        Initialize a fresh (uninitialized) copy of the simulator driver with the same initial input and configuration.
        """
        internal_mapper_t = type(self.internal_mapper)
        external_mapper_t = type(self.external_mapper)

        internal_mapper_copy = internal_mapper_t()
        external_mapper_copy = external_mapper_t()

        observer_factory = self.observer_factory

        return SimulatorDriver(
            input=self.input,
            internal_mapper=internal_mapper_copy,
            external_mapper=external_mapper_copy,
            observer_factory=observer_factory,
            simulator=None,
        )

    def reset(self):
        """
        Return a fresh copy of the simulator driver with the same initial input and configuration.
        (This is equivalent to calling fresh_copy()).
        """
        return self.fresh_copy()

    def time(self) -> int:
        """
        Returns the current time (in microseconds) of the simulator state.
        """
        return self.simulator.get_current_time()

    @property
    def time(self) -> int:
        """
        Returns the current time (in microseconds) of the simulator state.
        """
        return self.simulator.get_current_time()

    def task_finish_time(self, task_id: int) -> int:
        """
        Returns the finish time (in microseconds) of a task.
        """
        return self.simulator.get_task_finish_time(task_id)

    def copy(self) -> "SimulatorDriver":
        """
        Initialize a copy of the simulator driver at the current state (may be initialized if the source simulator is).
        Mappers and their internal state (if any) are copied as well.
        """
        internal_mapper_t = type(self.internal_mapper)
        external_mapper_t = type(self.external_mapper)

        internal_mapper_copy = internal_mapper_t(self.internal_mapper)
        external_mapper_copy = external_mapper_t(self.external_mapper)

        observer_factory = self.observer_factory

        simulator_copy = fastsim.Simulator(self.simulator)

        new_sim_driver = SimulatorDriver(
            input=self.input,
            internal_mapper=internal_mapper_copy,
            external_mapper=external_mapper_copy,
            observer_factory=observer_factory,
            simulator=simulator_copy,
        )
        return new_sim_driver

    def run_until_external_mapping(self) -> ExecutionState:
        """
        Run the simulator until a breakpoint, error, completion, or external mapping is reached.
        Will return the current state of the simulator at the exitpoint.
        """
        sim_state = self.simulator.run()
        return sim_state

    def run(self) -> ExecutionState:
        """
        Run the simulator until a breakpoint, error, or completion is reached.
        This DOES NOT STOP for external mapping. Use run_until_external_mapping() for that.
        External mapping will be called, if enabled, inside this function.
        Will return the current state of the simulator at the exitpoint.
        """
        sim_state = ExecutionState.RUNNING
        while sim_state == ExecutionState.RUNNING:
            sim_state = self.simulator.run()

            if sim_state == ExecutionState.BREAKPOINT:
                return sim_state

            if sim_state == ExecutionState.ERROR:
                return sim_state

            if sim_state == ExecutionState.EXTERNAL_MAPPING:
                actions = self.external_mapper.map_tasks(self)
                self.simulator.map_tasks(actions)
                sim_state = ExecutionState.RUNNING
        return sim_state


def create_graph_spec(
    max_tasks: int = 30,
    max_data: int = 30,
    max_devices: int = 5,
    max_edges_tasks_tasks: int = 30,
    max_edges_tasks_data: int = 30,
    max_candidates: int = 1,
):
    """
    Create a graph spec with the specified limits for tasks, data, devices, edges, and candidates.

    Parameters:
    max_tasks (int): The maximum number of task nodes.
    max_data (int): The maximum number of data nodes.
    max_devices (int): The maximum number of device nodes.
    max_edges_tasks_tasks (int): The maximum number of edges between task nodes.
    max_edges_tasks_data (int): The maximum number of edges between task and data nodes.
    max_candidates (int): The maximum number of candidate tasks to consider for mapping.
    """
    spec = fastsim.GraphSpec()
    spec.max_tasks = max_tasks
    spec.max_data = max_data
    spec.max_devices = max_devices
    spec.max_edges_tasks_tasks = max_edges_tasks_tasks
    spec.max_edges_tasks_data = max_edges_tasks_data
    spec.max_candidates = max_candidates

    # This should be max_candidates, but reverting to max_tasks to implement original NN architecture
    # spec.max_edges_tasks_devices = max_devices * max_candidates
    spec.max_edges_tasks_devices = max_devices * max_tasks
    return spec


class SimulatorFactory:
    def __init__(
        self,
        input: SimulatorInput,
        graph_spec: fastsim.GraphSpec,
        observer_factory: ExternalObserverFactory | Type[ExternalObserverFactory],
        internal_mapper: (
            fastsim.Mapper | Type[fastsim.Mapper]
        ) = fastsim.DequeueEFTMapper,
        external_mapper: ExternalMapper | Type[ExternalMapper] = ExternalMapper,
        seed: int = 0,
        priority_seed: int = 0,
        comm_seed: int = 0,
    ):
        self.input = input
        self.graph_spec = graph_spec
        self.internal_mapper = internal_mapper
        self.external_mapper = external_mapper

        self.seed = seed
        self.pseed = priority_seed
        self.cseed = comm_seed

        if isinstance(observer_factory, type):
            observer_factory = observer_factory(graph_spec)
        self.observer_factory = observer_factory

    def create(
        self,
        duration_seed: Optional[int] = None,
        priority_seed: Optional[int] = None,
        comm_seed: Optional[int] = None,
        use_external_mapper: bool = True,
    ) -> SimulatorDriver:
        if duration_seed is None:
            duration_seed = self.seed

        if priority_seed is None:
            priority_seed = self.pseed

        if comm_seed is None:
            comm_seed = self.cseed

        self.input.noise.task_noise.set_seed(duration_seed)
        self.input.noise.task_noise.set_pseed(priority_seed)

        simulator = SimulatorDriver(
            self.input,
            observer_factory=self.observer_factory,
            internal_mapper=self.internal_mapper,
            external_mapper=self.external_mapper,
        )
        self.input.noise.task_noise.randomize_duration()
        self.input.noise.task_noise.randomize_priority()
        simulator.initialize()
        simulator.initialize_data()
        if use_external_mapper:
            simulator.enable_external_mapper()
        else:
            simulator.disable_external_mapper()

        return simulator

    def set_seed(
        self,
        seed: Optional[int] = None,
        priority_seed: Optional[int] = None,
        comm_seed: Optional[int] = None,
    ):
        """
        Set the seed for the simulator.
        """
        if seed is not None:
            self.seed = seed
        if priority_seed is not None:
            self.pseed = priority_seed
        if comm_seed is not None:
            self.cseed = comm_seed


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
