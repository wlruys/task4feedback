from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Type, Self
from .types import (
    DeviceTuple,
    TaskTuple,
    DataBlockTuple,
    VariantTuple,
    ConnectionTuple,
    _bytes_to_readable,
)
import cxxfilt
from .lambdas import VariantBuilder, TaskLabeler, DataBlockTransformer
from rich import print
import numpy as np
import task4feedback.fastsim2 as fastsim
from task4feedback.fastsim2 import (
    Devices,
    Topology,
    Graph,
    TaskNoise,
    LognormalTaskNoise,
    StaticTaskInfo,
    RuntimeTaskInfo,
    Data,
    DeviceType,
    SchedulerInput,
    RangeTransitionConditions,
    DefaultTransitionConditions,
    BatchTransitionConditions,
    SchedulerState,
    Simulator,
)
from task4feedback.fastsim2 import ExecutionState, start_logger, EventType
import torch
from tensordict.tensordict import TensorDict
from torch_geometric.data import HeteroData, Batch


def _make_node_tensor(nodes, dim, single=False):
    if single:
        glb_shape = 1
        attr_shape = dim
    else:
        glb_shape = (nodes,)
        attr_shape = (nodes, dim)

    return TensorDict(
        {
            "glb": torch.zeros(glb_shape, dtype=torch.int64),
            "attr": torch.zeros(attr_shape, dtype=torch.float32),
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
            "count": torch.zeros((1,), dtype=torch.int64),
            # "count": torch.tensor([0], dtype=torch.int64),
        }
    )


class TaskGraph:
    def __init__(self):
        self.graph = Graph()
        self.static_graph = None
        self.tasks: dict[int, TaskTuple] = {}

    def add_task(self, name):
        idx = self.graph.add_task(name)
        self.tasks[idx] = TaskTuple(id=idx, name=name)
        return idx

    def add_tag(self, task_id, tag):
        self.graph.set_tag(task_id, tag)
        self.tasks[task_id].tag = tag

    def get_task(self, task_id):
        if task_id in self.tasks:
            return self.tasks[task_id]
        else:
            raise KeyError(f"Task with ID {task_id} does not exist in the graph.")

    def __len__(self):
        return self.graph.size()

    def get_task_dependencies(self, task_id: int) -> list[int]:
        """
        Return the list of dependency task IDs for the given task.
        """
        return self.graph.get_task_dependencies(task_id)

    def __len__(self):
        return self.graph.size()

    def __iter__(self):
        return iter(self.tasks.values())

    def add_dependencies(self, task, dependencies):
        self.tasks[task].dependencies.extend(dependencies)
        self.graph.add_dependencies(task, dependencies)

    def add_dependency(self, task, dependency):
        self.tasks[task].dependencies.append(dependency)
        self.graph.add_dependency(task, dependency)

    def add_read_data(self, task, dataidlist):
        self.tasks[task].read.extend(dataidlist)
        self.graph.add_read_data(task, dataidlist)

    def add_write_data(self, task, dataidlist):
        self.tasks[task].write.extend(dataidlist)
        self.graph.add_write_data(task, dataidlist)

    def add_retire_data(self, task, dataidlist):
        self.tasks[task].retire.extend(dataidlist)
        self.graph.add_retire_data(task, dataidlist)

    def apply_variant(self, variant_builder: type[VariantBuilder]):
        for i in range(self.graph.get_n_compute_tasks()):
            task = self.get_task(i)
            for arch in DeviceType:
                variant = variant_builder.build_variant(arch, task)

                if variant is None:
                    continue

                vcu_usage = int(variant.vcu_usage * fastsim.MAX_VCUS)
                self.graph.set_variant(
                    i,
                    arch,
                    vcu_usage,
                    variant.memory_usage,
                    variant.expected_time,
                )

    def finalize(self):
        self.graph.finalize()
        self.static_graph = StaticTaskInfo(self.graph)
        for task_id, v in self.tasks.items():
            v.dependencies = self.get_task_dependencies(task_id)

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

    def print_variants(self):
        for i in range(self.graph.get_n_compute_tasks()):
            task = self.get_task(i)
            print(f"Task {task.id}: {task.name} (Tag: {task.tag})")
            print("  Variants:")
            print(compute_task.get_variants())
            print("Dependencies:")
            print(compute_task.get_dependencies())
            print("Read Data:")
            print(compute_task.get_read())
            print("Write Data:")
            print(compute_task.get_write())

    def __str__(self):
        result = []

        task_count = self.graph.get_n_compute_tasks()
        result.append(f"TaskGraph with {task_count} tasks:")

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

    def add_block(self, name, size, location=0, id=None, x_pos=0, y_pos=0):
        if id is None:
            id = self.data.append_block(size, location, name)
        else:
            self.data.create_block(id, size, location, name)

        if x_pos != 0:
            self.data.set_x_pos(id, x_pos)

        if y_pos != 0:
            self.data.set_y_pos(id, y_pos)

        return DataBlockTuple(id, name, size, location)

    def set_location(self, block, location, convert=False):
        if convert and isinstance(block, str):
            block = self.data.get_id(block)
        self.data.set_location(block, location)

    def set_size(self, block, size, convert=False):
        if convert and isinstance(block, str):
            block = self.data.get_id(block)
        self.data.set_size(block, size)

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

    def create_device(self, name, arch, copy, memory):
        id = self.devices.append_device(name, arch, copy, memory)
        return DeviceTuple(name, id, self.devices.get_local_id(id), arch, memory)

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

    def __len__(self):
        return self.devices.size()

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


# @dataclass
# class NoiseConfig:
#     task_noise: TaskNoise

#     def __init__(
#         self,
#         graph: TaskGraph,
#         duration_seed: int = 0,
#         priority_seed: int = 0,
#     ):
#         self.task_noise = TaskNoise(graph.static_graph, duration_seed, priority_seed)


# @dataclass 
# class LognormalNoiseConfig(NoiseConfig):

#     def __init__(
#         self,
#         graph: TaskGraph,
#         duration_seed: int = 0,
#         priority_seed: int = 0,
#     ):
#         super().__init__(graph, duration_seed, priority_seed)
#         self.task_noise = LognormalTaskNoise(graph.static_graph, duration_seed, priority_seed)


class ExternalMapper:
    def __init__(self, mapper: Optional[Self] = None):
        pass

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        # print(candidate_tasks)
        candidates = torch.zeros((1), dtype=torch.int64)
        simulator.simulator.get_mappable_candidates(candidates)
        global_task_id = candidates[0]
        local_id = 0
        device = 1
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
    graph: TaskGraph
    data: DataBlocks
    system: System
    task_noise: TaskNoise
    transition_conditions: fastsim.TransitionConditions

    def __init__(
        self,
        graph: TaskGraph,
        data: DataBlocks,
        system: System,
        task_noise: Optional[TaskNoise] = None,
        transition_conditions: Optional[fastsim.TransitionConditions] = None,
    ):
        if transition_conditions is None:
            transition_conditions = fastsim.RangeTransitionConditions(5, 5, 16)
            
        if task_noise is None:
            task_noise = TaskNoise(graph.static_graph)

        self.task_noise = task_noise
        self.graph = graph
        self.data = data
        self.system = system
        self.transition_conditions = transition_conditions

    def to_input(self):
        return SchedulerInput(
            self.graph.graph,
            self.graph.static_graph,
            self.data.data,
            self.system.devices,
            self.system.topology,
            self.task_noise,
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
            # fastsim.OutDegreeTaskFeature(state),
            # fastsim.OneHotMappedDeviceTaskFeature(state),
        )
        data_feature_extractor = self.data_feature_factory(
            fastsim.DataSizeFeature(state),
            # fastsim.DataMappedLocationsFeature(state),
        )

        device_feature_extractor = self.device_feature_factory(
            fastsim.DeviceArchitectureFeature(state),
            fastsim.DeviceIDFeature(state),
            # fastsim.DeviceMemoryFeature(state),
            # fastsim.DeviceTimeFeature(state),
        )

        task_task_feature_extractor = self.task_task_feature_factory(
            fastsim.TaskTaskSharedDataFeature(state)
        )

        task_data_feature_extractor = self.task_data_feature_factory(
            # fastsim.TaskDataRelativeSizeFeature(state),
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
        # task_feature_factory.add(fastsim.TaskStateFeature)
        task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(fastsim.EmptyTaskFeature, 1)

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.DataSizeFeature)
        data_feature_factory.add(fastsim.DataMappedLocationsFeature)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(fastsim.DeviceArchitectureFeature)
        device_feature_factory.add(fastsim.DeviceIDFeature)
        # device_feature_factory.add(fastsim.DeviceMemoryFeature)
        # device_feature_factory.add(fastsim.DeviceTimeFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.TaskTaskSharedDataFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.TaskDataRelativeSizeFeature)
        # task_data_feature_factory.add(fastsim.TaskDataUsageFeature)

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
        splits = edge_key.split("_")

        if len(splits) == 2:
            target, source = splits
            usage = None
        elif len(splits) == 3:
            target, usage, source = splits
        else:
            raise ValueError(f"Invalid edge key format: {edge_key}")

        count = edge_data["count"][0]

        if usage is None:
            hetero_data[target, "to", source].edge_index = edge_data["idx"][:, :count]
            hetero_data[target, "to", source].edge_attr = edge_data["attr"][:count]

            if source != target:
                hetero_data[source, "to", target].edge_index = hetero_data[
                    target, "to", source
                ].edge_index.flip(0)
                hetero_data[source, "to", target].edge_attr = hetero_data[
                    target, "to", source
                ].edge_attr

            if source == target:
                hetero_data[source, "from", target].edge_index = hetero_data[
                    target, "to", source
                ].edge_index.flip(0)
                hetero_data[source, "from", target].edge_attr = hetero_data[
                    target, "to", source
                ].edge_attr
        else:
            hetero_data[target, usage, source].edge_index = edge_data["idx"][:, :count]
            hetero_data[target, usage, source].edge_attr = edge_data["attr"][:count]

            if source != target:
                hetero_data[source, usage, target].edge_index = hetero_data[
                    target, usage, source
                ].edge_index.flip(0)
                hetero_data[source, usage, target].edge_attr = hetero_data[
                    target, usage, source
                ].edge_attr

    return hetero_data.to(device)


def observation_to_heterodata(
    observation: TensorDict, idx: int = 0, device="cpu", actions=None
) -> HeteroData:
    hetero_data = HeteroData()

    hetero_data["time"].x = observation["aux"]["time"].unsqueeze(0)

    if actions is not None:
        # print("setting actions", actions.shape)
        hetero_data["actions"].x = actions

    for node_type, node_data in observation["nodes"].items():
        count = node_data["count"]
        # print("node count", node_type, count)
        hetero_data[f"{node_type}"].x = node_data["attr"]
        hetero_data[f"{node_type}_count"].x = count

    for edge_key, edge_data in observation["edges"].items():
        splits = edge_key.split("_")

        if len(splits) == 2:
            target, source = splits
            usage = None
        elif len(splits) == 3:
            target, usage, source = splits
        else:
            raise ValueError(f"Invalid edge key format: {edge_key}")

        count = edge_data["count"]

        if usage is None:
            hetero_data[target, "to", source].edge_index = edge_data["idx"]
            hetero_data[target, "to", source].edge_attr = edge_data["attr"]

            if source != target:
                hetero_data[source, "to", target].edge_index = hetero_data[
                    target, "to", source
                ].edge_index.flip(0)
                hetero_data[source, "to", target].edge_attr = hetero_data[
                    target, "to", source
                ].edge_attr

            if source == target:
                hetero_data[source, "from", target].edge_index = hetero_data[
                    target, "to", source
                ].edge_index.flip(0)
                hetero_data[source, "from", target].edge_attr = hetero_data[
                    target, "to", source
                ].edge_attr
        else:
            hetero_data[target, usage, source].edge_index = edge_data["idx"]
            hetero_data[target, usage, source].edge_attr = edge_data["attr"]

            if source != target:
                hetero_data[source, usage, target].edge_index = hetero_data[
                    target, usage, source
                ].edge_index.flip(0)
                hetero_data[source, usage, target].edge_attr = hetero_data[
                    target, usage, source
                ].edge_attr

    return hetero_data.to(device)


class AccessType(IntEnum):
    READ_WRITE: int = 0
    READ: int = 1
    WRITE: int = 2
    READ_MAPPED: int = 3
    RETIRE: int = 4


class NeighborhoodType(IntEnum):
    DEPENDENCIES: int = 0
    DEPENDENTS: int = 1
    BIDIRECTIONAL: int = 2
    ITERATIVE: int = 3


@dataclass
class ExternalObserver:
    simulator: "SimulatorDriver"
    graph_spec: fastsim.GraphSpec
    graph_extractor: fastsim.GraphExtractor
    task_features: fastsim.RuntimeFeatureExtractor
    data_features: fastsim.RuntimeFeatureExtractor
    device_features: fastsim.RuntimeFeatureExtractor
    task_task_features: fastsim.RuntimeEdgeFeatureExtractor
    task_data_features: fastsim.RuntimeEdgeFeatureExtractor
    task_device_features: Optional[fastsim.RuntimeEdgeFeatureExtractor] = None
    data_device_features: Optional[fastsim.RuntimeEdgeFeatureExtractor] = None
    truncate: bool = True

    def store_feature_types(self):
        """
        Store feature type information in the provided config dictionary.
        Includes both the feature extractor class names and the specific feature types.
        """
        config_dictionary = {}
        for field_name, field_value in self.__dict__.items():
            if field_name.endswith("features") and field_value is not None:
                # Store the class name of the feature extractor
                config_dictionary[field_name] = field_value.__class__.__name__

                # Store the specific feature type names
                if hasattr(field_value, "feature_type_names"):
                    feature_types = field_value.feature_type_names
                    feature_types = [cxxfilt.demangle(t) for t in feature_types]
                    # Format feature type names for better readability
                    formatted_types = [
                        t.split("::")[-1] if "::" in t else t for t in feature_types
                    ]

                    print(f"Feature types for {field_name}: {formatted_types}")
                    config_dictionary[f"{field_name}_types"] = formatted_types

        return config_dictionary

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

    def get_data_device_features(self, data_ids, workspace):
        length = self.data_device_features.get_features_batch(data_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_neighborhood(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_neighborhood(task_ids, depth, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_bidirectional(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_bidirectional(
            task_ids, depth, workspace
        )

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_dependencies(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_dependencies(task_ids, depth, workspace)
        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_k_hop_dependents(self, task_ids, workspace, depth: int = 1):
        length = self.graph_extractor.get_k_hop_dependents(task_ids, depth, workspace)
        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_used_data(self, task_ids, workspace):
        length = self.graph_extractor.get_unique_data(task_ids, workspace)

        if self.truncate:
            workspace = workspace[:length]
        return workspace, length

    def get_used_filtered_data(self, task_ids, workspace):
        """
        Only return data whose most recent writer has been mapped
        """
        length = self.graph_extractor.get_unique_filtered_data(task_ids, workspace)

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

    def get_task_data_edges(
        self,
        task_ids,
        data_ids,
        workspace,
        global_workspace,
        access_type: AccessType = AccessType.READ_WRITE,
    ):
        if access_type == AccessType.READ_WRITE:
            length = self.graph_extractor.get_task_data_edges_all(
                task_ids, data_ids, workspace, global_workspace
            )
        elif access_type == AccessType.READ:
            length = self.graph_extractor.get_task_data_edges_read(
                task_ids, data_ids, workspace, global_workspace
            )
        elif access_type == AccessType.WRITE:
            length = self.graph_extractor.get_task_data_edges_write(
                task_ids, data_ids, workspace, global_workspace
            )
        elif access_type == AccessType.READ_MAPPED:
            length = self.graph_extractor.get_task_data_edges_read_mapped(
                task_ids, data_ids, workspace, global_workspace
            )
        else:
            raise ValueError(
                f"Invalid access type operation for get_task_data_edges: {access_type}"
            )

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_task_edges_reverse(self, task_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_task_edges_reverse(
            task_ids, workspace, global_workspace
        )

        if self.truncate:
            workspace = workspace[:, :length]
        return workspace, length

    def get_task_device_edges(self, task_ids, workspace, global_workspace):
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
        # print("Making new buffer", self.task_feature_dim)
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
                "data_devices": _make_edge_tensor(
                    spec.max_edges_data_devices, self.data_device_features.feature_dim
                ),
            }
        )

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
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
        self,
        output: TensorDict,
        task_ids: Optional[torch.Tensor] = None,
        k: int = 1,
        neighborhood_type: NeighborhoodType = NeighborhoodType.BIDIRECTIONAL,
    ):
        # print("Task observation")
        if task_ids is None:
            n_candidates = output["aux"]["candidates"]["count"][0]
            task_ids = output["aux"]["candidates"]["idx"][:n_candidates]

        if neighborhood_type == NeighborhoodType.BIDIRECTIONAL:
            # print("Bidirectional")
            _, count = self.get_k_hop_bidirectional(
                task_ids, output["nodes"]["tasks"]["glb"], k
            )
        elif neighborhood_type == NeighborhoodType.DEPENDENCIES:
            # print("Dependencies")
            _, count = self.get_k_hop_dependencies(
                task_ids, output["nodes"]["tasks"]["glb"]
            )

        elif neighborhood_type == NeighborhoodType.DEPENDENTS:
            # print("Dependents")
            _, count = self.get_k_hop_dependents(
                task_ids, output["nodes"]["tasks"]["glb"]
            )

        elif neighborhood_type == NeighborhoodType.ITERATIVE:
            # print("Iterative")
            _, count = self.get_k_hop_neighborhood(
                task_ids, output["nodes"]["tasks"]["glb"], k
            )
        else:
            raise ValueError(
                f"Invalid neighborhood type operation for task observation: {neighborhood_type}"
            )

        # print("Task count", count)
        output["nodes"]["tasks"]["count"][0] = count
        self.get_task_features(
            output["nodes"]["tasks"]["glb"][:count], output["nodes"]["tasks"]["attr"]
        )
        # print("Task attribute", output["nodes"]["tasks"]["attr"])

    def data_observation(self, output: TensorDict):
        # print("Data observation")
        ntasks = output["nodes"]["tasks"]["count"][0]
        _, count = self.get_used_data(
            output["nodes"]["tasks"]["glb"][:ntasks], output["nodes"]["data"]["glb"]
        )
        output["nodes"]["data"]["count"][0] = count
        self.get_data_features(
            output["nodes"]["data"]["glb"][:count], output["nodes"]["data"]["attr"]
        )

    def device_observation(self, output: TensorDict):
        # print("Device observation")
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
        # print("Task-Task observation")
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
        # print("Task-Data observation")
        ntasks = output["nodes"]["tasks"]["count"][0]
        ndata = output["nodes"]["data"]["count"][0]
        _, count = self.get_task_data_edges(
            output["nodes"]["tasks"]["glb"][:ntasks],
            output["nodes"]["data"]["glb"][:ndata],
            output["edges"]["tasks_data"]["idx"],
            output["edges"]["tasks_data"]["glb"],
            AccessType.READ_MAPPED,
        )
        output["edges"]["tasks_data"]["count"][0] = count

        self.get_task_data_features(
            output["edges"]["tasks_data"]["glb"][:, :count],
            output["edges"]["tasks_data"]["attr"],
        )

    def task_device_observation(self, output: TensorDict, use_all_tasks=False):
        # print("Task-Device observation")
        if not use_all_tasks:
            ncandidates = output["aux"]["candidates"]["count"][0]
            task_ids = output["aux"]["candidates"]["idx"][:ncandidates]
        else:
            ntasks = output["nodes"]["tasks"]["count"][0]
            task_ids = output["nodes"]["tasks"]["glb"][:ntasks]

        ndevices = output["nodes"]["devices"]["count"][0]

        _, count = self.get_task_device_edges(
            task_ids,
            output["edges"]["tasks_devices"]["idx"],
            output["edges"]["tasks_devices"]["glb"],
        )

        output["edges"]["tasks_devices"]["count"][0] = count

        self.get_task_device_features(
            output["edges"]["tasks_devices"]["glb"][:, :count],
            output["edges"]["tasks_devices"]["attr"],
        )

    def candidate_observation(self, output: TensorDict):
        # print("Candidate observation")
        # print("Candidate observation", type(self))
        count = self.simulator.simulator.get_mappable_candidates(
            output["aux", "candidates", "idx"]
        )
        # output["aux"]["candidates"]["count"][0] = count
        output.set_at_(("aux", "candidates", "count"), count, 0)

    def get_observation(self, output: Optional[TensorDict] = None):
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)

        # print(output)

        # Get mappable candidates
        self.candidate_observation(output)

        # Node observations (all nodes must be processed before edges)
        self.task_observation(output)
        self.data_observation(output)
        self.device_observation(output)

        # print("Task attribute", output["nodes"]["tasks"]["attr"])
        # print("Data attribute", output["nodes"]["data"]["attr"])
        # print("Device attribute", output["nodes"]["devices"]["attr"])

        # Edge observations (edges depend on ids collected during node observation)
        self.task_task_observation(output)
        self.task_data_observation(output)
        self.task_device_observation(output)

        # Auxiliary observations
        output["aux"]["time"][0] = self.simulator.time
        output["aux"]["improvement"][0] = -100
        # print("Auxiliary observation")

        return output

    def reset(self):
        """
        Reset the observer state.
        This method can be overridden by subclasses to implement specific reset logic.
        """
        pass


@dataclass
class HeterogeneousExternalObserver(ExternalObserver):
    def __init__(
        self,
        simulator: "SimulatorDriver",
        graph_spec: fastsim.GraphSpec,
        graph_extractor: fastsim.GraphExtractor,
        task_features: fastsim.RuntimeFeatureExtractor,
        data_features: fastsim.RuntimeFeatureExtractor,
        device_features: fastsim.RuntimeFeatureExtractor,
        task_task_features: fastsim.RuntimeEdgeFeatureExtractor,
        task_data_features: fastsim.RuntimeEdgeFeatureExtractor,
        task_device_features: Optional[fastsim.RuntimeEdgeFeatureExtractor] = None,
        data_device_features: Optional[fastsim.RuntimeEdgeFeatureExtractor] = None,
    ):
        super().__init__(
            simulator,
            graph_spec,
            graph_extractor,
            task_features,
            data_features,
            device_features,
            task_task_features,
            task_data_features,
            task_device_features,
            data_device_features,
        )

    def data_observation(self, output: TensorDict):
        # print("Data observation")
        ntasks = output["nodes"]["tasks"]["count"][0]
        _, count = self.get_used_data(
            output["nodes"]["tasks"]["glb"][:ntasks], output["nodes"]["data"]["glb"]
        )
        output["nodes"]["data"]["count"][0] = count
        self.get_data_features(
            output["nodes"]["data"]["glb"][:count], output["nodes"]["data"]["attr"]
        )

    def get_task_device_edges(self, task_ids, workspace, global_workspace):
        length = self.graph_extractor.get_task_device_edges_mapped(
            task_ids, workspace, global_workspace
        )

        if self.truncate:
            workspace = workspace[:, :length]

        return workspace, length

    def get_data_device_edges(self, data_ids, workspace, global_workspace):
        length = self.graph_extractor.get_data_device_edges(
            data_ids, workspace, global_workspace
        )

        if self.truncate:
            workspace = workspace[:, :length]

        return workspace, length

    def new_observation_buffer(self, spec=None):
        if spec is None:
            spec = self.graph_spec

        buffer = super().new_observation_buffer(spec)
        buffer["edges"]["tasks_reads_data"] = _make_edge_tensor(
            spec.max_edges_tasks_data, self.task_data_features.feature_dim
        )
        buffer["edges"]["tasks_mapped_data"] = _make_edge_tensor(
            spec.max_edges_tasks_data, self.task_data_features.feature_dim
        )
        buffer["edges"]["tasks_write_data"] = _make_edge_tensor(
            spec.max_edges_tasks_data, self.task_data_features.feature_dim
        )

        return buffer

    def data_device_observation(self, output: TensorDict):
        # print("Data-Device observation")
        ndata = output["nodes"]["data"]["count"][0]
        ndevices = output["nodes"]["devices"]["count"][0]
        ntasks = output["nodes"]["tasks"]["count"][0]

        _, count = self.get_data_device_edges(
            output["nodes"]["data"]["glb"][:ndata],
            output["edges"]["data_devices"]["idx"],
            output["edges"]["data_devices"]["glb"],
        )

        output["edges"]["data_devices"]["count"][0] = count

        self.get_data_device_features(
            output["edges"]["data_devices"]["glb"][:, :count],
            output["edges"]["data_devices"]["attr"],
        )

    def task_data_observation(self, output: TensorDict):
        # print("Task-Data observation")
        ntasks = output["nodes"]["tasks"]["count"][0]
        ndata = output["nodes"]["data"]["count"][0]

        # Read
        _, count = self.get_task_data_edges(
            output["nodes"]["tasks"]["glb"][:ntasks],
            output["nodes"]["data"]["glb"][:ndata],
            output["edges"]["tasks_data"]["idx"],
            output["edges"]["tasks_data"]["glb"],
            AccessType.READ,
        )
        output["edges"]["tasks_data"]["count"][0] = count

        self.get_task_data_features(
            output["edges"]["tasks_data"]["glb"][:, :count],
            output["edges"]["tasks_data"]["attr"],
        )

        # Read only edges
        # _, count = self.get_task_data_edges(
        #     output["nodes"]["tasks"]["glb"][:ntasks],
        #     output["nodes"]["data"]["glb"][:ndata],
        #     output["edges"]["tasks_reads_data"]["idx"],
        #     output["edges"]["tasks_reads_data"]["glb"],
        #     AccessType.READ,
        # )
        # output["edges"]["tasks_reads_data"]["count"][0] = count

        # self.get_task_data_features(
        #     output["edges"]["tasks_reads_data"]["glb"][:, :count],
        #     output["edges"]["tasks_reads_data"]["attr"],
        # )

        # Write only edges
        # _, count = self.get_task_data_edges(
        #     output["nodes"]["tasks"]["glb"][:ntasks],
        #     output["nodes"]["data"]["glb"][:ndata],
        #     output["edges"]["tasks_write_data"]["idx"],
        #     output["edges"]["tasks_write_data"]["glb"],
        #     AccessType.WRITE,
        # )

        # output["edges"]["tasks_write_data"]["count"][0] = count

        self.get_task_data_features(
            output["edges"]["tasks_write_data"]["glb"][:, :count],
            output["edges"]["tasks_write_data"]["attr"],
        )

        # Mapped data edges
        _, count = self.get_task_data_edges(
            output["nodes"]["tasks"]["glb"][:ntasks],
            output["nodes"]["data"]["glb"][:ndata],
            output["edges"]["tasks_mapped_data"]["idx"],
            output["edges"]["tasks_mapped_data"]["glb"],
            AccessType.READ_MAPPED,
        )

        output["edges"]["tasks_mapped_data"]["count"][0] = count

        self.get_task_data_features(
            output["edges"]["tasks_mapped_data"]["glb"][:, :count],
            output["edges"]["tasks_mapped_data"]["attr"],
        )

    def get_observation(self, output=None):
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)

        # print(output)

        # # Get mappable candidates
        self.candidate_observation(output)

        # Node observations (all nodes must be processed before edges)
        self.task_observation(output, k=1)
        self.data_observation(output)
        self.device_observation(output)

        # Edge observations (edges depend on ids collected during node observation)
        self.task_task_observation(output)
        self.task_data_observation(output)
        self.task_device_observation(output, use_all_tasks=True)
        self.data_device_observation(output)

        # print("Task attribute", output["nodes"]["tasks"]["attr"])
        # print("Data attribute", output["nodes"]["data"]["attr"])
        # print("Device attribute", output["nodes"]["devices"]["attr"])

        # Auxiliary observations
        output["aux"]["time"][0] = self.simulator.time
        output["aux"]["improvement"][0] = -100
        # print("Auxiliary observation")

        output["hetero_data"] = observation_to_heterodata(output)

        # output["nodes"] = 0
        # output["edges"] = 0

        # print("All Data", output["edges"]["tasks_data"]["count"])
        # print("Read Data", output["edges"]["tasks_reads_data"]["count"])
        # print("Write Data", output["edges"]["tasks_write_data"]["count"])
        # print("Mapped Data", output["edges"]["tasks_mapped_data"]["count"])
        return output


class CandidateObserver(ExternalObserver):
    """
    Observer that only collects candidate information.
    Only 1 vector no graph information is directly collected. Useful for testing without overhead of graph extraction.
    """

    def new_observation_buffer(self, spec: Optional[fastsim.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec

        node_tensor = TensorDict(
            {"tasks": _make_node_tensor(1, self.task_features.feature_dim)}
        )

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
                "baseline": torch.ones((1), dtype=torch.float32),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": node_tensor,
                "aux": aux_tensor,
            }
        )

        return obs_tensor

    def get_observation(self, output: Optional[TensorDict] = None):
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)
            raise Warning("Allocating new observation buffer, this is not efficient!")

        # Get mappable candidates
        self.candidate_observation(output)

        output.set_(("nodes", "tasks", "glb"), output["aux", "candidates", "idx"])
        output.set_at_(("nodes", "tasks", "count"), 1, 0)

        self.get_task_features(
            output["nodes", "tasks", "glb"], output["nodes", "tasks", "attr"]
        )

        # Auxiliary observations
        output.set_at_(("aux", "progress"), -2.0, 0)
        output.set_at_(("aux", "time"), self.simulator.time, 0)
        output.set_at_(("aux", "improvement"), -100.0, 0)

        return output


class CnnSingleTaskObserver(ExternalObserver):
    """
    Observer that collects 2d flattened grid of task features.
    """

    def reset(self):
        graph = self.simulator.input.graph
        assert (
            self.graph_spec.max_candidates == 1
        ), "CnnSingleTaskObserver only supports 1 candidate"

        self.task_ids = torch.Tensor([-1 for _ in range(graph.config.n**2)])
        for task in graph.level_to_task[0]:
            self.task_ids[graph.xy_from_id(task)] = task
        if -1 in self.task_ids:
            raise ValueError(
                "Not all task ids were set during reset. Check the graph initialization."
            )
        self.prev_candidate = -1

    def new_observation_buffer(self, spec: Optional[fastsim.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec
        graph = self.simulator.input.graph

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
                "baseline": torch.ones((1), dtype=torch.float32),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": TensorDict(
                    {
                        "tasks": TensorDict(
                            {
                                "attr": torch.zeros(
                                    (graph.config.n**2, self.task_features.feature_dim),
                                    dtype=torch.float32,
                                )
                            }
                        )
                    }
                ),
                "aux": aux_tensor,
            }
        )

        return obs_tensor

    def get_observation(self, output: Optional[TensorDict] = None):
        graph = self.simulator.input.graph
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)
            raise Warning("Allocating new observation buffer, this is not efficient!")

        # Get mappable candidates
        self.candidate_observation(output)
        current_candidate = output["aux", "candidates", "idx"][0].item()
        idx = graph.xy_from_id(current_candidate)
        output.set_at_(("nodes", "tasks", "attr"), 1, (idx, -1))

        if current_candidate != self.prev_candidate and self.prev_candidate != -1:
            current_level = graph.task_to_level[self.prev_candidate]
            if current_level < (graph.config.steps - 1):
                for next_id in graph.level_to_task[current_level + 1]:
                    if (
                        graph.task_to_cell[next_id]
                        == graph.task_to_cell[self.prev_candidate]
                    ):
                        self.task_ids[graph.xy_from_id(self.prev_candidate)] = next_id
                        break
        self.prev_candidate = current_candidate

        # output.set_(("nodes", "tasks", "glb"), output["aux", "candidates", "idx"])
        # output.set_at_(("nodes", "tasks", "count"), 1, 0)
        self.get_task_features(self.task_ids, output["nodes", "tasks", "attr"])

        # Calibrate depth
        candidate_depth = output["nodes", "tasks", "attr"][idx][-2]
        output["nodes", "tasks", "attr"][:][-2] -= candidate_depth

        # for i in range(graph.config.n):
        #     for j in range(graph.config.n):
        #         idx = int(i * graph.config.n + j)
        #         print(
        #             f"{output['nodes', 'tasks', 'attr'][idx][-1].tolist()}({int(self.task_ids[idx])})",
        #             end=" ",
        #         )
        #     print()
        # print("\n\n")

        # Auxiliary observations
        output.set_at_(("aux", "progress"), -2.0, 0)
        output.set_at_(("aux", "time"), self.simulator.time, 0)
        output.set_at_(("aux", "improvement"), -100.0, 0)

        return output


class CnnBatchTaskObserver(ExternalObserver):
    """
    Observer that collects 2d flattened grid of task features.
    """

    task_ids = None

    def new_observation_buffer(self, spec: Optional[fastsim.GraphSpec] = None):
        if spec is None:
            spec = self.graph_spec
        graph = self.simulator.input.graph

        aux_tensor = TensorDict(
            {
                "candidates": _make_index_tensor(spec.max_candidates),
                "time": torch.zeros((1), dtype=torch.int64),
                "improvement": torch.zeros((1), dtype=torch.float32),
                "progress": torch.zeros((1), dtype=torch.float32),
                "baseline": torch.ones((1), dtype=torch.float32),
            }
        )

        obs_tensor = TensorDict(
            {
                "nodes": TensorDict(
                    {
                        "tasks": TensorDict(
                            {
                                "attr": torch.zeros(
                                    (graph.config.n**2, self.task_features.feature_dim),
                                    dtype=torch.float32,
                                )
                            }
                        )
                    }
                ),
                "aux": aux_tensor,
            }
        )

        return obs_tensor

    def get_observation(self, output: Optional[TensorDict] = None):
        graph = self.simulator.input.graph
        if output is None:
            output = self.new_observation_buffer(self.graph_spec)
            raise Warning("Allocating new observation buffer, this is not efficient!")
        if self.task_ids is None:
            self.task_ids = torch.Tensor([-1 for _ in range(graph.config.n**2)])

        # Get mappable candidates
        self.candidate_observation(output)
        assert (
            output["aux", "candidates", "count"][0] == graph.config.n**2
            or output["aux", "candidates", "count"][0] == 0
        ), "CnnBatchTaskObserver expects {} candidates but got {}.".format(
            graph.config.n**2, output["aux", "candidates", "count"][0].item()
        )
        for task_id in output["aux", "candidates", "idx"]:
            idx = graph.xy_from_id(task_id.item())
            self.task_ids[idx] = task_id.item()

        self.get_task_features(self.task_ids, output["nodes", "tasks", "attr"])

        # Auxiliary observations
        output.set_at_(("aux", "progress"), -2.0, 0)
        output.set_at_(("aux", "time"), self.simulator.time, 0)
        output.set_at_(("aux", "improvement"), -100.0, 0)

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
            self.observer = observer_factory.create(self)

    def get_state(self):
        return self.simulator.get_state()

    @property
    def state(self):
        return self.simulator.get_state()

    @property
    def processed_events(self):
        return self.simulator.processed_events

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

    def set_steps(self, steps: int):
        """
        Set the number of mapping steps to run the simulator.
        Will return in a breakpoint state.
        """
        self.simulator.set_steps(steps)

    def start_drain(self):
        self.simulator.start_drain()

    def stop_drain(self):
        self.simulator.stop_drain()

    def reset(self):
        """
        Return a fresh copy of the simulator driver with the same initial input and configuration.
        (This is equivalent to calling fresh_copy()).
        """
        return self.fresh_copy()

    @property
    def time(self) -> int:
        """
        Returns the current time (in microseconds) of the simulator state.
        """
        return self.simulator.get_current_time()

    @property
    def max_mem_usage(self) -> int:
        """
        Returns the maximum memory usage (in bytes) of the simulator state.
        """
        return self.simulator.get_max_memory_usage()

    def task_finish_time(self, task_id: int) -> int:
        """
        Returns the finish time (in microseconds) of a task.
        """
        return self.simulator.get_task_finish_time(task_id)

    def task_finish_time(self, task_id: int) -> int:
        """
        Returns the finish time (in microseconds) of a task.
        """
        return self.simulator.get_task_finish_time(task_id)

    def set_task_breakpoint(self, event: EventType, task_id: int) -> int:
        self.simulator.add_task_breakpoint(event, task_id)

    def clear_breakpoints(self):
        self.simulator.clear_breakpoints()

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
    max_tasks: int = 100,
    max_data: int = 100,
    max_devices: int = 5,
    max_edges_tasks_tasks: int = 200,
    max_edges_tasks_data: int = 200,
    max_edges_data_devices: int = 200,
    max_edges_tasks_devices: int = 200,
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
    spec.max_edges_data_devices = max_edges_data_devices
    spec.max_edges_tasks_devices = max_edges_tasks_devices

    spec.max_candidates = max_candidates

    # This should be max_candidates, but reverting to max_tasks to implement original NN architecture
    # spec.max_edges_tasks_devices = max_devices * max_candidates + 1
    # spec.max_edges_tasks_devices = max_devices
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

        self.input.task_noise.set_seed(duration_seed)
        self.input.task_noise.set_pseed(priority_seed)

        simulator = SimulatorDriver(
            self.input,
            observer_factory=self.observer_factory,
            internal_mapper=self.internal_mapper,
            external_mapper=self.external_mapper,
        )
        self.input.task_noise.randomize_duration(self.input.graph.static_graph)
        self.input.task_noise.randomize_priority(self.input.graph.static_graph)

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
    ):
        """
        Set the seed for the simulator.
        """
        if seed is not None:
            self.seed = seed
        if priority_seed is not None:
            self.pseed = priority_seed


def uniform_connected_devices(
    n_devices: int, mem: int, latency: int, h2d_bw: int, d2d_bw: int
) -> System:
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

    s.create_device("CPU:0", DeviceType.CPU, 2, 1000000000)
    for i in range(n_gpus):
        s.create_device(f"GPU:{i}", DeviceType.GPU, 4, mem)

    s.finalize_devices()

    for i in range(n_gpus):
        s.add_connection(0, i + 1, h2d_bw, latency)
        s.add_connection(i + 1, 0, h2d_bw, latency)

    for i in range(n_gpus):
        for j in range(n_gpus):
            if i == j:
                continue
            s.add_connection(i + 1, j + 1, d2d_bw, latency)
            s.add_connection(j + 1, i + 1, d2d_bw, latency)

    return s
