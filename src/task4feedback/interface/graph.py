from typing import Optional, List
from rich import print
import task4feedback.trip as trip
from task4feedback.trip import (
    Graph,
    StaticTaskInfo,
    Data,
    DeviceType,
)
from .types import (
    TaskTuple,
    DataBlockTuple,
    _bytes_to_readable,
)
from .lambdas import VariantBuilder, DataBlockTransformer

class TaskGraph:
    def __init__(self):
        self.graph = Graph()
        self.static_graph = None
        self.tasks: dict[int, TaskTuple] = {}
        self.is_finalized = False

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
        self.graph.clear_all_variants()
        for i in range(self.graph.get_n_compute_tasks()):
            task = self.get_task(i)
            for arch in DeviceType:
                variant = variant_builder.build_variant(arch, task)

                if variant is None:
                    continue

                vcu_usage = int(variant.vcu_usage * trip.MAX_VCUS)
                self.graph.set_variant(
                    i,
                    arch,
                    vcu_usage,
                    variant.memory_usage,
                    variant.expected_time,
                )

    def finalize(self):
        self.is_finalized = True
        self.graph.finalize()
        self.static_graph = StaticTaskInfo(self.graph)

        for task_id, v in self.tasks.items():
            v.dependencies = self.get_task_dependencies(task_id)

    def print_variants(self):
        for i in range(self.graph.get_n_compute_tasks()):
            task = self.get_task(i)
            print(f"Task {task.id}: {task.name} (Tag: {task.tag})")
            # Note: compute_task was not defined in original code, assuming it meant accessing graph variants
            # This part of original code seemed broken or relied on globals. 
            # I will comment it out or fix it if I can infer intent.
            # print("  Variants:")
            # print(compute_task.get_variants()) 
            print(f"Dependencies: {task.dependencies}")
            print(f"Read Data: {task.read}")
            print(f"Write Data: {task.write}") 

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

            result.append(f"  Task {task.id}: {task.name} (Tag: {task.tag})")
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
        return [self.data.get_id(block) if isinstance(block, str) else block for block in blocklist]

    def convert_ids_to_names(self, blocklist):
        return [self.data.get_name(block) if isinstance(block, int) else block for block in blocklist]

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

    def __str__(self):
        result = []

        block_count = self.data.size()
        result.append(f"DataWrapper with {block_count} blocks:")

        for i in range(block_count):
            block = self.get_block(i)
            result.append(f"  Block {block.id}: {block.name} (Size: {_bytes_to_readable(block.size)}, Location: {block.location}, Tag: {block.tag}, Type: {block.type})")

        return "\n".join(result)
