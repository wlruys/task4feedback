from random import Random
from tabnanny import check

from numpy import average
from pyparsing import C
from sympy import limit

from task4feedback.simulator import data
from .utilities import *
from ..types import *


@dataclass(slots=True)
class CustomConfig(GraphConfig):
    """
    Used to configure the generation of a random task graph.
    """

    # Function to determine task placement information
    # These are here specifically for random graph generation
    p2p_bandwidth: int = 10 * 1024 * 1024 * 1024  # 10 GB/s
    initial_placement: Callable[[DataID], Devices] = default_data_initial_placement
    task_dict = dict()
    data_dict = dict()

    def add_task(self, _task_id: int):
        task_id = TaskID("T", (_task_id,), 0)
        self.task_dict[task_id] = TaskInfo(
            task_id,
            self.task_config(task_id),
            [],
            TaskDataInfo(),
            get_mapping(self, task_id),
        )

    def add_dependency(self, _task_id: int, _dependency_id: int):
        task_id = TaskID("T", (_task_id,), 0)
        dependency_id = TaskID("T", (_dependency_id,), 0)
        self.task_dict[task_id].dependencies.append(dependency_id)


@register_graph_generator
def make_custom_graph(
    config: CustomConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)

    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()
    tasks_by_level = dict()

    random = Random(config.seed)

    # Level 0 starts with only one node
    task_id = TaskID("T", (0, 0), 0)
    task_placement_info = configurations(task_id)
    if config.no_data:
        data_dependencies, data_dict = get_data_dependencies(
            task_id, data_dict, data_config
        )
        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            [],
            data_dependencies,
            get_mapping(config, task_id),
        )
    else:
        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            [],
            TaskDataInfo(),
            get_mapping(config, task_id),
        )
    tasks_by_level[0] = [task_id]

    # Generate subsequent levels
    nodes = 1
    total_level = 1
    while nodes != config.nodes:
        # Determine the number of tasks for this level
        if config.nodes - nodes < config.max_width:
            width = config.nodes - nodes
        else:
            width = random.randint(1, min(config.max_width, config.nodes - nodes))

        task_list = []

        for j in range(width):
            task_idx = (total_level, j)
            task_id = TaskID("T", task_idx, 0)
            task_list.append(task_id)

            task_placement_info = configurations(task_id)
            dependency_list = []

            # Connect each task in level K to a random task in level K-1
            dep_task_idx = random.randint(0, len(tasks_by_level[total_level - 1]) - 1)
            dependency = tasks_by_level[total_level - 1][dep_task_idx]
            dependency_list.append(dependency)

            task_mapping = get_mapping(config, task_id)

            if config.no_data:
                data_dependencies, data_dict = get_data_dependencies(
                    task_id, data_dict, data_config
                )

                task_dict[task_id] = TaskInfo(
                    task_id,
                    task_placement_info,
                    dependency_list,
                    data_dependencies,
                    task_mapping,
                )
            else:
                task_dict[task_id] = TaskInfo(
                    task_id,
                    task_placement_info,
                    dependency_list,
                    TaskDataInfo(),
                    task_mapping,
                )
        # Store tasks of this level
        tasks_by_level[total_level] = task_list
        nodes += width
        total_level += 1
    print(f"Total levels: {total_level}")
    # Add random edges
    all_tasks = [task for level in tasks_by_level.values() for task in level]
    possible_tasks = [task for task in all_tasks if task.task_idx[0] < total_level - 1]
    success = 0
    num_random_edges = int(
        config.density * config.nodes * (config.nodes - 1) / 2 - config.nodes + 1
    )
    skipped_levels = []
    while success < num_random_edges:
        task1 = random.choice(possible_tasks)
        task1_level = task1.task_idx[0]

        # Select a task from a higher level
        higher_level_tasks = [
            task for task in all_tasks if task.task_idx[0] > task1_level
        ]
        if higher_level_tasks:
            task2 = random.choice(higher_level_tasks)
            if task1 not in task_dict[task2].dependencies:
                task_dict[task2].dependencies.append(task1)
                success += 1
                skipped_levels.append(task2.task_idx[0] - task1_level)
    if config.density > 0:
        print(f"Avg. skipped levels: {average(skipped_levels)}")
    else:
        print(f"Avg. skipped levels: {0}")
    if config.no_data is False:
        for task in task_dict.values():
            data_dependencies = TaskDataInfo()
            data_id = DataID((task.id.task_idx))
            limit_check = 0
            # Add data dependencies
            # This does not add data. Data id is already added by a task that performs R/W
            for dep_task in task.dependencies:
                dep_data_id = DataID((dep_task.task_idx))
                data_dependencies.read.append(DataAccess(dep_data_id, device=0))
                if dep_data_id not in data_dict:
                    data_placement = config.initial_placement(dep_data_id)
                    data_size = config.initial_sizes(dep_data_id)
                    data_dict[dep_data_id] = DataInfo(
                        dep_data_id, data_size, data_placement
                    )
                limit_check += data_dict[dep_data_id].size

            # Add data read/write access
            data_dependencies.read_write.append(DataAccess(data_id, device=0))
            if data_id not in data_dict:
                data_placement = config.initial_placement(data_id)
                data_size = config.initial_sizes(data_id)
                data_dict[data_id] = DataInfo(data_id, data_size, data_placement)
            limit_check += data_dict[data_id].size

            # Check if the data size is within the GPU size limit
            if limit_check > config.gpu_size_limit:
                raise ValueError(
                    f"Data size {data_size} exceeds the GPU size limit {config.gpu_size_limit}"
                )

            task.data_dependencies = data_dependencies

    return task_dict, data_dict
