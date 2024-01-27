from .utilities import *
from ..types import *


@dataclass(slots=True)
class ScatterReductionConfig(GraphConfig):
    """
    Used to configure the generation of a scatter-reduction task graph.
    """

    levels: int = 4
    branch_factor: int = 2


@register_graph_generator
def make_scatter_reduction_graph(config: ScatterReductionConfig):
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build Task Graph

    # Scatter phase
    for level in range(config.levels + 1):
        tasks_in_level = config.branch_factor ** (level)

        subtree_segment = tasks_in_level // config.n_devices

        for j in range(tasks_in_level):
            # Task ID:
            task_idx = (2 * config.levels - level, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if level > 0:
                dependency = TaskID(
                    "T",
                    (2 * config.levels - level + 1, j // config.branch_factor),
                    0,
                )
                dependency_list.append(dependency)

            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    # Reduction phase
    for level in range(config.levels - 1, -1, -1):
        tasks_in_level = config.branch_factor ** (level)
        subtree_segment = tasks_in_level / config.n_devices

        for j in range(tasks_in_level):
            # Task ID:
            task_idx = (level, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            for k in range(config.branch_factor):
                dependency = TaskID("T", (level + 1, config.branch_factor * j + k), 0)
                dependency_list.append(dependency)

            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_idx)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict
