from .utilities import *
from ..legacy_types import *

fully_connected_task_mapping_gpu = partial(round_robin_task_mapping_gpu, index=1)


@dataclass(slots=True)
class FullyConnectedConfig(GraphConfig):
    """
    Used to configure the generation of a fully connected synthetic task graph.
    """

    steps: int = 10
    width: int = 2


@register_graph_generator
def make_fully_connected_graph(
    config: FullyConnectedConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    for i in range(config.steps):
        for j in range(config.width):
            # Task ID:
            task_idx = (i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if i > 0:
                for k in range(config.width):
                    dependency = TaskID("T", (i - 1, k), 0)
                    dependency_list.append(dependency)

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_id)

            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )

    return task_dict, data_dict
