from .utilities import *
from ..types import *


@dataclass(slots=True)
class ButterflyConfig(GraphConfig):
    """
    Used to configure the generation of a butterfly synthetic task graph. (FFT Pattern)
    """

    width: int = 9
    steps: int = 4


@register_graph_generator
def make_butterfly_graph(
    config: ButterflyConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    assert config.steps <= np.log2(config.width) + 1

    # Build task graph
    for i in range(config.steps + 1):
        for j in range(config.width):
            # Task ID:
            task_idx = (i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if i > 0:
                dependency = TaskID("T", (i - 1, j), 0)
                dependency_list.append(dependency)

                step = 2 ** (config.steps - i)

                left_idx = j - step
                if left_idx >= 0 and left_idx < config.width:
                    dependency = TaskID("T", (i - 1, left_idx), 0)
                    dependency_list.append(dependency)

                right_idx = j + step
                if right_idx >= 0 and right_idx < config.width:
                    dependency = TaskID("T", (i - 1, right_idx), 0)
                    dependency_list.append(dependency)

            # Task Data Dependencies
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
