from .utilities import *
from ..types import *


@dataclass(slots=True)
class ReductionDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a reduction.
    """

    data_size: int = 32 * 1024 * 1024  # 32 MB
    n_devices: int = 1
    branch_factor: int = 2
    levels: int = 2

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)
        self.initial_sizes = lambda x: self.data_size

        def edges(task_id: TaskID):
            in_data_indices = []
            level = task_id.task_idx[0]
            j = task_id.task_idx[1]

            step = self.branch_factor ** (self.levels - level)
            start = step * j

            for k in range(1, self.branch_factor):
                in_data_indices.append(start + (step // self.branch_factor * k))

            inout_data_index = start

            data_info = TaskDataInfo()
            for i in in_data_indices:
                data_info.read.append(DataAccess(DataID((i,)), device=0))

            data_info.read_write.append(
                DataAccess(DataID((inout_data_index,)), device=0)
            )

            return data_info

        self.edges = edges


def reduction_task_mapping_gpu(
    task_id: TaskID, n_devices: int = 4, branch_factor: int = 2
) -> Devices:
    level = task_id.task_idx[0]
    j = task_id.task_idx[1]

    tasks_in_level = branch_factor ** (level)
    subtree_segment = tasks_in_level / n_devices
    device_index = int(j // subtree_segment)
    return Device(Architecture.GPU, device_id=device_index)


@dataclass(slots=True)
class ReductionConfig(GraphConfig):
    levels: int = 2
    branch_factor: int = 2


@register_graph_generator
def make_reduction_graph(
    config: ReductionConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build Task Graph
    count = 0

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
            if level < config.levels - 1:
                for k in range(config.branch_factor):
                    dependency = TaskID(
                        "T", (level + 1, config.branch_factor * j + k), 0
                    )
                    dependency_list.append(dependency)

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
