from .utilities import *
from ..types import *


@dataclass(slots=True)
class SweepDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a sweep graph.
    """

    n_devices: int = 1
    large_size: int = 2**16
    small_size: int = 2**8

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)

        self.initial_sizes = (
            lambda x: self.large_size if x.idx[0] == 0 else self.small_size
        )

        def edges(task_id: TaskID):
            data_info = TaskDataInfo()

            # A task reads/writes to an interior domain (unique for each task)
            data_info.read_write.append(
                DataAccess(DataID((0,) + task_id.task_idx), device=0)
            )

            # A task reads/writes to a small communication buffer (unique for each task)
            data_info.read_write.append(
                DataAccess(DataID((1,) + task_id.task_idx), device=0)
            )

            # A task reads the small communication buffers from its neighbors (dependencies)
            for i in range(len(task_id.task_idx)):
                if task_id.task_idx[i] > 0:
                    source_idx = list(task_id.task_idx)
                    source_idx[i] -= 1
                    data_info.read.append(
                        DataAccess(DataID((1,) + tuple(source_idx)), device=0)
                    )

            return data_info

        self.edges = edges


sweep_task_mapping_gpu = partial(round_robin_task_mapping_gpu, index=1)


@dataclass(slots=True)
class SweepConfig(GraphConfig):
    """
    Used to configure the generation of a sweep synthetic task graph.
    """

    width: int = 1
    dimensions: int = 1
    steps: int = 1


@register_graph_generator
def make_sweep_graph(
    config: SweepConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    from rich import print

    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    shape = tuple(config.width for _ in range(config.dimensions))

    for i in range(config.steps):
        grid_generator = np.ndindex(shape)

        for grid_tuple in grid_generator:
            # Task ID:
            task_idx = (i,) + grid_tuple
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            for j in range(config.dimensions + 1):
                if task_idx[j] > 0:
                    dependency_grid = list(task_idx)
                    dependency_grid[j] -= 1
                    dependency = TaskID("T", tuple(dependency_grid), 0)
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
