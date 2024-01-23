from .utilities import *
from ..types import *


@dataclass(slots=True)
class ReusedDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern where the data is reused across tasks.
    Only READ operations.
    """

    data_size: int = 32 * 1024 * 1024  # 32 MB
    n_partitions: int = 1
    n_reads: int = 1
    _internal_count: int = 0

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)
        self.initial_sizes = lambda x: self.data_size

        def edges(task_id: TaskID):
            data_info = TaskDataInfo()
            for i in range(self.n_reads):
                data_info.read.append(
                    DataAccess(
                        DataID(self._internal_count % self.n_partitions), device=0
                    )
                )
                self._internal_count += 1
            return data_info

        self.edges = edges


@dataclass(slots=True)
class IndependentDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern where the data is independent across tasks.
    Only READ operations.
    """

    data_size: int = 32 * 1024 * 1024  # 32 MB
    n_devices: int = 1
    n_reads: int = 1
    _internal_count: int = 0

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)
        self.initial_sizes = lambda x: self.data_size

        def edges(task_id: TaskID):
            data_info = TaskDataInfo()
            for i in range(self.n_reads):
                data_info.read.append(
                    DataAccess(DataID(self._internal_count), device=0)
                )
                self._internal_count += 1
            return data_info

        self.edges = edges


def independent_task_mapping_cpu(task_idx: TaskID) -> Devices:
    return Device(Architecture.CPU, 0)


independent_task_mapping_gpu = partial(round_robin_task_mapping_gpu, index=0)


@dataclass(slots=True)
class IndependentConfig(GraphConfig):
    """
    Used to configure the generation of an independent synthetic task graph.

    @field task_count: The number of tasks in the graph
    """

    task_count: int = 1


@register_graph_generator
def make_independent_graph(
    config: IndependentConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph
    for i in range(config.task_count):
        # Task ID
        task_idx = (i,)
        task_id = TaskID("T", task_idx, 0)

        # Task Placement Info
        task_placement_info = configurations(task_id)

        # Task Dependencies
        task_dependencies = []

        # Task Data Dependencies
        data_dependencies, data_dict = get_data_dependencies(
            task_id, data_dict, data_config
        )

        # Task Mapping
        task_mapping = get_mapping(config, task_id)

        task_dict[task_id] = TaskInfo(
            task_id,
            task_placement_info,
            task_dependencies,
            data_dependencies,
            task_mapping,
        )

    return task_dict, data_dict
