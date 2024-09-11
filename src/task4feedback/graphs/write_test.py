from .utilities import *
from ..types import *


@dataclass(slots=True)
class TestChainDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern where data in the same dependent chain is shared between tasks.
    """

    data_size: int = 32 * 1024 * 1024  # 32 MB
    chain_index: int = 1
    n_devices: int = 1

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)
        self.initial_sizes = lambda x: self.data_size

        def edges(task_id: TaskID):
            data_info = TaskDataInfo()
            data_idx = int(list(task_id.task_idx)[0])

            # print(task_id, data_idx)

            data_info.write.append(DataAccess(DataID(data_idx), device=0))

            if data_idx > 0:
                data_info.read.append(DataAccess(DataID(data_idx - 1), device=0))

            return data_info

        self.edges = edges


@dataclass(slots=True)
class TestChainConfig(GraphConfig):
    """
    Used to configure the generation of a serial synthetic task graph.

    @field steps: The number of steps in the graph
    @field dependency_count: The number of dependencies per task
    @field chains: The number of chains to generate that can run in parallel

    Example Graph (steps=3, dependency_count=1, chains=4):
    --T(0,1)--T(0, 2)--T(0, 3)-->
    --T(1,1)--T(1, 2)--T(1, 3)-->
    --T(2,1)--T(2, 2)--T(2, 3)-->
    --T(3,1)--T(3, 2)--T(3, 3)-->

    """

    steps: int = 1
    dependency_count: int = 1
    chains: int = 1


@register_graph_generator
def make_chain_graph(
    config: TestChainConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph
    for i in range(config.steps):
        for j in range(config.chains):
            # Task ID:
            task_idx = (i, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Runtime Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            dependency_limit = min(i, config.dependency_count)
            for k in range(1, dependency_limit + 1):
                assert i - k >= 0
                dependency = TaskID("T", (i - k, j), 0)
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
