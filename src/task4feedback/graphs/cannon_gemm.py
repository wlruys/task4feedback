import math
from .utilities import *
from ..types import *


@dataclass(slots=True)
class CannonGemmDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a reduction.
    """

    data_size: int = 256 * 16  # 4 * (n**2/p) n=2**8 and p=64
    n_devices: int = 4
    levels: int = 3
    blocks: int = 4

    def initial_data_placement(self, data_id: DataID) -> Devices:
        # return Device(Architecture.CPU, 0)
        step = math.sqrt(n_gpus)
        if data_id.idx[0][0] == 0:
            return Device(Architecture.GPU, data_id.idx[0][1])
        return Device(Architecture.GPU, int(data_id.idx[0][1] // step))

    def __post_init__(self):
        self.initial_placement = self.initial_data_placement
        self.initial_sizes = lambda x: self.data_size

        def edges(task_id: TaskID):
            in_data_indices = []
            level = task_id.task_idx[0]
            j = task_id.task_idx[1]
            step = int(math.sqrt(self.blocks))
            # mod = 2 * self.blocks - 1
            start_row = (j // step) * step
            start_col = j % step
            shift = (self.levels - 1) - level
            mod_a = start_row + step
            mod_b = self.blocks
            # step = self.branch_factor ** (self.levels - level)
            # start = step * j
            if level == 0:
                data_info = TaskDataInfo()
                return data_info

            if level == self.levels - 1:  # read data at the topmost level
                in_data_indices.append((step + 1, 2 * j))  # read A block
                in_data_indices.append((step + 1, 2 * j + 1))  # read B block
                # print("in_data_indices: ", in_data_indices)
            elif j == mod_a - 1:
                in_data_indices.append(
                    (step + 1, (2 * ((j + shift) % mod_a + start_row)))
                )  # read A block
                in_data_indices.append(
                    (step + 1, (2 * ((j + step * shift) % mod_b) + 1))
                )  # read A block
                # in_data_indices.append((step + 1, ((2 * ((j + step * shift) + 1) % mod)))) # read B block
                # print("in_data_indices: ", in_data_indices, task_id.task_idx)
            else:
                in_data_indices.append(
                    (step + 1, (2 * ((j + shift) % mod_a)))
                )  # read A block
                in_data_indices.append(
                    (step + 1, (2 * ((j + shift * step) % mod_b) + 1))
                )  # read A block
                # in_data_indices.append((step + 1, ((2 * ((j + step * shift) + 1) % mod)))) # read B block
                # print("in_data_indices: ", in_data_indices, task_id.task_idx)
            out_data_index = (0, j)  # always write to addition

            # inout_data_index = start

            data_info = TaskDataInfo()
            for i in in_data_indices:
                data_info.read_write.append(DataAccess(DataID((i,)), device=0))

            data_info.write.append(DataAccess(DataID((out_data_index,)), device=0))

            return data_info

        self.edges = edges


def cannon_gemm_task_mapping_gpu(
    task_id: TaskID, n_devices: int = 4, branch_factor: int = 2
) -> Devices:
    level = task_id.task_idx[0]
    j = task_id.task_idx[1]

    tasks_in_level = branch_factor ** (level)
    subtree_segment = tasks_in_level / n_devices
    device_index = int(j // subtree_segment)
    return Device(Architecture.GPU, device_id=device_index)


@dataclass(slots=True)
class CannonGemmConfig(GraphConfig):
    levels: int = 3
    blocks: int = 4
    # branch_factor: int = 2


@register_graph_generator
def make_cannon_gemm_graph(
    config: CannonGemmConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build Task Graph
    count = 0
    # levels = math.sqrt(config.blocks) + 1
    for level in range(
        config.levels - 1, -1, -1
    ):  # levels are going to be sq_root of # blocks + 1
        tasks_in_level = config.blocks
        subtree_segment = tasks_in_level / config.n_devices

        for j in range(tasks_in_level):
            # Task ID:
            task_idx = (level, j)
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if level == 0:  # addition depends on all the prior multiplication
                for k in range(config.levels - 1):
                    # print((level + k + 1, j))
                    dependency = TaskID("T", (level + k + 1, j), 0)
                    # print(dependency)
                    dependency_list.append(dependency)
                # print(dependency_list)

            elif (
                level < config.levels - 1
            ):  # all multiplications except 1 can take place only after all prior level tasks are finished
                for dep in range(tasks_in_level):
                    dependency = TaskID("T", (level + 1, dep), 0)
                    dependency_list.append(dependency)
                # for k in range(config.branch_factor):

            # print("level: ", (level,j))
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
