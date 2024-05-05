#Blocked GEMM + Cannon GEMM
import math
from .utilities import *
from ..types import *

@dataclass(slots=True)
class ImecDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a reduction.
    """
    hier_levels: int = 1
    levels: int = 3
    blocks: int = 4
    energy: list = field(default_factory=list)
    n: int = 8
    a: int = 8
    p: int = 4 # number of processors

    def initial_data_size(self, data_id: DataID) -> Devices:
        n_on_each_proc = (self.n * self.n) / self.p
        return n_on_each_proc * self.a
    
    def initial_data_placement(self, data_id: DataID) -> Devices:
        return Device(Architecture.GPU, 0, self.energy[data_id.idx[0][0]]) # everyting is on HBM at the start

    def __post_init__(self):
        self.initial_placement = self.initial_data_placement
        self.initial_sizes = self.initial_data_size

        def edges(task_id: TaskID):
            #print("Task ID: ", task_id.task_idx)
            in_data_indices = []
            irow = task_id.task_idx[0]
            jcol = task_id.task_idx[1]
            k = task_id.task_idx[2]
            level = task_id.task_idx[3]
            j = task_id.task_idx[4]
            step = int(math.sqrt(self.blocks))
            # mod = 2 * self.blocks - 1
            # print("TASK ID:", task_id.task_idx)
            start_row = (j // step) * step
            start_col = j % step
            shift = (self.levels - 1) - level
            mod_a = start_row + step
            mod_b = self.blocks
            # step = self.branch_factor ** (self.levels - level)
            # start = step * j
            if(level == 0):
                data_info = TaskDataInfo()
                return data_info

            elif(level == self.levels - 1): # read data at the topmost level
                # in_data_indices.append((irow, jcol, k))
                in_data_indices.append((irow, jcol, k, step + 1, 2 * j)) # read A block
                in_data_indices.append((irow, jcol, k, step + 1, 2 * j + 1)) # read B block
                if(k == 0): # Read C every i, j itr
                    in_data_indices.append((irow, jcol, k, step + 1, 2 * self.blocks + j)) # read C block
                print("IF in_data_indices: ", in_data_indices, task_id.task_idx)
            elif(j == mod_a - 1):
                in_data_indices.append((irow, jcol, k, step + 1, (2 * ((start_col + shift) % step + start_row)))) # read A block
                # in_data_indices.append((step + 1, (2 * ((j + shift) % mod_a + start_row)))) # read A block
                in_data_indices.append((irow, jcol, k, step + 1, (2 * ((j + step * shift) % mod_b) + 1))) # read B block
                # in_data_indices.append((step + 1, ((2 * ((j + step * shift) + 1) % mod)))) # read B block
                #print("ELIF in_data_indices: ", in_data_indices, task_id.task_idx)
            else:
                in_data_indices.append((irow, k, step + 1, (2 * ((start_col + shift) % step + start_row)))) # read A block
                # in_data_indices.append((step + 1, (2 * ((j + shift) % mod_a)))) # read A block
                in_data_indices.append((k, jcol, step + 1, (2 * ((j + shift * step) % mod_b) + 1))) # read B block

            #print(in_data_indices)
            
            data_info = TaskDataInfo()
            for i in in_data_indices:
                #print(i)
                data_info.read_write.append(DataAccess(DataID((i,)), device=0))
            #print(out_data_index)
            # for i in out_data_index:
            #     print(task_id.task_idx, " ", out_data_index)
            #     data_info.write.append(DataAccess(DataID((i,)), device=0))

            return data_info

        self.edges = edges


def imec_task_mapping_gpu(
    task_id: TaskID, n_devices: int = 4, branch_factor: int = 2
) -> Devices:
    level = task_id.task_idx[0]
    j = task_id.task_idx[1]

    tasks_in_level = branch_factor ** (level)
    subtree_segment = tasks_in_level / n_devices
    device_index = int(j // subtree_segment)
    return Device(Architecture.GPU, device_id=device_index)

@dataclass(slots=True)
class ImecConfig(GraphConfig):
    levels: int = 3
    blocks: int = 4
    B: int = 2


@register_graph_generator
def make_imec_graph(
    config: ImecConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build Task Graph
    count = 0
    # levels = math.sqrt(config.blocks) + 1
    prev = []
    for irow in range(config.B): # blocked GEMM
        for jcol in range(config.B):
            # prev = []
            for k in range(config.B):
                for level in range(config.levels - 1, -1, -1): #levels are going to be sq_root of # blocks + 1
                    tasks_in_level = config.blocks
                    subtree_segment = tasks_in_level / config.n_devices

                    for j in range(tasks_in_level):
                        # Task ID:
                        task_idx = (irow, jcol, k, level, j)
                        task_id = TaskID("T", task_idx, 0)
                        # print("Task ID:", task_id.idx)
                        # Task Placement Info
                        task_placement_info = configurations(task_id)

                        # Task Dependencies
                        dependency_list = []
                        if level == 0: # addition depends on all the prior multiplication
                            for l in range(config.levels - 1):
                                # print((level + k + 1, j))
                                dependency = TaskID(
                                    "T", (irow, jcol, k, level + l + 1, j), 0
                                )
                                # print(dependency)
                                dependency_list.append(dependency)
                            # prev.append(task_id) # add completion tasks of this cannon gemm as dependency of next cannon gemm
                                # print(dependency_list)

                        elif level < config.levels - 1: # all multiplications except 1 can take place only after all prior level tasks are finished
                            for dep in range(tasks_in_level):
                                dependency = TaskID(
                                    "T", (irow, jcol, k, level + 1, dep), 0
                                )
                                dependency_list.append(dependency)

                        elif level == config.levels - 1: #next block of Cannon Gemm depends on completion of previous block of cannon gemm
                            '''
                            Example: 
                            0 0 0 
                            0 0 1
                            0 1 0 
                            0 1 1
                            1 0 0
                            1 0 1
                            1 1 0
                            1 1 1 
                            Each row after the second row depends on the previous block cannon gemm
                            '''
                            i_idx = irow
                            j_idx = jcol
                            k_idx = k
                            if(k > 0):
                                k_idx = k - 1
                            elif(jcol > 0):
                                j_idx = jcol - 1
                                k_idx = config.B - 1
                            elif(irow > 0):
                                i_idx = irow - 1
                                j_idx = config.B - 1
                                k_idx = config.B - 1
                            if(irow > 0 or jcol > 0 or k > 0): # First block has no dependency
                                for dep in range(tasks_in_level):
                                    dependency = TaskID(
                                        "T", (i_idx, j_idx, k_idx, 0, dep), 0
                                    )
                                    dependency_list.append(dependency)
                        # elif level == config.levels - 1 and len(prev) != 0: #next Cannon Gemm depends on completion of previous cannon gemm
                        #     for l in prev:
                        #         dependency_list.append(l)
                            # prev = []

                        #for k in range(config.branch_factor):
                        # print("TASK: ", task_id.task_idx, "DEP: ", dependency_list)
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
