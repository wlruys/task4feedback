from .utilities import *
from ..types import *


@dataclass(slots=True)
class CholeskyDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a Block Cholesky factorization.
    """

    data_size: int = 1024 * 1024 * 1024  # 1 GB

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)
        self.initial_sizes = lambda x: self.data_size

        def edges(task_id: TaskID):
            if task_id.taskspace == "POTRF":
                j = task_id.task_idx[0]
                data_info = TaskDataInfo()
                data_info.read_write.append(DataAccess(DataID((j, j)), device=0))
            elif task_id.taskspace == "SYRK":
                j, k = task_id.task_idx
                data_info = TaskDataInfo()
                data_info.read.append(DataAccess(DataID((j, k)), device=0))
                data_info.read_write.append(DataAccess(DataID((j, j)), device=0))
            elif task_id.taskspace == "SOLVE":
                i, j = task_id.task_idx
                data_info = TaskDataInfo()
                data_info.read.append(DataAccess(DataID((j, j)), device=0))
                data_info.read_write.append(DataAccess(DataID((i, j)), device=0))
            elif task_id.taskspace == "GEMM":
                i, j, k = task_id.task_idx
                data_info = TaskDataInfo()
                data_info.read.append(DataAccess(DataID((i, k)), device=0))
                data_info.read.append(DataAccess(DataID((j, k)), device=0))
                data_info.read_write.append(DataAccess(DataID((i, j)), device=0))
            else:
                raise ValueError(
                    f"Unknown task type. Cannot generate data graph for {task_id}"
                )

            return data_info

        self.edges = edges


cholesky_task_mapping_gpu = partial(round_robin_task_mapping_gpu, index=0)


@dataclass(slots=True)
class CholeskyConfig(GraphConfig):
    """
    Used to configure the generation of a synthetic Cholesky task graph
    """

    blocks: int = 4


@register_graph_generator
def make_cholesky_graph(
    config: CholeskyConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    for j in range(config.blocks):
        for k in range(j):
            # Inter-block GEMM (update diagonal block)
            syrk_task_id = TaskID("SYRK", (j, k), 0)
            syrk_placement_info = configurations(syrk_task_id)
            dependency_list = [TaskID("SOLVE", (j, k), 0)] + [
                TaskID("SYRK", (j, l), 0) for l in range(k)
            ]
            data_dependencies, data_dict = get_data_dependencies(
                syrk_task_id, data_dict, data_config
            )
            task_mapping = get_mapping(config, syrk_task_id)
            task_dict[syrk_task_id] = TaskInfo(
                syrk_task_id,
                syrk_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
                func_id=config.func_id(syrk_task_id),
            )

        # Diagonal block Cholesky
        potrf_task_id = TaskID("POTRF", (j,), 0)
        potrf_placement_info = configurations(potrf_task_id)
        dependency_list = [TaskID("SYRK", (j, l), 0) for l in range(j)]
        data_dependencies, data_dict = get_data_dependencies(
            potrf_task_id, data_dict, data_config
        )
        task_mapping = get_mapping(config, potrf_task_id)
        task_dict[potrf_task_id] = TaskInfo(
            potrf_task_id,
            potrf_placement_info,
            dependency_list,
            data_dependencies,
            task_mapping,
            func_id=config.func_id(potrf_task_id),
        )

        for i in range(j + 1, config.blocks):
            for k in range(j):
                # Inter-block GEMM (update off-diagonal block)
                gemm_task_id = TaskID("GEMM", (i, j, k), 0)
                gemm_placement_info = configurations(gemm_task_id)
                dependency_list = [
                    TaskID("SOLVE", (i, k), 0),
                    TaskID("SOLVE", (j, k), 0),
                ] + [TaskID("GEMM", (i, j, l), 0) for l in range(k)]
                data_dependencies, data_dict = get_data_dependencies(
                    gemm_task_id, data_dict, data_config
                )
                task_mapping = get_mapping(config, gemm_task_id)
                task_dict[gemm_task_id] = TaskInfo(
                    gemm_task_id,
                    gemm_placement_info,
                    dependency_list,
                    data_dependencies,
                    task_mapping,
                    func_id=config.func_id(gemm_task_id),
                )

            # Panel solve
            solve_task_id = TaskID("SOLVE", (i, j), 0)
            solve_placement_info = configurations(solve_task_id)
            dependency_list = [TaskID("POTRF", (j,), 0)] + [
                TaskID("GEMM", (i, j, l), 0) for l in range(j)
            ]
            data_dependencies, data_dict = get_data_dependencies(
                solve_task_id, data_dict, data_config
            )
            task_mapping = get_mapping(config, solve_task_id)
            task_dict[solve_task_id] = TaskInfo(
                solve_task_id,
                solve_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
                func_id=config.func_id(solve_task_id),
            )

    return task_dict, data_dict
