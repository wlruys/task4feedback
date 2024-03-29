from .utilities import *
from ..types import *

from rich import print


@dataclass(slots=True)
class StencilDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a sweep graph.
    """

    n_devices: int = 1
    large_size: int = 2**16
    small_size: int = 2**8
    neighbor_distance: int = 1
    dimensions: int = 1
    width: int = 10

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)

        self.initial_sizes = lambda x: (
            self.large_size if x.idx[0] == 0 else self.small_size
        )

        def edges(task_id: TaskID):
            """
            Returns the data edges for a given stencil task.

            TaskID is of the form:
            (timestep, index)
            index is n-dimensional, where n is the number of dimensions in the stencil

            DataIDs are tuples of the form:
            (timestep%2, is_boundary, boundary_idx, index)
            index is n-dimensional, where n is the number of dimensions in the stencil

            Each task reads its interior domain and its boundaries from the previous timestep
            Each task writes to its interior domain and its boundaries at the current timestep
            Each task reads from its neighbors' boundaries from the previous timestep
            """

            data_info = TaskDataInfo()

            timestep_idx = task_id.task_idx[0]
            task_idx = task_id.task_idx[1:]

            self.dimensions = len(task_idx)

            old_timestep_flag = (timestep_idx - 1) % 2
            timestep_flag = timestep_idx % 2

            # A task reads its interior domain from the previous timestep
            old_interior_id = (old_timestep_flag,) + (0,) + (0,) + task_idx
            data_info.read.append(DataAccess(DataID(old_interior_id), device=0))

            # A task (read) writes to its interior domain at the current timestep
            interior_id = (timestep_flag,) + (0,) + (0,) + task_idx
            data_info.read_write.append(DataAccess(DataID(interior_id), device=0))

            n_boundaries = 2 * (self.dimensions)

            # A task reads its boundaries from the previous timestep
            for i in range(n_boundaries):
                old_boundary_id = (old_timestep_flag,) + (1,) + (i,) + task_idx
                data_info.read.append(DataAccess(DataID(old_boundary_id), device=0))

            # A task writes to its boundaries at the current timestep
            for i in range(n_boundaries):
                boundary_id = (timestep_flag,) + (1,) + (i,) + task_idx
                data_info.read_write.append(DataAccess(DataID(boundary_id), device=0))

            # A task reads the boundaries of its neighbors from the previous timestep
            if timestep_idx > 0:
                neighbor_generator = tuple(
                    self.neighbor_distance * 2 + 1 for _ in range(self.dimensions)
                )
                stencil_generator = np.ndindex(neighbor_generator)

                neighbor_counter = 0  # Used to order the boundary accesses

                for stencil_tuple in stencil_generator:
                    stencil_tuple = np.subtract(stencil_tuple, self.neighbor_distance)
                    if np.count_nonzero(stencil_tuple) == 1:
                        dependency_grid = tuple(np.add(task_idx, stencil_tuple))
                        out_of_bounds = any(
                            element < 0 or element >= self.width
                            for element in dependency_grid
                        )
                        if not out_of_bounds:
                            old_neighbor_boundary = (
                                (old_timestep_flag,)
                                + (1,)
                                + (neighbor_counter,)
                                + dependency_grid
                            )
                            neighbor_data = DataID(old_neighbor_boundary)

                            data_info.read.append(DataAccess(neighbor_data, device=0))
                        neighbor_counter += 1

            return data_info

        self.edges = edges


stencil_gpu_task_mapping = partial(round_robin_task_mapping_gpu, index=1)


@dataclass(slots=True)
class StencilConfig(GraphConfig):
    """
    Used to configure the generation of a stencil task graph.
    """

    width: int = 10
    steps: int = 2
    neighbor_distance: int = 1
    dimensions: int = 1


@register_graph_generator
def make_stencil_graph(
    config: StencilConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)
    configurations = config.task_config

    task_dict = dict()
    data_dict = dict()

    # Build task graph

    dimensions = tuple(config.width for _ in range(config.dimensions))

    for t in range(config.steps):
        grid_generator = np.ndindex(dimensions)
        for grid_tuple in grid_generator:
            # Task ID:
            task_idx = (t,) + grid_tuple
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies
            dependency_list = []
            if t > 0:
                neighbor_generator = tuple(
                    config.neighbor_distance * 2 + 1 for _ in range(config.dimensions)
                )
                stencil_generator = np.ndindex(neighbor_generator)
                for stencil_tuple in stencil_generator:
                    # Filter to only orthogonal stencil directions (no diagonals)
                    # This is inefficient, but allows easy testing of other stencil types
                    stencil_tuple = np.subtract(stencil_tuple, config.neighbor_distance)
                    if np.count_nonzero(stencil_tuple) <= 1:
                        dependency_grid = tuple(np.add(grid_tuple, stencil_tuple))
                        out_of_bounds = any(
                            element < 0 or element >= config.width
                            for element in dependency_grid
                        )
                        if not out_of_bounds:
                            dependency = TaskID("T", (t - 1,) + dependency_grid, 0)
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
