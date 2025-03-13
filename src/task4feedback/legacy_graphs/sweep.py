from .utilities import *
from ..legacy_types import *


def in_domain(steps, width, grid):
    if grid[0] >= steps or grid[0] < 0:
        return False
    for g in grid[1:]:
        if g >= width or g < 0:
            return False
    return True


@dataclass(slots=True)
class SimpleSweepDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a simple sweep graph.

    Each task in a graph reads and writes to a data item from each row/column of the previous step.
    """

    n_devices: int = 1
    size: int = 2**8

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)

        self.initial_sizes = lambda x: self.size

        def edges(task_id: TaskID):
            data_info = TaskDataInfo()

            task_idx = task_id.task_idx

            for i, idx in enumerate(task_idx):
                data_info.read_write.append(DataAccess(DataID((i, idx)), device=0))

            return data_info

        self.edges = edges


@dataclass(slots=True)
class SweepDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a sweep graph.
    """

    n_devices: int = 1
    large_size: int = 2**16
    small_size: int = 2**8
    direction_list: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)

        self.initial_sizes = lambda x: (
            self.large_size if x.idx[0] == 0 else self.small_size
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


@dataclass(slots=True)
class MultiDirectionSweepDataGraphConfig(DataGraphConfig):
    """
    Defines a data graph pattern for a sweep graph.

    TaskID: (direction; grid)
    DataID: interior: (0; direction; cell)
            boundary: (1; direction; which_dim; which_side; cell)
            material: (2; cell)

    is_boundary, direction; grid)

    """

    n_devices: int = 1
    large_size: int = 2**16
    small_size: int = 2**8
    steps: int = 5
    width: int = 5
    direction_list: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.initial_placement = lambda x: Device(Architecture.CPU, 0)

        self.initial_sizes = lambda x: (
            self.large_size if x.idx[0] == 0 else self.small_size
        )

        def edges(task_id: TaskID):
            data_info = TaskDataInfo()

            task_idx = task_id.task_idx
            # print("Task IDX", task_idx)
            iteration_num = task_idx[0]
            task_type = task_idx[1]

            if task_type == 0:
                direction_idx = task_idx[2]
                grid_idx = task_idx[3:]
                n_dimensions = len(grid_idx)

                # A sweep task reads/writes to an interior domain (unique for each cell and direction)
                data_info.read_write.append(
                    DataAccess(DataID((0,) + (direction_idx,) + grid_idx), device=0)
                )

                # A sweep task reads a material properties array for its cell
                # Unique for each cell
                data_info.read.append(DataAccess(DataID((2,) + grid_idx), device=0))

                # A sweep task reads/writes to a small communication buffer
                # unique for each cell and direction
                for dim in range(n_dimensions):
                    # top / bottom
                    for side in [-1, 1]:
                        boundary_data_idx = (
                            (1,) + (direction_idx,) + (dim,) + (side,) + grid_idx
                        )
                        data_info.read_write.append(
                            DataAccess(DataID(boundary_data_idx), device=0)
                        )

                direction_flags = f"{direction_idx:08b}"

                # A sweep task reads the boundary buffers from their neighbors
                for dim in range(n_dimensions):
                    if task_id.task_idx[dim] > 0:
                        source_idx = list(grid_idx)
                        step = (-1) ** int(direction_flags[-(dim + 1)])
                        source_idx[dim] -= step

                        boundary_data_idx = (
                            (1,)
                            + (direction_idx,)
                            + (dim,)
                            + (step,)
                            + tuple(source_idx)
                        )

                        if in_domain(self.steps, self.width, source_idx):
                            data_info.read.append(
                                DataAccess(
                                    boundary_data_idx,
                                    device=0,
                                )
                            )
            elif task_type == 1:
                # An update task reads all boundaries and interiors of a cell

                direction_idx = task_idx[1]
                grid_idx = task_idx[2:]
                n_dimensions = len(grid_idx)

                for dir in self.direction_list:
                    interior_data_idx = (0,) + (dir,) + grid_idx
                    data_info.read.append(DataAccess(interior_data_idx, device=0))

                    for dim in range(n_dimensions):
                        for side in [-1, 1]:
                            boundary_data_idx = (
                                (1,) + (dir,) + (dim,) + (side,) + grid_idx
                            )
                            data_info.read.append(
                                DataAccess(boundary_data_idx, device=0)
                            )

                material_data_idx = (2,) + grid_idx
                data_info.read_write.append(DataAccess(material_data_idx, device=0))

            else:
                print("ERROR")

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
    max_iter: int = 1
    direction_list: List[int] = field(default_factory=list)


def create_sweep_iteration(
    iter: int, config: SweepConfig, data_config: DataGraphConfig, task_dict, data_dict
):
    configurations = config.task_config
    shape = tuple(config.width for _ in range(config.dimensions))

    # Sweep Task IDs: (iter; 0; direction; grid)
    # Generate Sweep Tasks
    for direction_idx in config.direction_list:
        for i in range(config.steps):
            grid_generator = np.ndindex(shape)

            for grid_tuple in grid_generator:
                # Task ID:
                grid_idx = (i,) + grid_tuple
                task_idx = (iter,) + (0,) + (direction_idx,) + grid_idx
                task_id = TaskID("T", task_idx, 0)

                # Task Placement Info
                task_placement_info = configurations(task_id)

                # Direction Flags
                direction_flags = f"{direction_idx:08b}"

                # Task Dependencies (Sweep)
                dependency_list = []

                for j in range(config.dimensions + 1):
                    dependency_grid = list(grid_idx)
                    step = (-1) ** int(direction_flags[-(j + 1)])
                    dependency_grid[j] -= step
                    if in_domain(config.steps, config.width, dependency_grid):
                        dependency_idx = (
                            (iter,) + (0,) + (direction_idx,) + tuple(dependency_grid)
                        )
                        dependency = TaskID("T", dependency_idx, 0)
                        dependency_list.append(dependency)

                # Task Dependencies (Update/Reduce)
                if iter > 0:
                    dependency_idx = (iter - 1,) + (1,) + grid_idx
                    dependency = TaskID("T", dependency_idx, 0)
                    dependency_list.append(dependency)

                # Task Data Dependencies
                data_dependencies, data_dict = get_data_dependencies(
                    task_id, data_dict, data_config
                )

                # print("Creating Sweep Task: ", task_id)

                # Task Mapping
                task_mapping = get_mapping(config, task_id)
                task_dict[task_id] = TaskInfo(
                    task_id,
                    task_placement_info,
                    dependency_list,
                    data_dependencies,
                    task_mapping,
                )


def create_update_iteration(
    iter: int, config: SweepConfig, data_config: DataGraphConfig, task_dict, data_dict
):
    # Update task IDS: (iteration: 1; cell)

    configurations = config.task_config
    shape = tuple(config.width for _ in range(config.dimensions))
    for i in range(config.steps):
        grid_generator = np.ndindex(shape)
        for grid_tuple in grid_generator:
            grid_idx = (i,) + grid_tuple
            task_idx = (iter,) + (1,) + grid_idx
            task_id = TaskID("T", task_idx, 0)

            # Task Placement Info
            task_placement_info = configurations(task_id)

            # Task Dependencies (Sweep)
            dependency_list = []

            for dir in config.direction_list:
                dependent_task_idx = (iter,) + (0,) + (dir,) + grid_idx
                dependency = TaskID("T", dependent_task_idx, 0)
                dependency_list.append(dependency)

            # This part is complete hack around to manually set task dependencies for dim=1
            ###############################################################################
            if grid_idx[0] > 0:
                dependent_task_idx = (iter,) + (1,) + (grid_idx[0] - 1,) + grid_idx[1:]
                dependency = TaskID("T", dependent_task_idx, 0)
                dependency_list.append(dependency)
            if grid_idx[1] > 0:
                dependent_task_idx = (iter,) + (1,) + (grid_idx[0], grid_idx[1] - 1)
                dependency = TaskID("T", dependent_task_idx, 0)
                dependency_list.append(dependency)
            ###############################################################################

            # Task Data Dependencies
            data_dependencies, data_dict = get_data_dependencies(
                task_id, data_dict, data_config
            )

            # Task Mapping
            task_mapping = get_mapping(config, task_id)

            # print("Creating Update Task: ", task_id)
            task_dict[task_id] = TaskInfo(
                task_id,
                task_placement_info,
                dependency_list,
                data_dependencies,
                task_mapping,
            )


@register_graph_generator
def make_sweep_graph(
    config: SweepConfig, data_config: DataGraphConfig = NoDataGraphConfig()
) -> Tuple[TaskMap, DataMap]:
    check_config(config)

    task_dict = dict()
    data_dict = dict()
    ##Task ID:
    #   sweep: (iteration; 0; direction; cell)
    #   update: (iteration; 1; cell)
    print("\033[91mCurrently only supports 2D (dim=1) sweep graph\033[0m")
    print(
        "\033[93mThere is a complete hack around to manually set task dependencies for dim=1\033[0m"
    )
    print("\033[96mThis is not a general solution\033[0m")
    print("\033[95mPlease fix this in create_update_iteration function\033[0m")
    for iter in range(config.max_iter):
        create_sweep_iteration(iter, config, data_config, task_dict, data_dict)
        create_update_iteration(iter, config, data_config, task_dict, data_dict)

    return task_dict, data_dict
