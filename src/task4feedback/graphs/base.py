from .mesh.base import Geometry, Cell, Edge
from ..interface import DataBlocks, Graph
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
import gravis as gv
import os
from typing import List, Optional, Callable, Self
from .. import fastsim2 as fastsim
import numpy as np
from task4feedback.fastsim2 import DeviceType
import pymetis


def spring_layout(G):
    pos = nx.spring_layout(G, seed=5, scale=600)
    for name, (x, y) in pos.items():
        node = G.nodes[name]
        node["x"] = x
        node["y"] = y


def draw(G, filename="graph.html"):
    # spring_layout(G)
    fig = gv.d3(
        G,
        layout_algorithm_active=True,
        graph_height=1000,
        use_edge_size_normalization=True,
        use_node_size_normalization=True,
        node_size_normalization_max=30,
    )
    if os.path.exists(filename):
        os.remove(filename)

    if filename.endswith(".html"):
        fig.export_html(filename)
    elif filename.endswith(".png"):
        fig.export_png(filename)
    elif filename.endswith(".jpg"):
        fig.export_jpg(filename)


@dataclass(frozen=True, eq=True, order=True, slots=True)
class DataKey:
    object: Cell | Edge
    id: int | str | tuple


def filter_key_list(
    obj_list: list[DataKey],
) -> tuple[list[Cell], list[tuple[Edge, Cell]]]:
    edge_list = []
    cell_list = []

    for obj in obj_list:
        if isinstance(obj.object, Edge):
            edge_list.append((obj.object, obj.id[0]))
        elif isinstance(obj.object, Cell):
            cell_list.append(obj.object)

    return cell_list, edge_list


tree = lambda: defaultdict(tree)


@dataclass
class NestedKeyDict(dict):
    container: defaultdict = field(default_factory=tree)

    def add(self, keys: DataKey, value):
        d = self.container
        d = d[keys.object]
        keys = keys.id

        if not isinstance(keys, tuple):
            keys = (keys,)

        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value

    def get(self, keys: DataKey | Cell | Edge):
        if isinstance(keys, Cell) or isinstance(keys, Edge):
            return self.container[keys]

        d = self.container
        d = d[keys.object]
        keys = keys.id
        if not isinstance(keys, tuple):
            keys = (keys,)
        for key in keys:
            d = d[key]
        return d

    def _get_leaves(self, d, keys: Cell | Edge, values: Optional[list] = None):
        leaves = []
        for k, v in d.items():
            if values is not None:
                if isinstance(k, int):
                    if k not in values:
                        continue
            if isinstance(v, int):
                leaves.append(v)
            else:
                leaves.extend(self._get_leaves(v, k))
        return leaves

    def get_leaves(self, keys: DataKey | Cell | Edge, values: Optional[list] = None):
        # Get all leaf int values from the nested dict
        d = self.get(keys)
        leaves = []
        for k, v in d.items():
            if values is not None:
                if isinstance(k, int):
                    if k not in values:
                        continue
            if isinstance(v, int):
                leaves.append(v)
            else:
                leaves.extend(self._get_leaves(v, k))
        return leaves

    def __setitem__(self, keys: DataKey | Cell | Edge, value: int):
        self.add(keys, value)

    def __getitem__(self, keys: DataKey):
        return self.get(keys)


@dataclass
class GeometryIDMap:
    object_to_blocks: dict = field(default_factory=lambda: defaultdict(list))
    key_to_block: NestedKeyDict = field(default_factory=NestedKeyDict)
    block_to_object: dict = field(default_factory=dict)
    block_to_key: dict = field(default_factory=dict)

    def add_block(self, key: DataKey, block_id: int):
        self.object_to_blocks[key.object].append(block_id)
        self.key_to_block[key] = block_id
        self.block_to_object[block_id] = key.object
        self.block_to_key[block_id] = key

    def get_block(self, key: DataKey):
        return self.key_to_block[key]

    def get_object(self, block_id: int):
        return self.block_to_object[block_id]

    def get_blocks(self, object: Cell | Edge):
        return self.object_to_blocks[object]

    def get_key(self, block_id: int):
        return self.block_to_key[block_id]


@dataclass
class DataGeometry:
    geometry: Geometry
    blocks: DataBlocks = field(default_factory=DataBlocks)
    map: GeometryIDMap = field(default_factory=GeometryIDMap)

    def add_block(self, key: DataKey, size: int, location: int):
        block = self.blocks.add_block(name=str(key), size=size, location=location)
        self.map.add_block(key, block.id)

    def get_object(self, block_id: int):
        return self.map.get_object(block_id)

    def get_key(self, block_id: int):
        return self.map.get_key(block_id)

    def get_block(self, key: DataKey):
        return self.map.get_block(key)

    def get_blocks(self, object: Cell | Edge):
        return self.map.get_blocks(object)


class ComputeDataGraph(Graph):
    def ___init__(self, data: DataGeometry):
        super(ComputeDataGraph, self).__init__()
        self.data = data

    def get_blocks(self):
        return self.data.blocks

    def get_data_geometry(self):
        return self.data

    def get_compute_cost(self, task_id: int, arch: DeviceType):
        return self.graph.get_time(task_id, arch)

    def get_shared_data(self, task_self: int, task_other: int):
        # Total size of all shared data blocks from task_other to task_self
        # For example, if task_self depends on task_other it is the size of the data that task_self needs to read from task_other
        read_self = self.graph.get_read_data(task_self)

        read_other = self.graph.get_read_data(task_other)
        write_other = self.graph.get_write_data(task_other)

        shared = set(read_self) & (set(read_other) | set(write_other))

        total_size = 0

        for block_id in shared:
            block = self.data.blocks.get_block(block_id)
            size = block.size
            total_size += size
        return total_size

    def get_weighted_graph(
        self, arch: DeviceType, bandwidth: int = 1000, task_ids: Optional[list] = None
    ):
        adjacency_list = []
        adj_starts = []
        vweights = []
        eweights = []
        task_to_local = {}

        if task_ids is None:
            task_ids = range(len(self))

        for i, task_id in enumerate(task_ids):
            task_to_local[task_id] = i

        for i, task_id in enumerate(task_ids):
            adj_starts.append(len(adjacency_list))
            compute_cost = self.get_compute_cost(task_id, arch)
            vweights.append(compute_cost)

            # print(f"task_id: {task_id}, compute_cost: {compute_cost}")

            for dep_task_id in self.graph.get_dependencies(task_id):
                if dep_task_id not in task_ids:
                    continue
                data_cost = self.get_shared_data(task_id, dep_task_id)
                data_cost /= bandwidth
                eweights.append(data_cost)
                adjacency_list.append(dep_task_id)
                # print(
                #     f"task_id: {task_id}, dep_task_id: {dep_task_id}, data_cost: {data_cost}"
                # )

        adj_starts.append(len(adjacency_list))

        adjacency_list = np.array(adjacency_list)
        adj_starts = np.array(adj_starts)
        vweights = np.array(vweights)
        eweights = np.array(eweights)

        return task_to_local, adjacency_list, adj_starts, vweights, eweights


@dataclass
class WeightedCellGraph:
    cells: np.ndarray
    adjacency: np.ndarray
    xadj: np.ndarray
    vweights: np.ndarray
    eweights: np.ndarray


def weighted_partition(
    nparts: int,
    adjacency_list: np.ndarray,
    adj_starts: np.ndarray,
    vweights: np.ndarray,
    eweights: np.ndarray,
):
    adjacency_list = adjacency_list.astype(np.int32)
    adj_starts = adj_starts.astype(np.int32)
    vweights = vweights.astype(np.int32)
    eweights = eweights.astype(np.int32)

    # --- SYMMETRY FIX: average weights on mismatched edges ---
    nverts = vweights.shape[0]
    for u in range(nverts):
        start_u = adj_starts[u]
        end_u = adj_starts[u + 1]
        for idx in range(start_u, end_u):
            v = int(adjacency_list[idx])
            w_uv = int(eweights[idx])

            # find reverse edge v -> u
            start_v = adj_starts[v]
            end_v = adj_starts[v + 1]
            rev_idx = None
            for j in range(start_v, end_v):
                if int(adjacency_list[j]) == u:
                    rev_idx = j
                    break

            if rev_idx is not None:
                w_vu = int(eweights[rev_idx])
                if w_uv != w_vu:
                    avg = (w_uv + w_vu) // 2
                    eweights[idx] = avg
                    eweights[rev_idx] = avg
                    # print(
                    #     f"Fixed mismatch: set both edges ({u}->{v}) and ({v}->{u}) to weight {avg}"
                    # )
            # else:
            # optionally handle missing reverse edges
            # print(f"Warning: no reverse edge for {u}->{v}, weight={w_uv}")
    # ---------------------------------------------------------

    return pymetis.part_graph(
        nparts=nparts,
        adjncy=adjacency_list,
        xadj=adj_starts,
        vweights=vweights,
        eweights=eweights,
    )


def weighted_cell_partition(cell_graph: WeightedCellGraph, nparts: int):
    """
    Partition the cell graph using METIS.
    """
    # Convert the adjacency list to a format suitable for METIS
    adjacency_list = cell_graph.adjacency
    adj_starts = cell_graph.xadj
    vweights = cell_graph.vweights
    eweights = cell_graph.eweights

    # Call METIS to partition the graph
    edgecuts, parts = weighted_partition(
        nparts, adjacency_list, adj_starts, vweights, eweights
    )

    return edgecuts, parts


@dataclass
class EnvironmentState:
    time: int
    compute_tasks: List[fastsim.ComputeTask]
    data_tasks: List[fastsim.DataTask]
    compute_tasks_by_state: dict
    data_tasks_by_state: dict
    mapping_dict: dict
    data_task_source_device: dict
    data_task_virtual: dict
    data_task_block: dict

    def parse_state(env, time: Optional[int] = None):
        if time is None:
            time = env.simulator.time

        graph = env.simulator_factory.input.graph
        data = env.simulator_factory.input.data
        sim = env.simulator
        assert graph.ctasks is not None
        compute_tasks = graph.ctasks.get_compute_tasks()
        data_tasks = graph.ctasks.get_data_tasks()
        simulator_state = sim.state

        compute_tasks_by_state = defaultdict(lambda: list())
        mapping_dict = {}

        for task in compute_tasks:
            # print(f"Supported arch: {task.supported_architectures}")
            task_state = simulator_state.get_state_at(task.id, time)
            compute_tasks_by_state[task_state].append(task.id)
            device_id = simulator_state.get_mapping(task.id)
            mapping_dict[task.id] = device_id

        data_tasks_by_state = defaultdict(lambda: list())
        data_task_source_device = {}
        data_task_virtual = {}
        data_task_block = {}
        for task in data_tasks:
            task_state = simulator_state.get_state_at(task.id, time)
            data_tasks_by_state[task_state].append(task.id)
            associated_compute_task_id = task.get_compute_task()
            device_id = simulator_state.get_mapping(associated_compute_task_id)
            source_device = simulator_state.get_data_task_source(task.id)
            data_task_source_device[task.id] = source_device
            is_virtual = simulator_state.is_data_task_virtual(task.id)
            data_task_virtual[task.id] = is_virtual
            mapping_dict[task.id] = device_id
            data_task_block[task.id] = task.get_data_id()

        return EnvironmentState(
            time=time,
            compute_tasks=compute_tasks,
            data_tasks=data_tasks,
            compute_tasks_by_state=compute_tasks_by_state,
            data_tasks_by_state=data_tasks_by_state,
            mapping_dict=mapping_dict,
            data_task_source_device=data_task_source_device,
            data_task_virtual=data_task_virtual,
            data_task_block=data_task_block,
        )

    @staticmethod
    def from_env(env, time: Optional[int] = None):
        return EnvironmentState.parse_state(env, time)


class DynamicWorkload:
    def __init__(self, geom: Geometry):
        self.geom = geom
        n_cells = len(geom.cells)
        self.level_workload = defaultdict(lambda: np.zeros(n_cells))
        self.sink_source_counter = 0

    def set_inital_mass(self, mass_vector: list):
        self.level_workload[0] = np.asarray(mass_vector)
        self.level_workload[0] = np.clip(self.level_workload[0], 0, None)

    @property
    def num_cells(self) -> int:
        return len(self.geom.cells)

    @property
    def inital_mass(self) -> list:
        return self.level_workload[0]

    def generate_initial_mass(
        self, distribution: Callable[[int], float] = None, average_workload: int = 1000
    ):
        if distribution is None:

            def distribution(x):
                return 1.0

        cell_weights = [distribution(i) for i in range(self.num_cells)]

        weight_sum = sum(cell_weights)
        normalized_weights = [weight / weight_sum for weight in cell_weights]

        total_average_workload = average_workload * self.num_cells
        weights = [weight * total_average_workload for weight in normalized_weights]

        self.set_inital_mass(weights)

    def get_workload(self, level: int) -> list:
        return self.level_workload[level]

    def generate_workload(
        self,
        num_levels: int,
        start_step: int = 0,
        correlation_matrix: np.ndarray = None,
        step_size: float = 2000,
        lower_bound: float = 500,
        upper_bound: float = 3000,
        scale: float = 0.1,
    ):
        if correlation_matrix is None:
            correlation_matrix = np.eye(self.num_cells)
            centroids = np.zeros((self.num_cells, 2))
            for i, cell in enumerate(self.geom.cells):
                centroids[i] = self.geom.get_centroid(i)

            for i in range(self.num_cells):
                for j in range(self.num_cells):
                    dist = np.linalg.norm(centroids[i] - centroids[j]) ** 2
                    correlation_matrix[i, j] = np.exp(-dist / scale)

        # Shift diagonal and symmetrize just in case (if testing different kernels)
        correlation_matrix += np.eye(self.num_cells) * 1e-3
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

        # Scale by max
        max_val = np.max(correlation_matrix)
        correlation_matrix /= max_val

        L = np.linalg.cholesky(correlation_matrix)

        # Clip start step into the workload range
        start_step = max(start_step, 0)
        self.level_workload[start_step] = np.clip(
            self.level_workload[start_step], lower_bound, upper_bound
        )

        for i in range(start_step + 1, num_levels):
            z = np.random.normal(size=self.num_cells)

            eps = L @ z
            new_v = self.level_workload[i - 1] + np.sqrt(step_size) * eps

            for j in range(self.num_cells):
                if new_v[j] < lower_bound:
                    new_v[j] = lower_bound + (lower_bound - new_v[j])
                elif new_v[j] > upper_bound:
                    new_v[j] = upper_bound - (new_v[j] - upper_bound)

            self.level_workload[i] = new_v

    def work_per_level(self, level: int) -> float:
        return np.sum(self.level_workload[level])

    @property
    def levels(self) -> list:
        return sorted(self.level_workload.keys())

    def animate_workload(
        self,
        filename="workload_animation.mp4",
        interval=200,
        colormap="viridis",
        normalize=True,
        show=True,
        max_radius=0.1,
    ):
        """
        Animate the workload across different levels.

        Parameters:
        -----------
        filename : str
            Name of the output file for the animation
        interval : int
            Time interval between frames in milliseconds
        colormap : str
            Matplotlib colormap to use for coloring the circles
        normalize : bool
            Whether to normalize the radii across all levels
        show : bool
            Whether to display the animation
        max_radius : float
            Maximum radius for the circles as a fraction of the domain size

        Returns:
        --------
        Animation object
        """
        from matplotlib import pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle
        from .mesh.plot import create_mesh_plot

        fig, ax = create_mesh_plot(self.geom, title="Workload Animation")

        # Get domain size for scaling
        domain_width = self.geom.get_max_coordinate(0) - self.geom.get_min_coordinate(0)
        domain_height = self.geom.get_max_coordinate(1) - self.geom.get_min_coordinate(
            1
        )
        domain_size = min(domain_width, domain_height)

        # Find global max workload for normalization if required
        if normalize:
            max_workload = max(
                np.max(self.level_workload[level]) for level in self.levels
            )

        patches_collection = None

        # Animation update function
        def update(frame):
            nonlocal patches_collection

            # Remove previous circles
            if patches_collection is not None:
                patches_collection.remove()

            # Get the workload for this level
            level = self.levels[frame % len(self.levels)]
            workload = self.level_workload[level]

            ax.set_title(f"Workload at Level {level}")

            # Calculate circle sizes (proportional to workload)
            if normalize:
                radius_scale = max_workload
            else:
                radius_scale = np.max(workload)

            patches = []
            colors = []
            cmap = plt.get_cmap(colormap)

            for i, cell in enumerate(self.geom.cells):
                # Get cell centroid
                centroid = self.geom.get_centroid(i)

                radius = (
                    (workload[i] / radius_scale) * domain_size * max_radius
                    if radius_scale > 0
                    else 0
                )

                if radius > 0:
                    circle = Circle((centroid[0], centroid[1]), radius)
                    patches.append(circle)
                    colors.append(cmap(workload[i] / radius_scale))

            patches_collection = PatchCollection(
                patches, facecolors=colors, edgecolors="black", alpha=0.7, zorder=10
            )

            ax.add_collection(patches_collection)

            return [patches_collection]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.levels),
            interval=interval,
            blit=True,
            repeat=True,
        )

        if filename:
            try:
                ani.save(filename, writer="ffmpeg", fps=1000 / interval)
            except Exception as e:
                print(f"Error saving animation: {e}")

        if show:
            plt.show()

        return ani


@dataclass
class Trajectory:
    locations: np.ndarray
    bounds: np.ndarray


def make_random_walk_trajectory(
    geom: Geometry,
    num_steps: int,
    step_size: float = 0.1,
):
    """
    Generate trajectory with gaussian random walk with relecting boundary conditions.
    """

    # Generate random walk
    trajectory = np.zeros((num_steps, 2))

    # Get random start point
    start_idx = np.random.randint(0, len(geom.cells))
    trajectory[0] = geom.get_centroid(start_idx)

    # Adjust step size as a fraction of the domain size
    width = geom.get_max_coordinate(0) - geom.get_min_coordinate(0)
    height = geom.get_max_coordinate(1) - geom.get_min_coordinate(1)

    step_size = min(width, height) * step_size

    for i in range(1, num_steps):
        step = np.random.normal(size=2) * step_size
        new_location = trajectory[i - 1] + step

        if new_location[0] < geom.get_min_coordinate(0):
            new_location[0] = geom.get_min_coordinate(0) + (
                geom.get_min_coordinate(0) - new_location[0]
            )
        elif new_location[0] > geom.get_max_coordinate(0):
            new_location[0] = geom.get_max_coordinate(0) - (
                new_location[0] - geom.get_max_coordinate(0)
            )

        trajectory[i] = new_location

    return trajectory


def make_circle_trajectory(
    geom: Geometry, num_steps: int, radius: float = 0.5, center=None, max_angle=None
):
    if center is None:
        # Get center of mesh
        center = np.array(
            [
                (geom.get_min_coordinate(0) + geom.get_max_coordinate(0)) / 2,
                (geom.get_min_coordinate(1) + geom.get_max_coordinate(1)) / 2,
            ]
        )

    width = geom.get_max_coordinate(0) - geom.get_min_coordinate(0)
    height = geom.get_max_coordinate(1) - geom.get_min_coordinate(1)

    # Adjust radius as a fraction of the domain size
    radius = min(width, height) * radius

    if max_angle is None:
        max_angle = 2 * np.pi
    else:
        max_angle = max_angle * 2 * np.pi

    # Generate circle trajectory
    theta = np.linspace(0, max_angle, num_steps)
    trajectory = np.zeros((num_steps, 2))
    trajectory[:, 0] = center[0] + radius * np.cos(theta)
    trajectory[:, 1] = center[1] + radius * np.sin(theta)
    return trajectory


def gaussian_pdf(x, mean, std):
    grid = np.asarray(x)

    u = np.asarray(mean)
    s = np.asarray(std)
    d = grid.shape[1]

    sq_dist = np.sum((grid - u) ** 2, axis=1)
    norm_const = (2 * np.pi * s) ** (-0.5 * d)

    pdf_vals = norm_const * np.exp(-sq_dist / (2 * s))
    return pdf_vals


class TrajectoryWorkload(DynamicWorkload):
    def generate_workload(
        self,
        num_levels: int,
        start_step: int = 0,
        number_of_trajectories: int = 1,
        lower_bound: float = 1000,
        upper_bound: float = 3000,
        scale: float = 0.05,
        max_angle: float = 0.5,
    ):
        trajectory = make_circle_trajectory(
            self.geom, num_steps=num_levels, radius=0.25, max_angle=max_angle
        )
        # total_sum = 0.0
        centroids = np.zeros((self.num_cells, 2))
        for i, cell in enumerate(self.geom.cells):
            centroids[i] = self.geom.get_centroid(i)

        for j in range(start_step + 1, num_levels):
            self.level_workload[j] = np.copy(self.level_workload[0])

            gaussian_workload = (
                gaussian_pdf(centroids, trajectory[j], scale) * upper_bound
            )
            self.level_workload[j] += gaussian_workload

            self.level_workload[j] = np.clip(
                self.level_workload[j], min=lower_bound, max=upper_bound
            )
            # total_sum += np.sum(self.level_workload[j])
        # total_sum += np.sum(self.level_workload[0])
        # print(
        #     f"Total workload sum: {total_sum}, per cell: {total_sum / (self.num_cells * (num_levels+1))}"
        # )
        # exit()
