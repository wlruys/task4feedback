from .mesh.base import Geometry, Cell, Edge
from ..interface import DataBlocks, TaskGraph
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
import os
from typing import List, Optional, Callable, Self
from .. import fastsim2 as fastsim
import numpy as np
from task4feedback.fastsim2 import DeviceType
import pymetis
import inspect
from typing import Type
from task4feedback.interface import System, VariantBuilder
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import wandb
import os

def spring_layout(G):
    pos = nx.spring_layout(G, seed=5, scale=600)
    for name, (x, y) in pos.items():
        node = G.nodes[name]
        node["x"] = x
        node["y"] = y


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
        """Get the block ID for a given key."""
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

    def add_block(
        self, key: DataKey, size: int, location: int, x: float = 0, y: float = 0
    ):
        block = self.blocks.add_block(
            name=str(key), size=size, location=location, x_pos=x, y_pos=y
        )
        self.map.add_block(key, block.id)

    def get_object(self, block_id: int):
        return self.map.get_object(block_id)

    def get_key(self, block_id: int):
        return self.map.get_key(block_id)

    def get_block(self, key: DataKey):
        return self.map.get_block(key)

    def get_blocks(self, object: Cell | Edge):
        return self.map.get_blocks(object)


class ComputeDataGraph(TaskGraph):
    def ___init__(self, data: DataGeometry):
        super(ComputeDataGraph, self).__init__()
        self.data = data

    def get_blocks(self):
        return self.data.blocks

    def get_data_geometry(self):
        return self.data

    def get_compute_cost(self, task_id: int, arch: DeviceType):
        time = self.graph.get_time(task_id, arch)
        assert time >= 0, f"Task {task_id} has no time for architecture {arch}"
        return time

    def get_shared_data(self, task_self: int, task_other: int):
        """
         Total size of all shared data blocks from task_other to task_self
        """

        read_self = self.tasks[task_self].read
        read_other = self.tasks[task_other].read

        write_self = self.tasks[task_self].write
        write_other = self.tasks[task_other].write

        shared_reads = set(read_other) & (set(read_self))
        shared_write_reads = set(write_other) & (set(read_self))
        shared = shared_reads.union(shared_write_reads)

        total_size = 0
        for block_id in shared:
            block = self.data.blocks.get_block(block_id)
            size = block.size
            total_size += size
        return total_size

    def get_read_data(self, task_id: int):
        # Get size of all data blocks that this task reads
            read_self = self.tasks[task_id].read
            total_size = 0
            for block_id in read_self:
                block = self.data.blocks.get_block(block_id)
                size = block.size
                total_size += size
            return total_size
    
    def get_write_data(self, task_id: int):
        # Get size of all data blocks that this task writes
        write_self = self.tasks[task_id].write
        total_size = 0
        for block_id in write_self:
            block = self.data.blocks.get_block(block_id)
            size = block.size
            total_size += size
        return total_size

    def get_weighted_graph(
        self, arch: DeviceType, bandwidth: int = 1000, task_ids: Optional[list] = None, symmetric: bool = False 
    ):
        adjacency_list = []
        adj_starts = []
        vweights = []
        eweights = []
        task_to_local = {}
        bandwidth = bandwidth / (1e6)

        if task_ids is None:
            task_ids = range(len(self))

        task_ids = list(task_ids)

        for i, task_id in enumerate(task_ids):
            task_to_local[task_id] = i

        for _, task_id in enumerate(task_ids):
            compute_cost = self.get_compute_cost(task_id, arch)
            vweights.append(compute_cost)

        if not symmetric:
            for _, task_id in enumerate(task_ids):
                adj_starts.append(len(adjacency_list))
                for dep_task_id in self.tasks[task_id].dependencies:
                    if dep_task_id not in task_to_local:
                        continue
                    data_cost = self.get_shared_data(task_id, dep_task_id)
                    data_cost /= bandwidth
                    data_cost = max(data_cost, 1)

                    eweights.append(data_cost)
                    adjacency_list.append(dep_task_id)
            adj_starts.append(len(adjacency_list))
        else:
            edges_dir = {}
            for _, u in enumerate(task_ids):
                for v in self.tasks[u].dependencies:
                    if v not in task_to_local:
                        continue
                    w = self.get_shared_data(u, v) / bandwidth
                    w = max(w, 1)
                    edges_dir[(u, v)] = w

            undirected_max = {}
            for (u, v), w in edges_dir.items():
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                prev = undirected_max.get((a, b))
                if prev is None or w > prev:
                    undirected_max[(a, b)] = w

            neighbor_map = {tid: [] for tid in task_ids}
            weight_map = {tid: [] for tid in task_ids}

            for (a, b), w in undirected_max.items():
                neighbor_map[a].append(b); weight_map[a].append(w)
                neighbor_map[b].append(a); weight_map[b].append(w)

            for u in task_ids:
                adj_starts.append(len(adjacency_list))
                if neighbor_map[u]:
                    pairs = sorted(zip(neighbor_map[u], weight_map[u]), key=lambda p: p[0])
                    for v, w in pairs:
                        adjacency_list.append(v)
                        eweights.append(w)
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
    compute_tasks_by_state: dict
    compute_task_mapping_dict: dict

    data_task_mapping_dict: dict
    data_tasks_by_state: dict
    data_task_source_device: dict
    data_task_virtual: dict
    data_task_block: dict

    def parse_state(env, time: Optional[int] = None):
        if time is None:
            time = env.simulator.time

        sim = env.simulator
        simulator_state = sim.state
        task_runtime = simulator_state.get_task_runtime()
        static_graph = simulator_state.get_tasks()

        n_compute_tasks = task_runtime.get_n_compute_tasks()
        n_data_tasks = task_runtime.get_n_data_tasks()

        compute_tasks_by_state = defaultdict(lambda: list())
        data_tasks_by_state = defaultdict(lambda: list())
        compute_task_mapping_dict = {}

        data_task_mapping_dict = {}
        data_task_source_device = {}
        data_task_virtual = {}
        data_task_block = {}

        for i in range(n_compute_tasks):
            task_state = task_runtime.get_compute_task_state_at_time(i, time)
            compute_tasks_by_state[task_state].append(i)
            device_id = task_runtime.get_compute_task_mapped_device(i)
            compute_task_mapping_dict[i] = device_id

        for i in range(n_data_tasks):
            task_state = task_runtime.get_data_task_state_at_time(i, time)
            data_tasks_by_state[task_state].append(i)
            data_id = static_graph.get_data_id(i)

            device_id = task_runtime.get_data_task_mapped_device(i)
            source_device = task_runtime.get_data_task_source_device(i)
            is_virtual = task_runtime.is_data_task_virtual(i)

            data_task_source_device[i] = source_device
            data_task_virtual[i] = is_virtual
            data_task_mapping_dict[i] = device_id
            data_task_block[i] = data_id

        return EnvironmentState(
            time=time,
            compute_tasks_by_state=compute_tasks_by_state,
            data_tasks_by_state=data_tasks_by_state,
            compute_task_mapping_dict=compute_task_mapping_dict,
            data_task_mapping_dict=data_task_mapping_dict,
            data_task_source_device=data_task_source_device,
            data_task_virtual=data_task_virtual,
            data_task_block=data_task_block,
        )

    @staticmethod
    def from_env(env, time: Optional[int] = None):
        return EnvironmentState.parse_state(env, time)


class DynamicWorkload:
    def __init__(self):
        pass 

    def set_geometry(self, geom: Geometry):
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
        self, distribution: Optional[Callable[[int], float]] = None
    ):
        if distribution is None:
            # Default distribution is uniform
            def distribution(x):
                return 1.0

        cell_weights = [distribution(i) for i in range(self.num_cells)]
        weight_sum = sum(cell_weights)
        normalized_weights = [weight / weight_sum for weight in cell_weights]
        self.set_inital_mass(normalized_weights)

    def get_workload(self, level: int) -> list:
        return self.level_workload[level]
    
    def get_scaled_cell_workload(self, level: int, cell: int) -> float:
        return self.level_workload[level][cell] * self.num_cells

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
        title="workload_animation.mp4",
        folder=None,
        interval=None,
        colormap="viridis",
        normalize=True,
        show=True,
        max_radius=0.1,
        figsize=(8, 8),
        video_seconds=15,
        bitrate=300,
        dpi=300,
    ):
        from .mesh.plot import create_mesh_plot
        if folder is None:
            if wandb is None or wandb.run is None or wandb.run.dir is None:
                folder = "."
            else:
                folder = wandb.run.dir

        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, title)

        fig, ax = create_mesh_plot(self.geom, title=title)

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
        
        if interval is None:
            interval = int(video_seconds * 1000 / len(self.levels))

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.levels),
            interval=interval,
            blit=True,
            repeat=False,
        )

        if filename:
            try:
                ani.save(filename, writer="ffmpeg", dpi=dpi, bitrate=bitrate)
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


def make_drifting_circle_trajectory(
    geom: Geometry,
    num_steps: int,
    radius: float = 0.5,
    speed: float = 0.01,
    direction_std: float = 0.1,
    center=None,
    initial_angle: Optional[float] = None,
    seed: int = 0,
):
    """
    Generate a drifting circular trajectory that changes direction gradually.
    This is meant to simulate quasi-periodic behavior with smooth directional noise.

    Parameters:
    -----------
    geom : Geometry
        Geometry object used to define boundaries and centroids.
    num_steps : int
        Number of time steps in the trajectory.
    radius : float
        Radius of the local circle (as fraction of domain size).
    speed : float
        Drift speed per time step.
    direction_std : float
        Standard deviation of direction change per step (in radians).
    center : np.ndarray
        Initial center position. If None, randomly chosen.
    initial_angle : float
        Initial drift angle. If None, random in [0, 2π].

    Returns:
    --------
    np.ndarray
        (num_steps, 2) array of 2D coordinates.
    """
    width = geom.get_max_coordinate(0) - geom.get_min_coordinate(0)
    height = geom.get_max_coordinate(1) - geom.get_min_coordinate(1)
    domain_size = min(width, height)
    circle_radius = radius * domain_size
    drift_speed = speed * domain_size

    rng = np.random.RandomState(seed)

    if center is None:
        start_idx = rng.randint(0, len(geom.cells))
        center = np.copy(geom.get_centroid(start_idx))

    angle = (
        rng.uniform(-2 * np.pi, 2 * np.pi) if initial_angle is None else initial_angle
    )
    phase = 0.0
    traj = np.zeros((num_steps, 2))

    for i in range(num_steps):
        # 1) store point
        traj[i] = center + circle_radius * np.array([np.cos(phase), np.sin(phase)])

        # 2) update local phase and global heading
        phase += 2 * np.pi / 50.0
        angle += rng.normal(0.0, direction_std)
        center += drift_speed * np.array([np.cos(angle), np.sin(angle)])

        # 3) reflect if *edge* crosses a wall
        for d in (0, 1):
            lo, hi = geom.get_min_coordinate(d), geom.get_max_coordinate(d)
            if center[d] - circle_radius < lo:
                center[d] = lo + (lo - (center[d] - 2 * circle_radius))
                angle = np.pi - angle if d == 0 else -angle
            elif center[d] + circle_radius > hi:
                center[d] = hi - ((center[d] + 2 * circle_radius) - hi)
                angle = np.pi - angle if d == 0 else -angle

        # 4) keep angle bounded
        angle %= 2 * np.pi

    return traj



def gaussian_pdf(x, mean, std):
    """
    Isotropic d-variate Gaussian PDF.
    
    x     : array-like, shape (N, d) or (d,)
    mean  : array-like, shape (d,)
    std   : float or array-like (must be positive) — the standard deviation(s).
    """
    x    = np.atleast_2d(x)             # ensure shape (N, d)
    mu   = np.asarray(mean)
    sigma = np.asarray(std)
    if np.any(sigma <= 0):
        raise ValueError("`std` must be positive")
    
    d    = x.shape[1]
    var  = sigma**2                     # variance
    sq_dist = np.sum((x - mu)**2, axis=1)
    
    # normalization: (2πσ²)^(-d/2) = (2π)^{-d/2} · σ^{-d}
    norm_const = (2 * np.pi * var) ** (-0.5 * d)
    
    return norm_const * np.exp(-sq_dist / (2 * var))


def gaussian_bump(x, mean, std, scale=1.0):
    return scale * gaussian_pdf(x, mean, std)

def linear_growth_decay(start, end, num_steps, step):
    half = num_steps // 2
    if step < half:
        return start + (end - start) * (step / half)
    else:
        return end - (end - start) * ((step - half) / (num_steps - half))

def reverse_linear_growth_decay(start, end, num_steps, step):
    half = num_steps // 2
    if step < half:
        return end - (end - start) * (step / half)
    else:
        return start + (end - start) * ((step - half) / (num_steps - half))


def gaussian_bump_at_t(x, mean, min_std=0.1, max_std=0.2, min_scale=0.1, max_scale=0.5, t=0, num_steps=10):
    std = reverse_linear_growth_decay(min_std, max_std, num_steps, t)
    scale = linear_growth_decay(min_scale, max_scale, num_steps, t)
    return gaussian_bump(x, mean, std, scale)
    
@dataclass 
class GaussianBump:
    center: np.ndarray
    min_std: float = 0.1
    max_std: float = 0.2
    min_scale: float = 0.1
    max_scale: float = 0.5
    t: int = 0
    num_steps: int = 10

    def workload(self, x, t):
        return gaussian_bump_at_t(
            x, 
            self.center, 
            self.min_std, 
            self.max_std, 
            self.min_scale, 
            self.max_scale, 
            t, 
            self.num_steps
        )
    
    def get_workload_and_advance(self, x):
        w =  self.workload(x, self.t)
        self.t += 1
        return w 
    
    def is_alive(self):
        return self.t < self.num_steps

def create_bump_random_center(rng: np.random.RandomState, min_std = 0.1, max_std: float = 0.3, min_scale: float = 0.05, max_scale: float  = 0.5, min_life = 25, max_life = 50):
    x = rng.uniform(0, 1, size=2)
    life = rng.randint(min_life, max_life)

    return GaussianBump(x, min_std, max_std, min_scale, max_scale, t=0, num_steps=life)

class TrajectoryWorkload(DynamicWorkload):

    def generate_workload(
        self,
        num_levels: int,
        traj_type: str = "circle",
        start_step: int = 0,
        lower_bound: float = 0.05,
        upper_bound: float = 3,
        scale: float = 2,
        seed: int = 0,
        traj_specifics: Optional[dict] = None,
    ):
        if traj_type == "circle":
            trajectory = make_circle_trajectory(
                self.geom, num_steps=num_levels, **traj_specifics
            )
            self.random = False
        elif traj_type == "drift":
            trajectory = make_drifting_circle_trajectory(
                self.geom,
                num_steps=num_levels,
                **traj_specifics,
                seed=seed,
            )
            self.random = True
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")

        centroids = np.zeros((self.num_cells, 2))
        for i, cell in enumerate(self.geom.cells):
            centroids[i] = np.copy(self.geom.get_centroid(i))

        #Normalize starting step workload
        total_workload = np.sum(self.level_workload[start_step])
        assert( total_workload > 0), f"Total workload at level {start_step} is zero, cannot normalize."

        for j in range(start_step + 1, num_levels):
            self.level_workload[j] = np.copy(self.level_workload[0])

            gaussian_workload = (
                gaussian_pdf(centroids, trajectory[j], scale) * upper_bound
            )

            self.level_workload[j] = gaussian_workload

            self.level_workload[j] = np.clip(
                a=self.level_workload[j], a_min=lower_bound, a_max=upper_bound
            )

            # Keep total workload constant
            total_workload = np.sum(self.level_workload[j])
            assert( total_workload > 0), f"Total workload at level {j} is zero, cannot normalize."
            self.level_workload[j] /= total_workload


class BumpWorkload(DynamicWorkload):

    def generate_workload(
            self, 
            num_levels: int, 
            start_step: int = 0,
            lower_bound: float = 0.05,
            upper_bound: float = 3,
            seed: int = 0,
            **kwargs
    ):
        self.random = True
        rng = np.random.RandomState(seed)
        centroids = np.zeros((self.num_cells, 2))
        for i, cell in enumerate(self.geom.cells):
            centroids[i] = self.geom.get_centroid(i)

        total_workload = np.sum(self.level_workload[start_step])
        assert( total_workload > 0), f"Total workload at level {start_step} is zero, cannot normalize."

        bumps = [] 
        bumps.append(create_bump_random_center(rng=rng))

        for j in range(start_step + 1, num_levels):
            self.level_workload[j] = np.copy(self.level_workload[0])

            for bump in bumps:
                workload = bump.get_workload_and_advance(centroids)
                self.level_workload[j] += workload
            
            bumps = [b for b in bumps if b.is_alive()]
                
            #Create a new bump with probability 0.1
            if rng.rand() < 0.1:
                bumps.append(create_bump_random_center(rng=rng))

            self.level_workload[j] = np.clip(
                a=self.level_workload[j], a_min=lower_bound, a_max=upper_bound
            )

            # Keep total workload constant
            total_workload = np.sum(self.level_workload[j])
            assert( total_workload > 0), f"Total workload at level {j} is zero, cannot normalize."
            self.level_workload[j] /= total_workload


        

            


@dataclass
class GraphConfig:
    pass


class GraphRegistry:
    _registry: dict[Type, Callable[[Geometry, GraphConfig, System], TaskGraph]] = {}

    @classmethod
    def register(
        cls,
        func: Callable[[Geometry, GraphConfig, System], TaskGraph],
        config: Type[GraphConfig],
    ):
        cls._registry[config] = func
        return func

    @classmethod
    def get(
        cls, config: Type[GraphConfig]
    ) -> Optional[Callable[[Geometry, GraphConfig, System], TaskGraph]]:
        return cls._registry.get(config)

    @classmethod
    def build(cls, geometry: Geometry, config: GraphConfig, variant: Optional[type[VariantBuilder]] = None, system: Optional[System] = None) -> Optional[TaskGraph]:
        """
        Build a graph of the specified type using the provided geometry and configuration.
        """
        graph_builder = cls.get(type(config))
        if graph_builder is not None:
            return graph_builder(geometry, config, system=system)
        else:
            raise ValueError(f"Graph type '{config}' is not registered.")

def register_graph(cls, cfg):
    GraphRegistry.register(cls, cfg)


def build_graph(geometry: Geometry, config: GraphConfig, system: Optional[System] = None):
    return GraphRegistry.build(geometry, config, system=system)
