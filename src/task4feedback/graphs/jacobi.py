from .mesh.base import Geometry, Cell, Edge
from .mesh.partition import block_cyclic
from ..interface import DataBlocks, DeviceType, TaskTuple, VariantTuple
from .base import (
    DataGeometry,
    DataKey,
    GeometryIDMap,
    ComputeDataGraph,
    WeightedCellGraph,
    GraphConfig,
    weighted_cell_partition,
    weighted_partition,
    register_graph,
)
from dataclasses import dataclass
from ..interface.lambdas import VariantBuilder
from ..interface.wrappers import StaticExternalMapper
import random
from itertools import permutations
from collections import defaultdict
import torch
from typing import Self, List, Optional, Tuple, Dict
from task4feedback import fastsim2 as fastsim
from ..interface.wrappers import *
from scipy.optimize import linear_sum_assignment
import sympy
from ..interface.types import _bytes_to_readable
import numpy as np

from collections import deque
import math


@dataclass
class JacobiConfig(GraphConfig):
    steps: int = 1
    n: int = 4  # number of cells in x direction
    domain_ratio: float = 1.0  # ratio of n in y direction to x direction (1.0 = square grid)
    arithmetic_intensity: float = 1.0
    arithmetic_complexity: float = 1.0
    memory_intensity: float = 1.0
    boundary_width: float = 5.0
    boundary_complexity: float = 0.5
    level_memory: int = 1000000
    randomness: float = 0.0
    permute_idx: int = 0
    task_time: Optional[int] = None
    interior_time: Optional[int] = None
    boundary_time: Optional[int] = None
    compute_time: Optional[int] = None
    vcu_usage: float = 1.0
    task_internal_memory: int = 0
    bytes_per_element: int = 4  # Assuming float32 data type
    verbose: bool = True
    boundary_in_memory_calc: bool = True


def get_length_from_config(cfg: JacobiConfig):
    return int(np.ceil(cfg.n * cfg.domain_ratio))


class JacobiData(DataGeometry):
    @staticmethod
    def from_mesh(geometry: Geometry, config: JacobiConfig, system: Optional[System] = None):
        return JacobiData(geometry, config, system)

    def _create_blocks(self, system: Optional[System] = None):
        interiors_per_level = self.geometry.get_num_cells()
        edges_per_level = self.geometry.get_num_edges()

        y = sympy.symbols("y", real=True, positive=True)
        if not self.config.boundary_in_memory_calc:
            equation = interiors_per_level * y - self.config.level_memory / self.config.bytes_per_element
        else:
            equation = interiors_per_level * y + self.config.boundary_width * edges_per_level * (y) ** self.config.boundary_complexity - self.config.level_memory / self.config.bytes_per_element
        solution = sympy.solve(equation, y)
        y_value = solution[0].evalf()
        interior_elem = int(y_value)

        interior_size = interior_elem * self.config.bytes_per_element
        boundary_elem = interior_elem ** (self.config.boundary_complexity) * self.config.boundary_width
        boundary_size = boundary_elem * self.config.bytes_per_element

        if self.config.interior_time is not None:
            assert system is not None
            interior_size = system.fastest_bandwidth * self.config.interior_time

        if self.config.boundary_time is not None:
            assert system is not None
            boundary_size = system.fastest_bandwidth * self.config.boundary_time

        if self.config.compute_time is not None:
            assert system is not None
            assert self.config.interior_time is not None, "Interior time should be set to manually set compute time"
            assert self.config.boundary_time is not None, "Boundary time should be set to manually set compute time"
            self.config.memory_intensity = self.config.compute_time / interior_size * (system.fastest_gmbw / 1e6)

        interior_size = int(interior_size)
        boundary_size = int(boundary_size)

        self.interior_size = interior_size
        self.boundary_size = boundary_size

        assert system is not None
        if self.config.verbose:
            print(
                "Total (per-level) Interior Size",
                _bytes_to_readable(interior_size * interiors_per_level),
            )
            print(
                "Communication time for reference interior size: ",
                interior_size / system.fastest_bandwidth,
                _bytes_to_readable(interior_size),
                interior_elem,
                "elements",
            )
            print(
                "Communication time for reference boundary size: ",
                boundary_size / system.fastest_bandwidth,
                _bytes_to_readable(boundary_size),
            )
            print(
                "Compute time for reference interior: ",
                interior_elem**self.config.arithmetic_complexity * self.config.arithmetic_intensity / (system.fastest_flops / 1e6),
            )
            print(
                "Memory time for reference interior: ",
                (interior_size * self.config.memory_intensity) / (system.fastest_gmbw / 1e6),
            )

        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Centroid of the cell
            centroid = self.geometry.get_centroid(cell)
            centroid_x = centroid[0]
            centroid_y = centroid[1]

            # Create 2 data blocks per cell
            for i in range(2):
                self.add_block(
                    DataKey(Cell(cell), i),
                    size=interior_size,
                    location=0,
                    x=centroid_x,
                    y=centroid_y,
                )
                # print(f"Added interior block for cell {cell} at ({centroid_x}, {centroid_y}), size {interior_size}: ({centroid_x}, {centroid_y})")

            # Create 2 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                edge_center = self.geometry.get_edge_center(edge)
                edge_x = edge_center[0]
                edge_y = edge_center[1]
                for i in range(2):
                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), i)),
                        size=boundary_size,
                        location=0,
                        x=edge_x,
                        y=edge_y,
                    )
                    # print(f"Added edge block for cell {cell} at edge {edge}, size {boundary_size}: ({edge_x}, {edge_y})")

    def __init__(
        self,
        geometry: Geometry,
        config: JacobiConfig = JacobiConfig(),
        system: Optional[System] = None,
    ):
        super().__init__(geometry, DataBlocks(), GeometryIDMap())
        self.config = config
        self._create_blocks(system=system)

    def blocks_to_objects(self, blocks: list[int]):
        return [self.map.get_object(i) for i in blocks]

    def blocks_to_keys(self, blocks: list[int]):
        return [self.map.block_to_key[i] for i in blocks]

    def get_block_at_step(self, object: Cell | tuple[Cell, Edge], step: int):
        idx = self.idx_at_step(step)
        if isinstance(object, tuple):
            return self.map.get_block(DataKey(object[1], (object[0], idx)))
        return self.map.get_block(DataKey(object, idx))

    def idx_at_step(self, step: int):
        return step % 2

    def set_location(self, obj: Cell | Edge, location: int, step: Optional[int] = None):
        step_list = None if step is None else [step]
        id_list = self.map.key_to_block.get_leaves(obj, values=step_list)

        for i in id_list:
            self.blocks.set_location(i, location)

        if isinstance(obj, Cell):
            # Update edges as well
            for edge in self.geometry.cell_edges[obj.id]:
                id_list = self.map.key_to_block.get_leaves(DataKey(Edge(edge), (obj,)), values=step_list)
                for i in id_list:
                    self.blocks.set_location(i, location)

    def set_locations_from_list(self, location_list: list[int], step: Optional[int] = None):
        for i, location in enumerate(location_list):
            self.set_location(Cell(i), location, step)

    def randomize_locations(self, num_changes: int, location_list: list[int], step: Optional[int] = None):
        new_locations = []

        selected_cells = random.sample(range(len(self.geometry.cells)), num_changes)
        for i, cell in enumerate(selected_cells):
            new_location = random.choice(location_list)
            self.set_location(Cell(cell), new_location, step)
            new_locations.append(new_location)

        return selected_cells, new_locations

    def get_locations(self, as_dict: bool = False) -> list[int] | dict[int, int]:
        if not as_dict:
            # Return a list of locations for each cell
            locations = [0] * len(self.geometry.cells)
            for i in range(len(self.geometry.cells)):
                block_id = self.get_blocks(Cell(i))
                block_id = block_id[0]
                locations[i] = self.blocks.get_location(block_id)
            return locations

        locations = {}
        for i in range(len(self.geometry.cells)):
            block_id = self.get_blocks(Cell(i))
            block_id = block_id[0]
            locations[i] = self.blocks.get_location(block_id)
        return locations

    def remap_locations(self, location_map: dict[int, int]):
        cell_locations = self.get_locations()
        for cell_id, location in enumerate(cell_locations):
            if location in location_map:
                new_location = location_map[location]
                self.set_location(Cell(cell_id), new_location)

    def permute_locations(self, location_map: dict[int, int], permutation_idx: Optional[int] = None):
        # Form and apply a permutation of the location_map
        # NOTE: This is a brute force implementation (FORMS ALL PERMUTATIONS AT EVERY CALL)
        # This is terrifyingly inefficient, but it works for small location maps
        keys = list(location_map.keys())
        values = list(location_map.values())
        p = permutations(values)
        p = list(p)
        if permutation_idx is None:
            # Select a random permutation
            permutation_idx = random.randint(0, len(p) - 1)
        else:
            perm = p[permutation_idx]
        perm_map = {keys[i]: perm[i] for i in range(len(keys))}
        self.remap_locations(perm_map)
        return permutation_idx


class JacobiGraph(ComputeDataGraph):
    def xy_from_id(self, taskid: int) -> int:
        """
        Convert a task ID to its (x, y) coordinates in the Jacobi grid.
        And returns row-major order index.
        Only works for rectangular grids.
        """
        cell_id = self.task_to_cell[taskid]
        centroid = self.data.geometry.cell_points[self.data.geometry.cells[cell_id]].mean(axis=0)
        n = self.nx
        centroid = np.floor(centroid * n)

        x = int(centroid[0])
        y = int(centroid[1])

        # print(f"Task ID {taskid} -> Cell ID {cell_id} -> Centroid {centroid} -> (x, y) = ({x}, {y}) -> Index {int(x * n + y)}")
        return int(x * n + y)

    @property
    def nx(self) -> int:
        """
        Only works for rectangular grids.
        """
        return self.config.n

    @property
    def ny(self) -> int:
        """
        Only works for rectangular grids.
        """
        return int(np.ceil(self.config.n * self.config.domain_ratio))

    def _build_graph(self, retire_data: bool = False, system: System = None):
        self.task_to_cell = {}
        self.task_to_level = {}
        self.level_to_task = defaultdict(list)
        prev_interiors = {}
        self.max_requirement = 0

        if retire_data:
            self.dynamic = True
        else:
            self.dynamic = False

        for i in range(self.config.steps):
            for j, (cell, edges) in enumerate(self.data.geometry.cell_edges.items()):
                # Create task that:
                # -reads all of its block (interior and edges) and the edges of its neighbors
                # -writes to blocks of its self (interior and edges)

                idx = self.data.idx_at_step(i)

                name = f"Task(Cell({cell}), {i})"
                task_id = self.add_task(name)

                self.task_to_cell[task_id] = cell
                self.task_to_level[task_id] = i
                self.level_to_task[i].append(task_id)

                # print(f"Task {task_id} created with name {name}")

                interior_block = self.data.get_block_at_step(Cell(cell), i)
                interior_edges = []
                exterior_edges = []
                data_req = 0
                for edge in edges:
                    cell_dict = self.data.get_block(Edge(edge))
                    for neighbor, v in cell_dict.items():
                        if neighbor.id != cell:
                            exterior_edges.append(v[idx])
                        else:
                            interior_edges.append(v[idx])

                next_interior_block = self.data.get_block_at_step(Cell(cell), i + 1)
                next_interior_edges = []
                for edge in edges:
                    next_interior_edges.append(self.data.get_block_at_step((Cell(cell), Edge(edge)), i + 1))

                read_blocks = interior_edges + exterior_edges + [interior_block]
                write_blocks = next_interior_edges + [next_interior_block]
                prev_interiors[(cell, i)] = interior_edges + [interior_block]
                self.add_read_data(task_id, read_blocks)
                self.add_write_data(task_id, write_blocks)

                for data_id in read_blocks:
                    data_req += self.data.blocks.data.get_size(data_id)
                for data_id in write_blocks:
                    if data_id not in read_blocks:
                        data_req += self.data.blocks.data.get_size(data_id)

                assert (
                    system is None or data_req < system.arch_to_maxmem[DeviceType.GPU]
                ), f"Task {task_id} requires {data_req / 1e9:.2f} GB of data, which exceeds the maximum memory for GPU {system.arch_to_maxmem[DeviceType.GPU] / 1e9:.1f} GB"
                # Raise a warning if data_req exceeds half of maxmem
                if system is not None and data_req > system.arch_to_maxmem[DeviceType.GPU] / 2:
                    print(f"Warning: Task {task_id} requires {data_req / 1e9:.2f} GB of data, which exceeds half of the maximum memory for GPU {system.arch_to_maxmem[DeviceType.GPU] / 1e9:.1f} GB")
                self.max_requirement = max(self.max_requirement, data_req)
                # if data_req > 80e9:
                #     print(f"Task {task_id} requires {data_req/1e9} GB of data")
                #     print(f"Interior size: {self.data.blocks.data.get_size(interior_block)/1e9}")
                #     print(f"Interior edges size: {[self.data.blocks.data.get_size(e)/1e9 for e in interior_edges]} = {sum([self.data.blocks.data.get_size(e)/1e9 for e in interior_edges])}")
                #     print(f"Exterior edges size: {[self.data.blocks.data.get_size(e)/1e9 for e in exterior_edges]} = {sum([self.data.blocks.data.get_size(e)/1e9 for e in exterior_edges])}")
                #     print(f"Next interior size: {self.data.blocks.data.get_size(next_interior_block)/1e9}")
                #     print(f"Next interior edges size: {[self.data.blocks.data.get_size(e)/1e9 for e in next_interior_edges]} = {sum([self.data.blocks.data.get_size(e)/1e9 for e in next_interior_edges])}")

                if i > 0 and retire_data:
                    self.add_retire_data(task_id, prev_interiors[(cell, i - 1)])

    def __init__(
        self,
        geometry: Geometry,
        config: JacobiConfig,
        system: Optional[System] = None,
        variant: Optional[type[VariantBuilder]] = None,
    ):
        assert system is not None
        super(JacobiGraph, self).__init__()
        self.data = JacobiData.from_mesh(geometry, config, system=system)
        self.config = config
        self._build_graph()
        self.dynamic = False

        if variant is not None:
            self.apply_variant(variant)
        elif system is not None:
            self._apply_workload_variant(system)
        else:
            print("Warning: No variant or system provided, using default Jacobi variant for task time and architecture specs.")
            self.apply_variant(JacobiVariant)

        self.finalize()

    def _apply_workload_variant(self, system: System):
        # print("Building custom variant for system", system)

        class JacobiVariant(VariantBuilder):
            @staticmethod
            def build_variant(arch: DeviceType, task: TaskTuple) -> Optional[VariantTuple]:
                memory_usage = self.config.task_internal_memory
                vcu_usage = self.config.vcu_usage

                if system.get_flops(arch) == 0:
                    return None

                if self.config.task_time is not None:
                    expected_time = self.config.task_time
                    expected_time = int(expected_time)
                else:
                    num_elements = self.data.interior_size // self.config.bytes_per_element
                    expected_work = num_elements**self.config.arithmetic_complexity * self.config.arithmetic_intensity
                    expected_time = expected_work / system.get_flop_ms(arch)

                    expected_memory = self.data.interior_size * self.config.memory_intensity
                    expected_time = max(expected_time, expected_memory / system.get_gmbw_ms(arch))
                    expected_time = int(max(expected_time, 1))

                return VariantTuple(
                    arch,
                    memory_usage=memory_usage,
                    vcu_usage=vcu_usage,
                    expected_time=expected_time,
                )

        self.apply_variant(JacobiVariant)

    def randomize_locations(
        self,
        perc_change: float,
        location_list: Optional[list[int]] = None,
        min_loc: int = 0,
        max_loc: Optional[int] = None,
        verbose: bool = False,
        step: Optional[int] = None,
    ):
        num_changes = int(perc_change * len(self.data.geometry.cells))
        if verbose:
            print(f"Randomizing {num_changes} locations out of {len(self.data.geometry.cells)}")
        if location_list is None:
            if max_loc is None:
                raise ValueError("max_loc must be provided if location_list is None")
            location_list = list(range(min_loc, max_loc))

        selected_cells, new_locations = self.data.randomize_locations(num_changes, location_list, step)

        if verbose:
            print(f"Randomized locations for {len(selected_cells)} cells on step {step}:")
            for cell, new_location in zip(selected_cells, new_locations):
                print(f"Cell {cell} -> New Location: {new_location}")

        return selected_cells, new_locations

    def set_cell_locations(self, location_list: list[int], step: Optional[int] = None):
        self.data.set_locations_from_list(location_list, step)

    def set_cell_locations_from_dict(self, location_dict: dict[int, int]):
        for cell, location in location_dict.items():
            self.data.set_location(Cell(cell), location)

    def get_cell_locations(self, as_dict: bool = True) -> list[int] | dict[int, int]:
        return self.data.get_locations(as_dict=as_dict)

    def task_mapping_to_level_mapping(self, task_to_device: dict[int, int]):
        level_map = defaultdict(dict)
        for task_id, device in task_to_device.items():
            cell = self.task_to_cell[task_id]
            level = self.task_to_level[task_id]
            level_map[level][cell] = device
        return level_map

    def get_mapping_from_locations(self, as_dict=True) -> list[int] | dict[int, int]:
        mapping = []
        if as_dict:
            mapping = {}
        else:
            mapping = [0] * len(self)

        for task in self:
            for block in task.read:
                obj = self.data.get_object(block)
                if isinstance(obj, Cell):
                    location = self.data.blocks.get_location(block)
                    mapping[task.id] = location

        return mapping

    def get_num_iterations(self):
        return self.num_iterations

    def permute_locations(self, location_map: dict[int, int], permutation_idx: Optional[int] = None):
        return self.data.permute_locations(location_map, permutation_idx)

    def get_weighted_cell_graph(self, arch: DeviceType, bandwidth=1000, levels: Optional[list[int]] = None):
        """
        Given a list of levels, return the weighted cell interactions
        """

        if levels is None:
            levels = list(self.level_to_task.keys())

        tasks_in_levels = []
        for level in levels:
            tasks_in_levels += self.level_to_task[level]

        task_to_local, adj_list, adj_starts, vweights, eweights = self.get_weighted_graph(arch, bandwidth=bandwidth, task_ids=tasks_in_levels, symmetric=True)

        cell_vertex_cost = defaultdict(int)
        cell_neighbors_cost = defaultdict(lambda: defaultdict(int))

        for task_id in tasks_in_levels:
            local_task_id = task_to_local[task_id]
            cell = self.task_to_cell[task_id]
            cell_vertex_cost[cell] += vweights[local_task_id]
            start_idx = adj_starts[local_task_id]
            end_idx = adj_starts[local_task_id + 1]

            for i in range(start_idx, end_idx):
                neighbor = adj_list[i]
                neighbor_cell = self.task_to_cell[neighbor]
                if neighbor_cell != cell:
                    cell_neighbors_cost[cell][neighbor_cell] += eweights[i]
                else:
                    # self loop
                    cell_vertex_cost[cell] += eweights[i]

        # Unroll and return with local indicies (in case all cells are not present)
        cells = list(cell_vertex_cost.keys())

        vweights = [cell_vertex_cost[cell] for cell in cells]
        adj_list = []
        adj_starts = []
        eweights = []

        for i, cell in enumerate(cells):
            adj_starts.append(len(adj_list))
            for neighbor, weight in cell_neighbors_cost[cell].items():
                adj_list.append(neighbor)
                eweights.append(weight)

        adj_starts.append(len(adj_list))

        cells = np.asarray(cells, dtype=np.int64)
        adj_list = np.asarray(adj_list, dtype=np.int64)
        adj_starts = np.asarray(adj_starts, dtype=np.int64)
        vweights = np.asarray(vweights, dtype=np.int64)
        eweights = np.asarray(eweights, dtype=np.int64)

        # Make graph symmetric

        return WeightedCellGraph(cells, adj_list, adj_starts, vweights, eweights)

    def get_distributed_weighted_graph(
        self, bandwidth: int, task_ids: List[int], partition: List[int], arch: DeviceType = DeviceType.GPU, future_levels: int = 0, width: int = 8, length: int = 8, n_compute_devices: int = 4
    ) -> Tuple[
        List[List[int]],  # partitioned_tasks
        np.ndarray,  # vtxdist
        List[np.ndarray],  # xadj
        List[np.ndarray],  # adjncy
        List[np.ndarray],  # vwgt
        List[np.ndarray],  # adjwgt
        List[np.ndarray],  # vsize
    ]:
        """
        Build a weighted graph (CSR per partition) for distributed partitioning.

        Returns:
            partitioned_tasks: list of tasks for each partition (by partition index)
            vtxdist:           METIS-style vertex distribution array (prefix sums of per-part sizes)
            xadj:              CSR row pointer per partition
            adjncy:            CSR adjacency indices per partition (indices are global-local mapping per code logic)
            vwgt:              vertex weights (compute cost) per partition
            adjwgt:            edge weights (data transfer cost) per partition
            vsize:             vertex sizes (internal data size used for balancing) per partition
        Notes:
            - Uses integer division for bandwidth-normalized costs.
            - Keeps original semantics: adjncy indices reference the *global-local* index (over the
            sorted list of (task,part) pairs), not the position within a single partition.
        """

        # ---------- Basic checks & setup ----------
        num_tasks = len(task_ids)
        assert len(partition) == num_tasks, f"Length of 'partition' ({len(partition)}) must match number of tasks ({num_tasks})."

        min_part = min(partition)
        # assert min_part == 0, f"Partition must start from 0, got {min_part}"

        bandwidth = bandwidth / (1e6)

        # stride bounds how far "future levels" can look
        stride = width * length
        max_task_id = max(task_ids) if task_ids else 0
        future_levels = min(future_levels, (self.graph.size() - max_task_id) // stride)

        # ---------- Data structures per partition ----------
        # CSR components and weights per partition
        xadj: List[List[int]] = [[0] for _ in range(n_compute_devices)]
        adjncy: List[List[int]] = [[] for _ in range(n_compute_devices)]
        vwgt: List[List[int]] = [[] for _ in range(n_compute_devices)]  # compute time
        adjwgt: List[List[int]] = [[] for _ in range(n_compute_devices)]  # data transfer time
        vsize: List[List[int]] = [[] for _ in range(n_compute_devices)]  # internal data size proxy

        vtxdist: List[int] = [0]  # prefix of vertex counts (will accumulate when partition changes)
        partitioned_tasks: List[List[int]] = [[] for _ in range(n_compute_devices)]

        # Pair tasks with their partition and sort by partition to make vtxdist/xadj simpler.
        pairs: List[Tuple[int, int]] = sorted(zip(task_ids, partition), key=lambda x: x[1])

        # Map task_id -> "global-local" index (i.e., index into the sorted 'pairs' list).
        task_to_local: Dict[int, int] = {task_id: i for i, (task_id, _) in enumerate(pairs)}

        # ---------- Helpers ----------
        def avg_over_future(value: int) -> int:
            """Average a cumulative cost over (future_levels + 1), floor to >= 1."""
            return value // (future_levels + 1)

        # ---------- Main construction ----------
        partition_count = 0
        for i, (task_id, part) in enumerate(pairs):
            partitioned_tasks[part].append(task_id)

            # --- Vertex weights (compute + internal data size proxy) ---
            compute_cost = 0
            repartition_cost = 0

            current_level = self.task_to_level[task_id]

            # Base task contribution
            compute_cost += self.get_compute_cost(task_id, arch)
            repartition_cost += self.get_write_data(task_id) // bandwidth

            # Look-ahead into future levels within the *same cell*
            for fl in range(future_levels):
                level = current_level + fl
                for parent in self.level_to_task[level]:
                    if self.task_to_cell[parent] == self.task_to_cell[task_id]:
                        compute_cost += self.get_compute_cost(parent, arch)
                        repartition_cost += self.get_write_data(parent) // bandwidth
                    # original code broke after the first same-cell match
                    break

            vwgt[part].append(avg_over_future(compute_cost))
            vsize_val = max(avg_over_future(repartition_cost), 1)
            vsize[part].append(vsize_val)

            # --- Edges to dependencies that cross cells (boundary edges) ---
            for dep_task_id in self.tasks[task_id].dependencies:
                # Skip if dependency is in the same cell; only boundary costs matter here
                if self.task_to_cell[dep_task_id] == self.task_to_cell[task_id]:
                    continue

                # Base boundary data transfer
                boundary_cost = self.get_shared_data(task_id, dep_task_id) // bandwidth

                dep_level = self.task_to_level[dep_task_id]

                # Add look-ahead boundary costs between future "matching" tasks in same cells
                for fl in range(future_levels):
                    # Find a future dep task within dep cell at dep_level + fl
                    future_dep_task_id = 0
                    for ft_dep in self.level_to_task[dep_level + fl]:
                        if self.task_to_cell[ft_dep] == self.task_to_cell[dep_task_id]:
                            future_dep_task_id = ft_dep
                            break

                    # Find a future current task within current cell at current_level + fl
                    for ft_cur in self.level_to_task[current_level + fl]:
                        if self.task_to_cell[ft_cur] == self.task_to_cell[task_id]:
                            boundary_cost += self.get_shared_data(future_dep_task_id, ft_cur) // bandwidth
                            break

                adjwgt[part].append(max(avg_over_future(boundary_cost), 1))

                # adjncy uses the first task on current_level that lives in the *dependency's* cell
                for t in self.level_to_task[current_level]:
                    if self.task_to_cell[t] == self.task_to_cell[dep_task_id]:
                        adjncy[part].append(task_to_local[t])
                        break

            # Close the CSR row for this vertex
            xadj[part].append(len(adjncy[part]))

            # Update vtxdist when partition changes in the globally sorted order
            if i > 0 and pairs[i - 1][1] != part:
                vtxdist.append(i)
                partition_count += 1
            while partition_count != part:
                partition_count += 1
                vtxdist.append(i)

        # Final vertex count
        while partition_count < n_compute_devices:
            vtxdist.append(num_tasks)
            partition_count += 1

        # ---------- Final checks ----------
        assert len(vtxdist) == n_compute_devices + 1, f"vtxdist length {len(vtxdist)} does not match number of partitions + 1 ({n_compute_devices + 1})."

        # ---------- Symmetry: take max weight for (u,v) and (v,u) ----------
        # Build max weight per undirected pair using global indices.
        edge_max = {}  # key: (min(u_gl, v_gl), max(u_gl, v_gl)) -> max_w

        # 1) Collect max weight over all partitions/edges
        for p in range(n_compute_devices):
            for u_local in range(len(vwgt[p])):
                u_gl = int(vtxdist[p]) + u_local
                start = int(xadj[p][u_local])
                end = int(xadj[p][u_local + 1])
                for eidx in range(start, end):
                    v_gl = int(adjncy[p][eidx])
                    w = int(adjwgt[p][eidx])
                    a, b = (u_gl, v_gl) if u_gl < v_gl else (v_gl, u_gl)
                    prev = edge_max.get((a, b))
                    if prev is None or w > prev:
                        edge_max[(a, b)] = w

        # 2) Rewrite all edge weights to that max
        for p in range(n_compute_devices):
            for u_local in range(len(vwgt[p])):
                u_gl = int(vtxdist[p]) + u_local
                start = int(xadj[p][u_local])
                end = int(xadj[p][u_local + 1])
                for eidx in range(start, end):
                    v_gl = int(adjncy[p][eidx])
                    a, b = (u_gl, v_gl) if u_gl < v_gl else (v_gl, u_gl)
                    adjwgt[p][eidx] = int(edge_max[(a, b)])

        # ---------- Convert to numpy-friendly outputs ----------
        xadj_np = [np.asarray(x, dtype=np.int32) for x in xadj]
        adjncy_np = [np.asarray(x, dtype=np.int32) for x in adjncy]
        vwgt_np = [np.asarray(x, dtype=np.int32) for x in vwgt]
        adjwgt_np = [np.asarray(x, dtype=np.int32) for x in adjwgt]
        vsize_np = [np.asarray(x, dtype=np.int32) for x in vsize]
        vtxdist_np = np.asarray(vtxdist, dtype=np.int32)

        return (
            partitioned_tasks,
            vtxdist_np,
            xadj_np,
            adjncy_np,
            vwgt_np,
            adjwgt_np,
            vsize_np,
        )

    def initial_mincut_partition(
        self,
        arch: DeviceType = DeviceType.GPU,
        bandwidth: int = 1000,
        n_parts: int = 4,
        offset: int = 1,  # 1 to ignore cpu
    ):
        cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth, levels=[0, 1])
        edge_cut, partition = weighted_cell_partition(cell_graph, nparts=n_parts)
        partition = [x + offset for x in partition]
        return partition

    def mincut_per_levels(
        self,
        arch: DeviceType = DeviceType.GPU,
        bandwidth: int = 1000,
        level_chunks: int = 1,
        levels_per_chunk: int | None = None,  # override number of levels per chunk
        n_parts: int = 4,
        offset: int = 1,  # 1 to ignore cpu
        mode: str = "metis",
    ):
        # Oracle mode takes in level chunks and returns partitions based on the full knowledge of the workload
        partitions = {}
        levels = list(self.level_to_task.keys())
        levels = sorted(levels)

        if mode == "metis":
            level_size = len(levels) // level_chunks

            if levels_per_chunk is not None:
                level_size = levels_per_chunk

            for i in range(level_chunks):
                start = i * level_size
                end = (i + 1) * level_size

                if i == level_chunks - 1:
                    end = len(levels)

                levels_to_compute = levels[start:end]
                cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth, levels=levels_to_compute)

                edge_cut, partition = weighted_cell_partition(cell_graph, nparts=n_parts)
                partition = [x + offset for x in partition]
                partitions[(start, end)] = partition
        # Dynamic mode changes the partitions based on the current workload if certain thresholds are met
        elif mode == "dynamic":
            cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth, levels=[0])
            edge_cut, current_partition = weighted_cell_partition(cell_graph, nparts=n_parts)
            prev_level = 0
            for level in levels[1:]:
                # Check load imbalance of current partition
                load_per_part = [0 for _ in range(n_parts)]
                for task_id in self.level_to_task[level]:
                    cell_id = self.task_to_cell[task_id]
                    part = current_partition[cell_id]
                    load_per_part[part] += self.get_compute_cost(task_id, arch)
                # Check if the load imbalance is above a threshold
                if max(load_per_part) / min(load_per_part) > 1.25:
                    # Recompute the partition
                    current_partition = [x + offset for x in current_partition]
                    partitions[(prev_level, level)] = current_partition
                    prev_level = level
                    cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth, levels=[level])
                    edge_cut, current_partition = weighted_cell_partition(cell_graph, nparts=n_parts)
            current_partition = [x + offset for x in current_partition]
            partitions[(prev_level, levels[-1] + 1)] = current_partition
        elif mode == "predict":
            cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth, levels=[0])
            edge_cut, current_partition = weighted_cell_partition(cell_graph, nparts=n_parts)
            prev_level = 0

            level_size = len(levels) // level_chunks

            predictor = PredictWorkload(
                cells=len(self.data.geometry.cells),
                window_size=level_size,
            )

            cell_workload = [0 for _ in range(len(self.data.geometry.cells))]
            for i in levels:
                tasks = self.level_to_task[levels[i]]
                workload = [self.get_compute_cost(task_id, arch) for task_id in tasks]
                for j, task_id in enumerate(tasks):
                    cell_id = self.task_to_cell[task_id]
                    cell_workload[cell_id] += workload[j]
                predictor.submit_workload(cell_workload)

                # load_per_part = [0 for _ in range(n_parts)]
                # for task_id in self.level_to_task[i]:
                #     cell_id = self.task_to_cell[task_id]
                #     part = current_partition[cell_id]
                #     load_per_part[part] += self.get_compute_cost(task_id, arch)

                # # Check if the load imbalance is above a threshold
                # if max(load_per_part) / min(load_per_part) > 1.25:
                if i > 0 and i % level_size == 0:
                    current_partition = [x + offset for x in current_partition]
                    partitions[(prev_level, i)] = current_partition
                    prev_level = i
                    forecast = predictor.predict_workload(k=level_size)
                    for cell in cell_graph.cells:
                        scale = forecast[cell] / cell_graph.vweights[cell] if cell_graph.vweights[cell] > 0 else 1
                        # print(f"Scale for cell {cell}: {scale}")
                        cell_graph.vweights[cell] = forecast[cell]
                        start = cell_graph.xadj[cell]
                        end = cell_graph.xadj[cell + 1]
                        for j in range(start, end):
                            cell_graph.eweights[j] = int(cell_graph.eweights[j] * scale)
                    edge_cut, current_partition = weighted_cell_partition(cell_graph, nparts=n_parts)

            current_partition = [x + offset for x in current_partition]
            partitions[(prev_level, levels[-1] + 1)] = current_partition

        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.partitions = partitions
        self.align_partitions()
        return partitions

    def align_partitions(self):
        # Convert to numpy and check shapes
        sorted_keys = sorted(self.partitions.keys())
        aligned = [np.asarray(self.partitions[v], dtype=int) for v in sorted_keys]
        n = aligned[0].shape[0]
        if any(v.shape[0] != n for v in aligned):
            raise ValueError("All membership vectors must have the same length")

        # Find global K
        K = max(v.max() for v in aligned) + 1

        perms = [None] * len(aligned)
        flips = [0] * len(aligned)

        for i in range(1, len(aligned)):
            prev = aligned[i - 1]
            curr = aligned[i]

            # Build confusion via bincount on flattened indices
            idx = prev * K + curr
            cm = np.bincount(idx, minlength=K * K).reshape(K, K)

            # Solve max‐agreement assignment on -cm
            row_ind, col_ind = linear_sum_assignment(-cm)

            # Build a direct lookup array
            perm = np.arange(K, dtype=int)
            perm[col_ind] = row_ind
            perms[i] = perm

            # Apply mapping
            aligned[i] = perm[curr]

            # Count flips
            flips[i] = int((aligned[i] != prev).sum())

        # self.partitions = aligned
        for i, k in enumerate(sorted_keys):
            self.partitions[k] = aligned[i]

        return aligned, perms, flips


register_graph(JacobiGraph, JacobiConfig)


class JacobiVariant(VariantBuilder):
    @staticmethod
    def build_variant(arch: DeviceType, task: TaskTuple) -> Optional[VariantTuple]:
        memory_usage = 0
        vcu_usage = 1
        expected_time = 1000
        if arch == DeviceType.GPU:
            return VariantTuple(arch, memory_usage, vcu_usage, expected_time)
        else:
            return None


class PredictWorkload:
    def __init__(
        self,
        cells: int,
        window_size: Optional[int] = None,
        alpha: float = 0.3,
    ):
        """
        cells        : number of parallel series (cells)
        window_size  : max number of past datapoints to keep per cell (for fallback, not needed for EMA)
        alpha        : smoothing factor for EMA (0 < alpha <= 1)
        """
        self.cells = cells
        self.window_size = window_size
        self.alpha = alpha

        # Buffers retained if you still want to keep history; otherwise you can drop this.
        if window_size is not None:
            self.buffers: List[deque[float]] = [deque(maxlen=window_size) for _ in range(cells)]
        else:
            self.buffers: List[List[float]] = [[] for _ in range(cells)]

        # Initialize EMA values to None (will be set on first submit)
        self.ema_values: List[Optional[float]] = [None] * cells

    def submit_workload(self, workloads: List[float]):
        """
        Append the new workloads and update EMA for each cell.
        """
        if len(workloads) != self.cells:
            raise ValueError(f"Expected {self.cells} workloads, got {len(workloads)}")

        for i, w in enumerate(workloads):
            # keep history if desired
            self.buffers[i].append(w)

            # update EMA
            if self.ema_values[i] is None:
                # first data point: seed EMA
                self.ema_values[i] = w
            else:
                # EMA update rule
                self.ema_values[i] = self.alpha * w + (1.0 - self.alpha) * self.ema_values[i]

    def compute_next_k(self, i: int, k: int) -> List[float]:
        """
        Forecast k steps for cell i by projecting the current EMA forward.
        """
        ema = self.ema_values[i]
        assert ema is not None, f"No EMA for cell {i}, cannot predict."
        return [ema] * k

    def predict_workload(self, k: int) -> List[int]:
        """
        For each cell i, forecast the next k steps (using EMA) and
        return the integer total (sum over k).
        """
        preds: List[int] = []
        for i in range(self.cells):
            next_vals = self.compute_next_k(i, k)
            preds.append(int(sum(next_vals)))
        return preds


class GraphMETISMapper(StaticExternalMapper):

    def __init__(self, mapper: Optional[Self] = None, n_devices: int = 4, offset: int = 1, graph: Optional[ComputeDataGraph] = None, arch: DeviceType = DeviceType.GPU, bandwidth: int = 100e9):
        self.n_devices = n_devices
        self.offset = offset
        if mapper is not None:
            assert isinstance(mapper, GraphMETISMapper), "Mapper must be of type GraphMETISMapper, is " + str(type(mapper))
            self.mapping_dict = mapper.mapping_dict
        elif graph is not None:
            task_to_local, adjacency_list, adj_starts, vweights, eweights = graph.get_weighted_graph(arch=arch, bandwidth=bandwidth, task_ids=None, symmetric=True)
            cut_counts, part = weighted_partition(n_devices, adjacency_list, adj_starts, vweights, eweights)
            print(f"GraphMETISMapper: cut counts {cut_counts}")
            for i in range(n_devices):
                print(f"GraphMETISMapper: partition {i} has {part.count(i)} tasks")
                # Get vertex sum per partition
                print(f"GraphMETISMapper: partition {i} vertex sum {sum(vweights[j] for j in range(len(part)) if part[j] == i)}")
            print(f"GraphMETISMapper: total tasks {len(task_to_local)}")
            self.mapping_dict = {i: device + self.offset for i, device in enumerate(part)}
        else:
            raise ValueError("Either mapper or graph must be provided for GraphMETISMapper")


class PartitionMapper:
    def __init__(
        self,
        mapper: Optional[Self] = None,
        cell_to_mapping: Optional[dict] = None,
        level_start: int = 0,
    ):
        if mapper is not None:
            assert isinstance(mapper, PartitionMapper), "Mapper must be of type PartitionMapper, is " + str(type(mapper))
            self.cell_to_mapping = mapper.cell_to_mapping

        elif cell_to_mapping is not None:
            self.cell_to_mapping = cell_to_mapping
        else:
            self.cell_to_mapping = {}

        self.level_start = level_start

    def set_mapping_dict(self, cell_to_mapping):
        self.cell_to_mapping = cell_to_mapping

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        candidates = torch.zeros((simulator.observer.graph_spec.max_candidates), dtype=torch.int64)
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            local_id = i
            graph = simulator.input.graph
            assert isinstance(graph, JacobiGraph)
            level = graph.task_to_level[global_task_id]
            cell_id = graph.task_to_cell[global_task_id]
            device = self.cell_to_mapping[cell_id]
            if level < self.level_start:
                device = np.random.randint(1, 4)
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(global_task_id)
            mapping_result.append(fastsim.Action(local_id, device, mapping_priority, mapping_priority))
        return mapping_result


class BlockCyclicMapper(PartitionMapper):
    def __init__(self, mapper: Optional[Self] = None, geometry: Optional[Geometry] = None, n_devices: int = 4, block_size: int = 2, offset: int = 1):
        self.level_start = 0
        self.offset = offset
        self.geometry = geometry
        if mapper is not None:
            assert isinstance(mapper, BlockCyclicMapper), "Mapper must be of type BlockCyclicMapper, is " + str(type(mapper))
            self.cell_to_mapping = mapper.cell_to_mapping
        elif geometry is not None:
            x_dev = n_devices // 2
            y_dev = n_devices // 2
            if x_dev + y_dev != n_devices:
                x_dev += 1
            n_cells = len(geometry.cells)
            partition = block_cyclic(geometry, n_row_parts=x_dev, n_col_parts=y_dev, parts_per_column=block_size, parts_per_row=block_size)
            self.cell_to_mapping = {cell: device + self.offset for cell, device in enumerate(partition)}
        else:
            raise ValueError("Either mapper or geometry must be provided for BlockCyclicMapper")


class LevelPartitionMapper:
    def __init__(
        self,
        mapper: Optional[Self] = None,
        level_cell_mapping: dict[tuple[int, int] : list[int]] = None,
    ):
        if mapper is not None:
            self.level_cell_mapping = mapper.level_cell_mapping

        elif level_cell_mapping is not None:
            self.level_cell_mapping = level_cell_mapping
        else:
            self.level_cell_mapping = {}

    def set_mapping_dict(self, level_cell_mapping):
        self.level_cell_mapping = level_cell_mapping

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        graph: JacobiGraph = simulator.input.graph
        assert isinstance(graph, JacobiGraph)

        candidates = torch.zeros((simulator.observer.graph_spec.max_candidates), dtype=torch.int64)
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            level = graph.task_to_level[global_task_id]
            cell_id = graph.task_to_cell[global_task_id]
            device = -1
            for k, v in self.level_cell_mapping.items():
                if k[0] <= level < k[1]:
                    device = v[cell_id]
                    break
            if device == -1:
                print(self.level_cell_mapping)
            assert device != -1, f"Device not found for task {global_task_id} at level {level}"
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(global_task_id)
            mapping_result.append(fastsim.Action(i, device, mapping_priority, mapping_priority))
        return mapping_result


class JacobiRoundRobinMapper:
    def __init__(self, n_devices: int = 4, setting: int = 0, offset: int = 1, mapper: Optional[Self] = None):
        """
        Initialize the JacobiRoundRobinMapper.
        setting == 1 : Row cyclic
        setting == 0 : Checker board
        """
        self.n_devices = n_devices
        self.setting = setting
        self.offset = offset

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        graph: JacobiGraph = simulator.input.graph
        assert isinstance(graph, JacobiGraph)
        candidates = torch.zeros((simulator.observer.graph_spec.max_candidates), dtype=torch.int64)
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        ny = graph.ny
        nx = graph.nx
        stride = ny * nx

        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            idx = global_task_id % stride
            row = idx // nx
            col = idx % nx
            if self.setting == 0:
                # Checkerboard-style mapping
                if self.n_devices == 2:
                    # Classic 2-color checkerboard
                    device = (row + col) & 1
                elif self.n_devices == 4:
                    # 2x2 tiled checkerboard: devices 1..4 repeat like
                    # 1 2
                    # 3 4
                    device = (row & 1) * 2 + (col & 1)
                else:
                    # General fallback: diagonal stripes that still alternate locally
                    device = (row + col) % self.n_devices
            else:
                # Previous round-robin behavior (row-major)
                device = (row * nx + col) % self.n_devices
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(global_task_id)
            mapping_result.append(fastsim.Action(i, device + self.offset, mapping_priority, mapping_priority))
        return mapping_result


class JacobiQuadrantMapper:
    def __init__(
        self,
        n_devices: int,
        graph: JacobiGraph,
        offset: int = 1,
        mapper: Optional[Self] = None,
    ):
        self.n_devices = n_devices
        self.width = graph.nx
        self.length = graph.ny
        self.n_tasks = self.width * self.width
        self.graph = graph
        self.offset = offset

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        graph: JacobiGraph = simulator.input.graph
        assert isinstance(graph, JacobiGraph)
        candidates = torch.zeros((simulator.observer.graph_spec.max_candidates), dtype=torch.int64)
        num_candidates = simulator.simulator.get_mappable_candidates(candidates)
        mapping_result = []
        for i in range(num_candidates):
            global_task_id = candidates[i].item()
            x = global_task_id % self.n_tasks // self.width // (self.width // 2)
            y = global_task_id % self.n_tasks % self.width // (self.width // 2)
            device = x * 2 + y + self.offset
            mapping_priority = simulator.simulator.get_state().get_mapping_priority(global_task_id)
            mapping_result.append(fastsim.Action(i, device, mapping_priority, mapping_priority))
        return mapping_result


class JacobiVariantGPUOnly(VariantBuilder):
    @staticmethod
    def build_variant(arch: DeviceType, task: TaskTuple) -> Optional[VariantTuple]:
        memory_usage = 0
        vcu_usage = 1
        expected_time = 1000
        if arch == DeviceType.GPU:
            return VariantTuple(arch, memory_usage, vcu_usage, expected_time)
        else:
            return None


@dataclass(kw_only=True)
class XYExternalObserver(ExternalObserver):
    def data_observation(self, output):
        super().data_observation(output)
        graph: JacobiGraph = self.simulator.input.graph
        data: JacobiData = graph.data

        count = output["nodes"]["data"]["count"][0]
        for i, id in enumerate(output["nodes"]["data"]["glb"][:count]):
            id = int(id)
            datakey = data.get_key(id)
            if isinstance(datakey.id, Cell):
                datakey = datakey.id.id
            elif isinstance(datakey.id, tuple):
                datakey = datakey.id[0].id
            else:
                datakey = datakey.object.id
            centroid = graph.data.geometry.get_centroid(datakey)

            # Assume last two entries are x, y coordinates
            output["nodes"]["data"]["attr"][i][-2] = centroid[0]
            output["nodes"]["data"]["attr"][i][-1] = centroid[1]


@dataclass(kw_only=True)
class XYNormalizedDeviceQueueObserver(XYExternalObserver):
    def device_observation(self, output: TensorDict):
        super().device_observation(output)

        count = output["nodes"]["devices"]["count"][0]

        # Assume each device feature vector is only duration queue lengths
        with torch.no_grad():
            max_length = 0
            for i in range(count):
                total_queue_length = output["nodes"]["devices"]["attr"][i].sum()
                if total_queue_length > max_length:
                    max_length = total_queue_length

            if max_length > 0:
                for i in range(count):
                    output["nodes"]["devices"]["attr"][i] /= max_length


# @dataclass(kw_only=True)
# class XYHeterogeneousObserver(HeterogeneousExternalObserver):
#     def data_observation(self, output):
#         super().data_observation(output)
#         graph: JacobiGraph = self.simulator.input.graph
#         data: JacobiData = graph.data

#         count = output["nodes"]["data"]["count"][0]
#         for i, id in enumerate(output["nodes"]["data"]["glb"][:count]):
#             id = int(id)
#             datakey = data.get_key(id)
#             if isinstance(datakey.id, Cell):
#                 datakey = datakey.id.id
#             elif isinstance(datakey.id, tuple):
#                 datakey = datakey.id[0].id
#             else:
#                 datakey = datakey.object.id
#             centroid = graph.data.geometry.get_centroid(datakey)

#             # Assume last two entries are x, y coordinates
#             output["nodes"]["data"]["attr"][i][-2] = centroid[0]
#             output["nodes"]["data"]["attr"][i][-1] = centroid[1]

#     def device_observation(self, output: TensorDict):
#         super().device_observation(output)

#         count = output["nodes"]["devices"]["count"][0]

#         # Assume last three entries are queue lengths (mapped, reserved, and launched)
#         with torch.no_grad():
#             max_length = 0
#             for i in range(count):
#                 total_queue_length = output["nodes"]["devices"]["attr"][i][-3:].sum()
#                 if total_queue_length > max_length:
#                     max_length = total_queue_length

#             if max_length > 0:
#                 for i in range(count):
#                     output["nodes"]["devices"]["attr"][i][-3:] /= max_length


@dataclass(kw_only=True)
class XYExternalObserverFactory(ExternalObserverFactory):
    def create(self, simulator: SimulatorDriver):
        state = simulator.get_state()
        graph_spec = self.graph_spec
        graph_extractor = self.graph_extractor_t(state)
        task_feature_extractor = self.task_feature_factory.create(state)
        data_feature_extractor = self.data_feature_factory.create(state)
        device_feature_extractor = self.device_feature_factory.create(state)
        task_task_feature_extractor = self.task_task_feature_factory.create(state)
        task_data_feature_extractor = self.task_data_feature_factory.create(state)
        task_device_feature_extractor = self.task_device_feature_factory.create(state) if self.task_device_feature_factory is not None else None
        data_device_feature_extractor = self.data_device_feature_factory.create(state) if self.data_device_feature_factory is not None else None

        return XYNormalizedDeviceQueueObserver(
            simulator,
            graph_spec,
            graph_extractor,
            task_feature_extractor,
            data_feature_extractor,
            device_feature_extractor,
            task_task_feature_extractor,
            task_data_feature_extractor,
            task_device_feature_extractor,
            data_device_feature_extractor,
        )


# @dataclass(kw_only=True)
# class XYExternalHeterogeneousObserverFactory(ExternalObserverFactory):
#     def create(self, simulator: SimulatorDriver):
#         state = simulator.get_state()
#         graph_spec = self.graph_spec
#         graph_extractor = self.graph_extractor_t(state)
#         task_feature_extractor = self.task_feature_factory.create(state)
#         data_feature_extractor = self.data_feature_factory.create(state)
#         device_feature_extractor = self.device_feature_factory.create(state)
#         task_task_feature_extractor = self.task_task_feature_factory.create(state)
#         task_data_feature_extractor = self.task_data_feature_factory.create(state)
#         task_device_feature_extractor = (
#             self.task_device_feature_factory.create(state)
#             if self.task_device_feature_factory is not None
#             else None
#         )
#         data_device_feature_extractor = (
#             self.data_device_feature_factory.create(state)
#             if self.data_device_feature_factory is not None
#             else None
#         )

#         return XYHeterogeneousObserver(
#             simulator,
#             graph_spec,
#             graph_extractor,
#             task_feature_extractor,
#             data_feature_extractor,
#             device_feature_extractor,
#             task_task_feature_extractor,
#             task_data_feature_extractor,
#             task_device_feature_extractor,
#             data_device_feature_extractor,
#         )


@dataclass(kw_only=True)
class CandidateExternalObserverFactory(ExternalObserverFactory):
    def create(self, simulator: SimulatorDriver):
        state = simulator.get_state()
        graph_spec = self.graph_spec
        graph_extractor = self.graph_extractor_t(state)
        task_feature_extractor = self.task_feature_factory.create(state)
        data_feature_extractor = self.data_feature_factory.create(state)
        device_feature_extractor = self.device_feature_factory.create(state)
        task_task_feature_extractor = self.task_task_feature_factory.create(state)
        task_data_feature_extractor = self.task_data_feature_factory.create(state)
        task_device_feature_extractor = self.task_device_feature_factory.create(state) if self.task_device_feature_factory is not None else None
        data_device_feature_extractor = self.data_device_feature_factory.create(state) if self.data_device_feature_factory is not None else None

        return CandidateObserver(
            simulator,
            graph_spec,
            graph_extractor,
            task_feature_extractor,
            data_feature_extractor,
            device_feature_extractor,
            task_task_feature_extractor,
            task_data_feature_extractor,
            task_device_feature_extractor,
            data_device_feature_extractor,
        )


# class XYHeterogeneousObserverFactory(XYExternalHeterogeneousObserverFactory):
#     def __init__(self, spec: fastsim.GraphSpec):
#         graph_extractor_t = fastsim.GraphExtractor
#         task_feature_factory = FeatureExtractorFactory()
#         task_feature_factory.add(fastsim.DepthTaskFeature)
#         # task_feature_factory.add(fastsim.InDegreeTaskFeature)
#         # task_feature_factory.add(fastsim.OutDegreeTaskFeature)
#         task_feature_factory.add(fastsim.TaskStateFeature)

#         data_feature_factory = FeatureExtractorFactory()
#         data_feature_factory.add(fastsim.DataSizeFeature)
#         data_feature_factory.add(fastsim.EmptyDataFeature, 2)

#         device_feature_factory = FeatureExtractorFactory()
#         device_feature_factory.add(fastsim.DeviceIDFeature)
#         device_feature_factory.add(fastsim.DeviceTimeFeature)

#         task_task_feature_factory = EdgeFeatureExtractorFactory()
#         task_task_feature_factory.add(fastsim.TaskTaskDefaultEdgeFeature)

#         task_data_feature_factory = EdgeFeatureExtractorFactory()
#         task_data_feature_factory.add(fastsim.TaskDataDefaultEdgeFeature)

#         task_device_feature_factory = EdgeFeatureExtractorFactory()
#         task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

#         data_device_feature_factory = EdgeFeatureExtractorFactory()
#         data_device_feature_factory.add(fastsim.DataDeviceDefaultEdgeFeature)

#         super().__init__(
#             spec,
#             graph_extractor_t,
#             task_feature_factory,
#             data_feature_factory,
#             device_feature_factory,
#             task_task_feature_factory,
#             task_data_feature_factory,
#             task_device_feature_factory,
#             data_device_feature_factory,
#         )


class XYObserverFactory(XYExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec):
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(fastsim.DepthTaskFeature)
        task_feature_factory.add(fastsim.TagTaskFeature)
        # task_feature_factory.add(fastsim.TaskStateFeature)

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.ScaledDataMappedLocationsFeature)
        data_feature_factory.add(fastsim.EmptyDataFeature, 2)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(fastsim.DeviceTimeFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.TaskTaskDefaultEdgeFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.TaskDataDefaultEdgeFeature)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = EdgeFeatureExtractorFactory()
        data_device_feature_factory.add(fastsim.DataDeviceDefaultEdgeFeature)

        super().__init__(
            spec,
            graph_extractor_t,
            task_feature_factory,
            data_feature_factory,
            device_feature_factory,
            task_task_feature_factory,
            task_data_feature_factory,
            task_device_feature_factory,
            data_device_feature_factory,
        )


# class XYMinimalObserverFactory(XYExternalObserverFactory):
#     def __init__(self, spec: fastsim.GraphSpec):
#         graph_extractor_t = fastsim.GraphExtractor
#         task_feature_factory = FeatureExtractorFactory()
#         # task_feature_factory.add(fastsim.InDegreeTaskFeature)
#         # task_feature_factory.add(fastsim.OutDegreeTaskFeature)
#         # task_feature_factory.add(fastsim.TaskStateFeature)
#         # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
#         task_feature_factory.add(
#             fastsim.EmptyTaskFeature, 1
#         )  # 2 for x, y position, last for whether it is mapped

#         data_feature_factory = FeatureExtractorFactory()
#         data_feature_factory.add(fastsim.DataSizeFeature)
#         data_feature_factory.add(fastsim.EmptyDataFeature, 2)
#         # data_feature_factory.add(fastsim.DataMappedLocationsFeature)

#         device_feature_factory = FeatureExtractorFactory()
#         # device_feature_factory.add(fastsim.DeviceArchitectureFeature)
#         device_feature_factory.add(fastsim.DeviceIDFeature)
#         # device_feature_factory.add(fastsim.DeviceMemoryFeature)
#         device_feature_factory.add(fastsim.DeviceTimeFeature)

#         task_task_feature_factory = EdgeFeatureExtractorFactory()
#         task_task_feature_factory.add(fastsim.TaskTaskSharedDataFeature)

#         task_data_feature_factory = EdgeFeatureExtractorFactory()
#         task_data_feature_factory.add(fastsim.TaskDataRelativeSizeFeature)
#         # task_data_feature_factory.add(fastsim.TaskDataUsageFeature)

#         task_device_feature_factory = EdgeFeatureExtractorFactory()
#         task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

#         data_device_feature_factory = None

#         super().__init__(
#             spec,
#             graph_extractor_t,
#             task_feature_factory,
#             data_feature_factory,
#             device_feature_factory,
#             task_task_feature_factory,
#             task_data_feature_factory,
#             task_device_feature_factory,
#             data_device_feature_factory,
#         )


class CandidateObserverFactory(CandidateExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec):
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        task_feature_factory.add(fastsim.CandidateVectorFeature)
        # task_feature_factory.add(fastsim.TaskDeviceMappedTimeFeature)
        # task_feature_factory.add(fastsim.TaskDataMappedLocationsFeature)
        # task_feature_factory.add(fastsim.InDegreeTaskFeature)
        # #task_feature_factory.add(fastsim.StandardizedGPUDurationTaskFeature)
        # task_feature_factory.add(fastsim.StandardizedInputOutputTaskFeature)

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.EmptyDataFeature, 1)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(fastsim.EmptyDeviceFeature, 1)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.EmptyTaskTaskFeature, 1)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.EmptyTaskDataFeature, 1)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = EdgeFeatureExtractorFactory()
        data_device_feature_factory.add(fastsim.DataDeviceDefaultEdgeFeature)

        super().__init__(
            spec,
            graph_extractor_t,
            task_feature_factory,
            data_feature_factory,
            device_feature_factory,
            task_task_feature_factory,
            task_data_feature_factory,
            task_device_feature_factory,
            data_device_feature_factory,
        )


class CandidateCoordinateObserverFactory(CandidateExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec, version: str, batched: bool = False):
        self.batched = batched
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        # task_feature_factory.add(fastsim.CandidateVectorFeature)
        if "A" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "B" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "C" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "D" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "E" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "F" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "G" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "H" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, 1)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.EmptyDataFeature, 1)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(fastsim.EmptyDeviceFeature, 1)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.EmptyTaskTaskFeature, 1)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.EmptyTaskDataFeature, 1)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = EdgeFeatureExtractorFactory()
        data_device_feature_factory.add(fastsim.DataDeviceDefaultEdgeFeature)

        super().__init__(
            spec,
            graph_extractor_t,
            task_feature_factory,
            data_feature_factory,
            device_feature_factory,
            task_task_feature_factory,
            task_data_feature_factory,
            task_device_feature_factory,
            data_device_feature_factory,
        )


@dataclass(kw_only=True)
class CnnTaskObserverFactory(ExternalObserverFactory):
    def __init__(
        self,
        spec: fastsim.GraphSpec,
        width: int,
        length: int,
        prev_frames: int,
        version: str,
        batched: bool = False,
    ):
        self.batched = batched
        assert (not batched and spec.max_candidates == 1) or (
            spec.max_candidates == width * length
        ), f"Batched {self.batched} CNN observer requires max_candidates to be {width*length if self.batched else 1}, but got {spec.max_candidates}"
        task_feature_factory = FeatureExtractorFactory()

        if "A" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "B" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "C" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "D" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "E" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "F" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            # task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "G" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        elif "H" in version:
            task_feature_factory.add(fastsim.PrevReadSizeFeature, 0, 0, True, prev_frames)
            task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)
            task_feature_factory.add(fastsim.TaskCoordinatesFeature)
            task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)

        # task_feature_factory.add(fastsim.TaskMeanDurationFeature)
        # task_feature_factory.add(fastsim.CandidateVectorFeature)
        # task_feature_factory.add(fastsim.TaskDataMappedSizeFeature)

        # if prev_frames > 0:
        #     task_feature_factory.add(fastsim.PrevReadSizeFeature, width, length, True, prev_frames)  # CNN-A,B,C
        # task_feature_factory.add(fastsim.PrevMappedDeviceFeature, width, length, False, prev_frames) # CNN-B
        # if prev_frames > 0:
        #     task_feature_factory.add(
        #         fastsim.PrevMappedSizeFeature, width, False, prev_frames
        #     )
        # if not batched:
        #     # Difference in depth doesn't exist in batched
        #     task_feature_factory.add(fastsim.DepthTaskFeature)
        #     # Tag candidate only when it is not batched
        #     task_feature_factory.add(fastsim.EmptyTaskFeature, 1)

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.EmptyDataFeature, 1)

        device_feature_factory = FeatureExtractorFactory()
        device_feature_factory.add(fastsim.EmptyDeviceFeature, 1)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.EmptyTaskTaskFeature, 1)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.EmptyTaskDataFeature, 1)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = EdgeFeatureExtractorFactory()
        data_device_feature_factory.add(fastsim.DataDeviceDefaultEdgeFeature)

        super().__init__(
            spec,
            fastsim.GraphExtractor,
            task_feature_factory,
            data_feature_factory,
            device_feature_factory,
            task_task_feature_factory,
            task_data_feature_factory,
            task_device_feature_factory,
            data_device_feature_factory,
        )

    def create(self, simulator: SimulatorDriver):
        state = simulator.get_state()
        graph_spec = self.graph_spec
        graph_extractor = self.graph_extractor_t(state)
        task_feature_extractor = self.task_feature_factory.create(state)
        data_feature_extractor = self.data_feature_factory.create(state)
        device_feature_extractor = self.device_feature_factory.create(state)
        task_task_feature_extractor = self.task_task_feature_factory.create(state)
        task_data_feature_extractor = self.task_data_feature_factory.create(state)
        task_device_feature_extractor = self.task_device_feature_factory.create(state) if self.task_device_feature_factory is not None else None
        data_device_feature_extractor = self.data_device_feature_factory.create(state) if self.data_device_feature_factory is not None else None
        if self.batched:
            return CnnBatchTaskObserver(
                simulator,
                graph_spec,
                graph_extractor,
                task_feature_extractor,
                data_feature_extractor,
                device_feature_extractor,
                task_task_feature_extractor,
                task_data_feature_extractor,
                task_device_feature_extractor,
                data_device_feature_extractor,
            )
        else:
            return CnnSingleTaskObserver(
                simulator,
                graph_spec,
                graph_extractor,
                task_feature_extractor,
                data_feature_extractor,
                device_feature_extractor,
                task_task_feature_extractor,
                task_data_feature_extractor,
                task_device_feature_extractor,
                data_device_feature_extractor,
            )
