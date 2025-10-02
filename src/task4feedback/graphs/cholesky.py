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
class CholeskyConfig(GraphConfig):
    n: int = 4  # number of blocks
    arithmetic_intensity: float = 1.0
    arithmetic_complexity: float = 1.0
    domain_ratio: float = 1.0  # height/width ratio of the domain
    memory_intensity: float = 1.0
    boundary_width: float = 5.0
    boundary_complexity: float = 0.5
    level_memory: int = 1000000
    randomness: float = 0.0
    permute_idx: int = 0
    task_time: Optional[int] = None
    block_time: Optional[int] = None
    compute_time: Optional[int] = None
    vcu_usage: float = 1.0
    task_internal_memory: int = 0
    bytes_per_element: int = 4  # Assuming float32 data type
    verbose: bool = True
    boundary_in_memory_calc: bool = True



class CholeskyData(DataGeometry):
    @staticmethod
    def from_mesh(geometry: Geometry, config: CholeskyConfig, system: Optional[System] = None):
        return CholeskyData(geometry, config, system)
    
    def _initialize_cell_to_ij(self):
        self.cell_to_ij = {}
        #Loop over all cells and assign (i, j) based on position in the grid
        n = self.config.n
        for k in range(len(self.geometry.cells)):
            centroid = self.geometry.get_centroid(k)
            centroid = np.floor(centroid * n)
            i = int(centroid[0])
            j = int(centroid[1])
            self.cell_to_ij[k] = (i, j)

        self.ij_to_cell = {v: k for k, v in self.cell_to_ij.items()}

    

    def _create_blocks(self, system: Optional[System] = None):

        # Create a block A[i, j] for every lower trangular entry in the matrix
        n = self.config.n

        #Overkill for just dividing two numbers, but keeping impl same as Jacobi for consistency
        total_blocks = n * (n + 1) // 2
        y = sympy.symbols('y', real=True, positive=True)
        memory_eq = total_blocks * y  - self.config.level_memory / self.config.bytes_per_element
        solution = sympy.solve(memory_eq, y)
        y_value = solution[0].evalf()
        block_elem = int(y_value)
        block_size = block_elem * self.config.bytes_per_element
        block_size = max(block_size, 1)
        block_size = int(block_size)
        self.block_size = block_size

        if self.config.verbose:
            print(f"Cholesky Graph with {n}x{n} blocks ({total_blocks} total blocks)")
            print(f"Each block has size {block_size} elements ({_bytes_to_readable(block_size * self.config.bytes_per_element)}) to fit within {self.config.level_memory} bytes of level memory")

        if self.config.block_time is not None:
            block_size = system.fastest_bandwidth * self.config.block_time 
            if self.config.verbose:
                print("Overriding block size to ", block_size, " elements based on block_time and system bandwidth")
                print("Total memory required for all resized blocks: ", _bytes_to_readable(total_blocks * block_size * self.config.bytes_per_element))

        if self.config.compute_time is not None:
            raise NotImplementedError("compute_time is not implemented for Cholesky graph")
        
        block_size = int(block_size)
        self._initialize_cell_to_ij()

        for i in range(n):
            for j in range(i+1):
                cell_id = self.ij_to_cell[(i, j)]
                centroid = self.geometry.get_centroid(cell_id)
                centroid_x = centroid[0]
                centroid_y = centroid[1]

                self.add_block(
                    DataKey(Cell(cell_id), 0),
                    size=block_size,
                    location=0,
                    x=centroid_x,
                    y=centroid_y,
                )

    def __init__(
        self,
        geometry: Geometry,
        config: CholeskyConfig = CholeskyConfig(),
        system: Optional[System] = None,
    ):
        super().__init__(geometry, DataBlocks(), GeometryIDMap())
        self.config = config
        self._create_blocks(system=system)

    def blocks_to_objects(self, blocks: list[int]):
        return [self.map.get_object(i) for i in blocks]

    def blocks_to_keys(self, blocks: list[int]):
        return [self.map.block_to_key[i] for i in blocks]


    def set_location(self, obj: Cell | Edge, location: int):
        id_list = self.map.key_to_block.get_leaves(obj)
        for i in id_list:
            self.blocks.set_location(i, location)

    def set_locations_from_list(self, location_list: list[int]):
        for i, location in enumerate(location_list):
            self.set_location(Cell(i), location)

    def randomize_locations(self, num_changes: int, location_list: list[int], **kwargs):
        new_locations = []

        lower_triangular_cells = [cell for cell, (i, j) in self.cell_to_ij.items() if i >= j]

        if num_changes > len(lower_triangular_cells):
            num_changes = len(lower_triangular_cells)

        selected_cells = random.sample(lower_triangular_cells, num_changes)

        for k, cell in enumerate(selected_cells):
            new_location = random.choice(location_list)
            self.set_location(Cell(cell), new_location)
            new_locations.append(new_location)

        return selected_cells, new_locations

    def get_locations(self, as_dict: bool = False) -> list[int] | dict[int, int]:
        lower_triangular_cells = [cell for cell, (i, j) in self.cell_to_ij.items() if i >= j]

        if not as_dict:
            # Return a list of locations for each cell
            #TODO(wlr): Need state that is not CPU for non-existent blocks
            locations = [0] * len(self.geometry.cells)
            for i in lower_triangular_cells:
                block_id = self.get_blocks(Cell(i))
                block_id = block_id[0]
                locations[i] = self.blocks.get_location(block_id)
            return locations

        locations = {}
        for i in range(len(self.geometry.cells)):
            locations[i] = 0 #TODO(wlr): Need state that is not CPU for non-existent blocks

        for i in lower_triangular_cells:
            block_id = self.get_blocks(Cell(i))
            block_id = block_id[0]
            locations[i] = self.blocks.get_location(block_id)
        return locations

    def remap_locations(self, location_map: dict[int, int]):
        cell_locations = self.get_locations()
        for cell_id, location in enumerate(cell_locations):

            #check if cell is in lower triangular part
            i, j = self.cell_to_ij[cell_id]
            if i < j:
                continue

            # Remap location if in map
            if location in location_map:
                new_location = location_map[location]
                self.set_location(Cell(cell_id), new_location)

    def permute_locations(self, location_map: dict[int, int], permutation_idx: Optional[int] = None):
        valid_locations = {
            self.blocks.get_location(self.get_blocks(Cell(cell))[0])
            for cell, (i, j) in self.cell_to_ij.items()
            if i >= j
        }

        filtered_keys = [k for k in location_map if k in valid_locations and location_map[k] in valid_locations]
        if not filtered_keys:
            return permutation_idx

        filtered_values = [location_map[k] for k in filtered_keys]
        all_perms = list(permutations(filtered_values))
        if not all_perms:
            return permutation_idx

        if permutation_idx is None:
            permutation_idx = random.randint(0, len(all_perms) - 1)
        else:
            permutation_idx %= len(all_perms)

        perm = all_perms[permutation_idx]
        perm_map = {filtered_keys[i]: perm[i] for i in range(len(filtered_keys))}
        if perm_map:
            self.remap_locations(perm_map)
        return permutation_idx


class CholeskyGraph(ComputeDataGraph):
    def xy_from_id(self, taskid: int) -> int:
        """
        Convert a task ID to its (x, y) coordinates in the Jacobi grid.
        And returns row-major order index.
        Only works for rectangular grids.
        """
        cell_id = self.task_to_cell[taskid]
        i, j = self.data.cell_to_ij[cell_id]
        return i, j 

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
    
    def _check_requirements(self, read_blocks: List[int], write_blocks: List[int], system: Optional[System] = None):
        data_req = 0
        for data_id in read_blocks:
            data_req += self.data.blocks.data.get_size(data_id)
        for data_id in write_blocks:
            if data_id not in read_blocks:
                data_req += self.data.blocks.data.get_size(data_id)

        assert (
            system is None or data_req < system.arch_to_maxmem[DeviceType.GPU]
        ), f"Task requires {data_req / 1e9:.2f} GB of data, which exceeds the maximum memory for GPU {system.arch_to_maxmem[DeviceType.GPU] / 1e9:.1f} GB"
        # Raise a warning if data_req exceeds half of maxmem
        if system is not None and data_req > system.arch_to_maxmem[DeviceType.GPU] / 2:
            print(f"Warning: Task requires {data_req / 1e9:.2f} GB of data, which exceeds half of the maximum memory for GPU {system.arch_to_maxmem[DeviceType.GPU] / 1e9:.1f} GB")
        self.max_requirement = max(self.max_requirement, data_req)

    def _create_syrk_task(self, j: int, k: int, system: Optional[System] = None):
        name = f"SYRK(j={j}, k={k})"
        task_id = self.add_task(name)

        self.task_to_cell[task_id] = self.data.ij_to_cell[(j, j)]
        self.task_to_type[task_id] = "SYRK"
        self.type_to_tasks["SYRK"].append(task_id)

        reads = [self.data.ij_to_cell[(j, k)], self.data.ij_to_cell[(j, j)]]
        writes = [self.data.ij_to_cell[(j, j)]]

        read_blocks = [self.data.get_blocks(Cell(r))[0] for r in reads]
        write_blocks = [self.data.get_blocks(Cell(w))[0] for w in writes]
        if self.config.verbose:
            print("Task ", task_id, " SYRK(", j, k, ") reads ", reads, " writes ", writes)
        self.add_read_data(task_id, read_blocks)
        self.add_write_data(task_id, write_blocks)
        self._check_requirements(read_blocks, write_blocks, system=system)
        self.task_list.append(task_id)

    def _create_gemm_task(self, i: int, j: int, k: int, system: Optional[System] = None):
        name = f"GEMM(i={i}, j={j}, k={k})"
        task_id = self.add_task(name)

        self.task_to_cell[task_id] = self.data.ij_to_cell[(j, i)]
        self.task_to_type[task_id] = "GEMM"
        self.type_to_tasks["GEMM"].append(task_id)

        reads = [self.data.ij_to_cell[(i, k)], self.data.ij_to_cell[(j, k)], self.data.ij_to_cell[(j, i)]]
        writes = [self.data.ij_to_cell[(j, i)]]

        read_blocks = [self.data.get_blocks(Cell(r))[0] for r in reads]
        write_blocks = [self.data.get_blocks(Cell(w))[0] for w in writes]
        if self.config.verbose:
            print("Task ", task_id, " GEMM(", i, j, k, ") reads ", reads, " writes ", writes)
        self.add_read_data(task_id, read_blocks)
        self.add_write_data(task_id, write_blocks)
        self._check_requirements(read_blocks, write_blocks, system=system)
        self.task_list.append(task_id)

    def _create_potrf_task(self, j: int, system: Optional[System] = None):
        name = f"POTRF(j={j})"
        task_id = self.add_task(name)

        self.task_to_cell[task_id] = self.data.ij_to_cell[(j, j)]
        self.task_to_type[task_id] = "POTRF"
        self.type_to_tasks["POTRF"].append(task_id)

        reads = [self.data.ij_to_cell[(j, j)]]
        writes = [self.data.ij_to_cell[(j, j)]]

        read_blocks = [self.data.get_blocks(Cell(r))[0] for r in reads]
        write_blocks = [self.data.get_blocks(Cell(w))[0] for w in writes]
        if self.config.verbose:
            print("Task ", task_id, " POTRF(", j, ") reads ", reads, " writes ", writes)
        self.add_read_data(task_id, read_blocks)
        self.add_write_data(task_id, write_blocks)
        self._check_requirements(read_blocks, write_blocks, system=system)
        self.task_list.append(task_id)

    def _create_solve_task(self, i: int, j: int, system: Optional[System] = None):
        name = f"SOLVE(i={i}, j={j})"
        task_id = self.add_task(name)

        self.task_to_cell[task_id] = self.data.ij_to_cell[(i, j)]
        self.task_to_type[task_id] = "SOLVE"
        self.type_to_tasks["SOLVE"].append(task_id)

        reads = [self.data.ij_to_cell[(j, j)], self.data.ij_to_cell[(i, j)]]
        writes = [self.data.ij_to_cell[(i, j)]]

        read_blocks = [self.data.get_blocks(Cell(r))[0] for r in reads]
        write_blocks = [self.data.get_blocks(Cell(w))[0] for w in writes]
        if self.config.verbose:
            print("Task ", task_id, " SOLVE(", i, j, ") reads ", reads, " writes ", writes)
        self.add_read_data(task_id, read_blocks)
        self.add_write_data(task_id, write_blocks)
        self._check_requirements(read_blocks, write_blocks, system=system)
        self.task_list.append(task_id)

    def _build_graph(self, retire_data: bool = False, system: System = None):
        self.task_to_cell = {}
        self.task_to_type = {}
        self.type_to_tasks = defaultdict(list)
        self.task_list = []

        self.max_requirement = 0
        n = self.config.n
        for k in range(n):
            self._create_potrf_task(k, system=system)

            for i in range(k + 1, n):
                self._create_solve_task(i, k, system=system)
            
            for i in range(k + 1, n):
                for j in range(i, n):
                    if i == j:
                        self._create_syrk_task(i, k, system=system)
                    self._create_gemm_task(i, j, k, system=system)


    def __init__(
        self,
        geometry: Geometry,
        config: CholeskyConfig,
        system: Optional[System] = None,
        variant: Optional[type[VariantBuilder]] = None,
    ):
        assert system is not None
        super(CholeskyGraph, self).__init__()
        self.data = CholeskyData.from_mesh(geometry, config, system=system)
        self.config = config
        self._build_graph()
        self.dynamic = False
        self.reference_partition = []
        # half = config.n // 2
        # for j in range(config.n):  # column-wise unrolling
        #     for i in range(config.n):
        #         if i < half and j < half:
        #             self.reference_partition.append(0)  # top-left
        #         elif i < half and j >= half:
        #             self.reference_partition.append(1)  # top-right
        #         elif i >= half and j < half:
        #             self.reference_partition.append(2)  # bottom-left
        #         else:
        #             self.reference_partition.append(3)  # bottom-right

        if variant is not None:
            self.apply_variant(variant)
        elif system is not None:
            self._apply_workload_variant(system)
        else:
            print("Warning: No variant or system provided, using default Jacobi variant for task time and architecture specs.")
            self.apply_variant(CholeskyVariant)

        self.finalize()

    def _apply_workload_variant(self, system: System):
        # print("Building custom variant for system", system)

        class CholeskyVariant(VariantBuilder):
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
                    num_elements = self.data.block_size // self.config.bytes_per_element
                    expected_work = num_elements**self.config.arithmetic_complexity * self.config.arithmetic_intensity
                    expected_time = expected_work / system.get_flop_ms(arch)

                    expected_memory = self.data.block_size * self.config.memory_intensity
                    expected_time = max(expected_time, expected_memory / system.get_gmbw_ms(arch))
                    expected_time = int(max(expected_time, 1))

                return VariantTuple(
                    arch,
                    memory_usage=memory_usage,
                    vcu_usage=vcu_usage,
                    expected_time=expected_time,
                )

        self.apply_variant(CholeskyVariant)

    def randomize_locations(
        self,
        perc_change: float,
        location_list: Optional[list[int]] = None,
        min_loc: int = 0,
        max_loc: Optional[int] = None,
        verbose: bool = False,
    ):
        n_lower_triangular_cells = self.config.n * (self.config.n + 1) // 2
        num_changes = int(perc_change * n_lower_triangular_cells)
        if verbose:
            print(f"Randomizing {num_changes} locations out of {n_lower_triangular_cells} blocks")
        if location_list is None:
            if max_loc is None:
                raise ValueError("max_loc must be provided if location_list is None")
            location_list = list(range(min_loc, max_loc))

        selected_cells, new_locations = self.data.randomize_locations(num_changes, location_list)

        if verbose:
            print(f"Randomized locations for {len(selected_cells)} blocks on step:")
            for cell, new_location in zip(selected_cells, new_locations):
                print(f"Cell {cell} -> New Location: {new_location}")

        return selected_cells, new_locations

    def set_cell_locations(self, location_list: list[int], **kwargs):
        self.data.set_locations_from_list(location_list)

    def set_cell_locations_from_dict(self, location_dict: dict[int, int], **kwargs):
        for cell, location in location_dict.items():
            self.data.set_location(Cell(cell), location)

    def get_cell_locations(self, as_dict: bool = True, **kwargs) -> list[int] | dict[int, int]:
        return self.data.get_locations(as_dict=as_dict)

    def get_mapping_from_locations(self, as_dict=True) -> list[int] | dict[int, int]:
        mapping = []
        if as_dict:
            mapping = {}
        else:
            mapping = [0] * len(self)

        for task in self:
                cell_id = self.task_to_cell[task.id]
                block_id = self.data.get_blocks(Cell(cell_id))[0]
                location = self.data.blocks.get_location(block_id)
                mapping[task.id] = location

        return mapping

    def get_num_iterations(self):
        return self.num_iterations

    def permute_locations(self, location_map: dict[int, int], permutation_idx: Optional[int] = None):
        return self.data.permute_locations(location_map, permutation_idx)

    def get_weighted_cell_graph(self, arch: DeviceType, bandwidth=1000):
        """
        Build a weighted graph where each vertex is a cell (block) and edges represent communication between cells. 
        Summed over the whole graph (not temporally).
        """
        raise NotImplementedError("get_weighted_cell_graph is not implemented for CholeskyGraph")

        task_to_local, adj_list, adj_starts, vweights, eweights = self.get_weighted_graph(arch, bandwidth=bandwidth, task_ids=self.task_list, symmetric=True)

        cell_vertex_cost = defaultdict(int)
        cell_neighbors_cost = defaultdict(lambda: defaultdict(int))

        for task_id in self.task_list:
            local_task_id = task_id
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

        
        #Loop over graph to make it symmetric
        edge_dict = defaultdict(int)
        for i in range(len(adj_list)):
            src = cells[i]
            dst = adj_list[i]
            weight = eweights[i]
            if (dst, src) in edge_dict:
                edge_dict[(dst, src)] += weight
            else:
                edge_dict[(src, dst)] += weight
            
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

        raise NotImplementedError("get_distributed_weighted_graph is not implemented for CholeskyGraph")

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
        cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth)
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
        raise NotImplementedError("mincut_per_levels is not implemented for CholeskyGraph")
        # Oracle mode takes in level chunks and returns partitions based on the full knowledge of the workload
        partitions = {}
        levels = list(self.level_to_task.keys())
        levels = sorted(levels)

        if mode == "metis":
            level_size = len(levels) // level_chunks

            if levels_per_chunk is not None:
                level_size = levels_per_chunk
                level_chunks = (len(levels)) // level_size  # ceil div

            for i in range(level_chunks):
                start = i * level_size
                end = (i + 1) * level_size

                if i == level_chunks - 1:
                    end = len(levels)

                levels_to_compute = levels[start:end]
                cell_graph = self.get_weighted_cell_graph(arch, bandwidth=bandwidth, levels=levels_to_compute)

                edge_cut, partition = weighted_cell_partition(cell_graph, nparts=n_parts)
                if i == 0:
                    partition = self.maximize_matches(partition)
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
        raise NotImplementedError("align_partitions is not implemented for CholeskyGraph")
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

    def maximize_matches(self, list2):
        """
        Relabel `list2` to best match `self.reference_partition` using a Hungarian
        assignment on the confusion matrix (maximizing agreement).

        Returns
        -------
        aligned : list[int]
            `list2` with labels permuted to best align with the reference.
        perm : list[int]
            Lookup array such that aligned = perm[list2]. Maps labels in `list2`
            to labels in the reference.
        flips : int
            Number of positions where aligned != reference (mismatches after alignment).
        """
        ref = np.asarray(self.reference_partition, dtype=int).ravel()
        cur = np.asarray(list2, dtype=int).ravel()

        if ref.shape != cur.shape:
            raise ValueError("Both membership vectors must have the same length.")
        if ref.size == 0:
            return [], [], 0
        if ref.min() < 0 or cur.min() < 0:
            raise ValueError("Labels must be non-negative integers (0..K-1).")

        # Global K across both labelings
        K = int(max(ref.max(), cur.max())) + 1

        # Confusion matrix via bincount over flattened pair indices
        idx = ref * K + cur
        cm = np.bincount(idx, minlength=K * K).reshape(K, K)

        # Max-agreement assignment
        row_ind, col_ind = linear_sum_assignment(-cm)

        # Build label mapping: map each label in `cur` (columns) -> label in `ref` (rows)
        perm = np.arange(K, dtype=int)
        perm[col_ind] = row_ind

        # Apply mapping
        aligned = perm[cur]

        # Count mismatches ("flips" relative to ref)
        flips = int((aligned != ref).sum())

        return aligned.tolist()


register_graph(CholeskyGraph, CholeskyConfig)


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

