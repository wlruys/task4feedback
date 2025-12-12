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
from task4feedback import trip as trip
from ..interface.wrappers import *
from scipy.optimize import linear_sum_assignment
import sympy
from ..interface.types import _bytes_to_readable
import numpy as np

from collections import deque
import math

@dataclass
class CholeskyConfig(GraphConfig):
    width: int = 4  # number of blocks
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

    def __post_init__(self):
        self.length = self.width


class CholeskyData(DataGeometry):
    @staticmethod
    def from_mesh(geometry: Geometry, config: CholeskyConfig, system: Optional[System] = None):
        return CholeskyData(geometry, config, system)
    
    def _initialize_cell_to_ij(self):
        self.cell_to_ij = {}
        #Loop over all cells and assign (i, j) based on position in the grid
        n = self.width
        for k in range(len(self.geometry.cells)):
            centroid = self.geometry.get_centroid(k)
            centroid = np.floor(centroid * n)
            i = int(centroid[0])
            j = int(centroid[1])
            self.cell_to_ij[k] = (i, j)

        self.ij_to_cell = {v: k for k, v in self.cell_to_ij.items()}

    

    def _create_blocks(self, system: Optional[System] = None):

        # Create a block A[i, j] for every lower trangular entry in the matrix
        n = self.config.width

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
        return self.config.width

    @property
    def ny(self) -> int:
        """
        Only works for rectangular grids.
        """
        return self.config.length
    
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
        n = self.config.width
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

        if variant is not None:
            self.apply_variant(variant)
        elif system is not None:
            self._apply_workload_variant(system)
        else:
            print("Warning: No variant or system provided, using default Jacobi variant for task time and architecture specs.")
            self.apply_variant(CholeskyVariant)

        self.finalize()

    def _apply_workload_variant(self, system: System):

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
        n_lower_triangular_cells = self.config.width * (self.config.width + 1) // 2
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
        raise NotImplementedError("get_weighted_cell_graph is not implemented for CholeskyGraph")

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
        raise NotImplementedError("get_distributed_weighted_graph is not implemented for CholeskyGraph")

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
       

    def align_partitions(self):
        raise NotImplementedError("align_partitions is not implemented for CholeskyGraph")

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

