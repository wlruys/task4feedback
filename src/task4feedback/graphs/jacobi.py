from .mesh.base import Geometry, Cell, Edge
from ..interface import DataBlocks, Graph, DeviceType, TaskTuple, VariantTuple
from .base import DataGeometry, DataKey, GeometryIDMap, ComputeDataGraph
from dataclasses import dataclass
from ..interface.lambdas import VariantBuilder
import random
from typing import Optional
from itertools import permutations
from collections import defaultdict


class JacobiData(DataGeometry):
    @staticmethod
    def from_mesh(geometry: Geometry):
        return JacobiData(geometry)

    def _create_blocks(
        self, interior_size: int = 1000000, boundary_size: int = 1000000
    ):
        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 2 data blocks per cell
            for i in range(2):
                self.add_block(DataKey(Cell(cell), i), size=interior_size, location=0)

            # Create 2 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                for i in range(2):
                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), i)),
                        size=boundary_size,
                        location=0,
                    )

    def __init__(self, geometry: Geometry):
        super().__init__(geometry, DataBlocks(), GeometryIDMap())
        self._create_blocks()

    def blocks_to_objects(self, blocks: list[int]):
        return [self.map.get_object(i) for i in blocks]

    def blocks_to_keys(self, blocks: list[int]):
        return [self.map.block_to_key[i] for i in blocks]

    def get_block_at_step(self, object: Cell | tuple[Cell, Edge], step: int):
        idx = step % 2
        if isinstance(object, tuple):
            return self.map.get_block(DataKey(object[1], (object[0], idx)))
        return self.map.get_block(DataKey(object, idx))

    def idx_at_step(self, step: int):
        return step % 2

    def set_location(self, obj: Cell | Edge, location: int):
        id_list = self.map.key_to_block.get_leaves(obj)
        for i in id_list:
            self.blocks.set_location(i, location)

        if isinstance(obj, Cell):
            # Update edges as well
            for edge in self.geometry.cell_edges[obj.id]:
                id_list = self.map.key_to_block.get_leaves(DataKey(Edge(edge), (obj,)))
                for i in id_list:
                    self.blocks.set_location(i, location)

    def set_locations_from_list(self, location_list: list[int]):
        for i, location in enumerate(location_list):
            self.set_location(Cell(i), location)

    def randomize_locations(self, num_changes: int, location_list: list[int]):
        new_locations = []

        selected_cells = random.sample(range(len(self.geometry.cells)), num_changes)
        for i, cell in enumerate(selected_cells):
            new_location = random.choice(location_list)
            self.set_location(Cell(cell), new_location)
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

    def permute_locations(
        self, location_map: dict[int, int], permutation_idx: Optional[int] = None
    ):
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
    def _build_graph(self):
        self.task_to_cell = {}
        self.task_to_level = {}
        self.level_to_task = defaultdict(list)
        for i in range(self.num_iterations):
            for j, (cell, edges) in enumerate(self.data.geometry.cell_edges.items()):
                # Create task that:
                # -reads all of its block (interior and edges) and the edges of its neighbors
                # -writes to blocks of its self (interior and edges)

                idx = self.data.idx_at_step(i)

                name = f"Task(Cell({cell}), {i})"
                task_id = self.add_task(name, j)

                self.task_to_cell[task_id] = cell
                self.task_to_level[task_id] = i
                self.level_to_task[i].append(task_id)

                # print(f"Task {task_id} created with name {name}")

                interior_block = self.data.get_block_at_step(Cell(cell), i)
                interior_edges = []
                exterior_edges = []
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
                    next_interior_edges.append(
                        self.data.get_block_at_step((Cell(cell), Edge(edge)), i + 1)
                    )

                read_blocks = interior_edges + exterior_edges + [interior_block]
                write_blocks = next_interior_edges + [next_interior_block]

                self.add_read_data(task_id, read_blocks)
                self.add_write_data(task_id, write_blocks)

        self.fill_data_flow_dependencies()

        # print tasks and dependencies
        # for task in self:
        #   print(task)

    def __init__(self, geometry: Geometry, num_iterations: int):
        super(JacobiGraph, self).__init__()
        data = JacobiData.from_mesh(geometry)
        self.data = data
        self.task_to_cell = {}
        self.num_iterations = num_iterations
        self._build_graph()

    def randomize_locations(
        self,
        perc_change: float,
        location_list: Optional[list[int]] = None,
        min_loc: int = 0,
        max_loc: Optional[int] = None,
        verbose: bool = False,
    ):
        num_changes = int(perc_change * len(self.data.geometry.cells))
        if verbose:
            print(
                f"Randomizing {num_changes} locations out of {len(self.data.geometry.cells)}"
            )
        if location_list is None:
            if max_loc is None:
                raise ValueError("max_loc must be provided if location_list is None")
            location_list = list(range(min_loc, max_loc))

        selected_cells, new_locations = self.data.randomize_locations(
            num_changes, location_list
        )

        if verbose:
            print(f"Randomized locations for {len(selected_cells)} cells:")
            for cell, new_location in zip(selected_cells, new_locations):
                print(f"Cell {cell} -> New Location: {new_location}")

        return selected_cells, new_locations

    def set_cell_locations(self, location_list: list[int]):
        self.data.set_locations_from_list(location_list)

    def set_cell_locations_from_dict(self, location_dict: dict[int, int]):
        for cell, location in location_dict.items():
            self.data.set_location(Cell(cell), location)

    def get_cell_locations(self, as_dict: bool = True) -> list[int] | dict[int, int]:
        return self.data.get_locations(as_dict=as_dict)

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


@dataclass
class JacobiConfig:
    L: int = 4
    n: int = 4 
    steps: int = 1
    n_part: int = 4
    randomness: float = 0
    permute_idx: int = 0
    
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