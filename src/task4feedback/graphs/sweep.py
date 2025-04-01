from .mesh.base import Geometry, Cell, Edge
from ..interface import DataBlocks, Graph, DeviceType, TaskTuple, VariantTuple
from .base import DataGeometry, DataKey, GeometryIDMap, ComputeDataGraph
from dataclasses import dataclass
from ..interface.lambdas import VariantBuilder
import random
from typing import Optional
from itertools import permutations
from collections import defaultdict
import torch
from typing import Self
from task4feedback import fastsim2 as fastsim
from ..interface.wrappers import *


@dataclass
class SweepConfig:
    L: int = 4
    n: int = 4
    randomness: float = 0
    permute_idx: int = 0
    directions: Optional[np.ndarray] = None
    threshold: float = 0.1
    steps: int = 1
    round_in: int = 2
    round_out: int = 2
    reduce: bool = True
    interior_size: int = 1000000
    boundary_size: int = 1000000


class SweepData(DataGeometry):
    @staticmethod
    def from_mesh(geometry: Geometry, config: SweepConfig = SweepConfig()):
        return SweepData(geometry, config)

    def _create_blocks(
        self,
        interior_size: int = 1000000,
        boundary_size: int = 1000000,
        directions: Optional[np.ndarray] = None,
    ):
        if directions is None:
            directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            directions = np.array([[-1, -1], [1, 1]])

        directions = directions.astype(np.float64)

        for i, direction in enumerate(directions):
            directions[i] = direction / np.linalg.norm(direction)

        self.directions = directions

        self.centroid_projections = []

        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 1 shared block per cell
            self.add_block(
                DataKey(Cell(cell), 0), size=self.config.interior_size, location=0
            )

        for i, direction in enumerate(directions):
            # print(f"Creating blocks in direction {i}: {direction}")

            self.centroid_projections.append(np.zeros(len(self.geometry.cells)))

            # Loop over cells
            for cell in range(len(self.geometry.cells)):
                # Create 1 block per cell per direction
                self.add_block(
                    DataKey(Cell(cell), i + 1),
                    size=self.config.interior_size,
                    location=0,
                )

                for edge in self.geometry.cell_edges[cell]:
                    normal_to_edge = self.geometry.get_normal_to_edge(
                        edge,
                        cell,
                        round_in=self.config.round_in,
                        round_out=self.config.round_out,
                    )
                    prod_with_direction = np.dot(normal_to_edge, direction)

                    centroid = self.geometry.get_centroid(
                        cell, round_out=self.config.round_out
                    )
                    self.centroid_projections[i][cell] = np.dot(centroid, direction)

                    # print(f"Normal to edge {edge}: {normal_to_edge}, {prod_with_direction}")
                    if prod_with_direction > self.config.threshold:
                        # Create boundary blocks for each edge

                        self.add_block(
                            DataKey(Edge(edge), (Cell(cell), i + 1)),
                            size=self.config.boundary_size,
                            location=0,
                        )

                    # print(f"Creating block for (cell, edge) {(cell, edge)} in direction {i}: {direction}")
                    # print(f"Product with direction: {prod_with_direction}")

    def __init__(self, geometry: Geometry, config: SweepConfig = SweepConfig()):
        super().__init__(geometry, DataBlocks(), GeometryIDMap())
        self.config = config
        self._create_blocks(directions=config.directions)

    def blocks_to_objects(self, blocks: list[int]):
        return [self.map.get_object(i) for i in blocks]

    def blocks_to_keys(self, blocks: list[int]):
        return [self.map.block_to_key[i] for i in blocks]

    def get_block_at_direction(self, object: Cell | tuple[Cell, Edge], direction: int):
        idx = direction + 1
        if isinstance(object, tuple):
            return self.map.get_block(DataKey(object[1], (object[0], idx)))
        return self.map.get_block(DataKey(object, idx))

    def get_shared_block(self, object: Cell | tuple[Cell, Edge]):
        idx = 0
        if isinstance(object, tuple):
            return self.map.get_block(DataKey(object[1], (object[0], idx)))
        return self.map.get_block(DataKey(object, idx))

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


class SweepGraph(ComputeDataGraph):
    def _build_sweep_tasks(self, step: int = 0, direction: int = 0):
        # Note this assumes that dependencies will only occur in the order of the cells sorted by the centroid projections
        # This is not true for all geometries
        # Likewise this (should) prevent most cycles from occuring

        direction_vec = self.data.directions[direction]
        centroid_projections = self.data.centroid_projections[direction]
        n_cells = len(self.data.geometry.cells)

        sort_cell_idx = np.argsort(centroid_projections)

        for i, idx in enumerate(sort_cell_idx):
            cell = idx
            edges = self.data.geometry.cell_edges[cell]

            # Create task that reads its upstream edges and writes to its downstream edges

            name = f"Task(SWEEP, Cell({cell}), {step}, {direction_vec})"
            task_id = self.add_task(name, i + direction * n_cells)

            self.task_to_cell[task_id] = cell
            self.task_to_level[task_id] = step
            self.task_to_type[task_id] = "SWEEP"
            self.task_to_direction[task_id] = direction

            self.type_to_task["SWEEP"].append(task_id)
            self.level_to_task[step]["SWEEP"].append(task_id)
            self.direction_to_task["SWEEP"][direction].append(task_id)

            read_blocks = []
            write_blocks = []

            # Read shared block (material properties, etc)
            shared_block = self.data.get_shared_block(Cell(cell))
            read_blocks.append(shared_block)

            # Read and write to own interior block for this step
            interior_block = self.data.get_block_at_direction(Cell(cell), direction)
            read_blocks.append(interior_block)
            write_blocks.append(interior_block)

            # Read upstream edges
            for edge in edges:
                normal_to_edge = self.data.geometry.get_normal_to_edge(
                    edge,
                    cell,
                    round_in=self.config.round_in,
                    round_out=self.config.round_out,
                )
                prod_with_direction = np.dot(normal_to_edge, direction_vec)

                for neighbor_cell in self.data.geometry.edge_cell_dict[edge]:
                    if neighbor_cell != cell:
                        if prod_with_direction < -self.config.threshold:
                            # Read from upstream neighbor
                            block = self.data.get_block_at_direction(
                                (Cell(neighbor_cell), Edge(edge)), direction
                            )

                            if isinstance(block, int):
                                read_blocks.append(block)

                        if prod_with_direction > self.config.threshold:
                            # Read and write to all self edges
                            block = self.data.get_block_at_direction(
                                (Cell(cell), Edge(edge)), direction
                            )
                            if isinstance(block, int):
                                read_blocks.append(block)
                                write_blocks.append(block)

            self.add_read_data(task_id, read_blocks)
            self.add_write_data(task_id, write_blocks)

    def _build_reduction_tasks(self, step: int = 0):
        # For each cell, create a task that reads the interior blocks for all directions and reads/writes to the shared block

        for i in range(len(self.data.geometry.cells)):
            # Create task that reads its upstream edges and writes to its downstream edges

            name = f"Task(REDUCE, Cell({i}), {step})"
            task_id = self.add_task(name, i + step * len(self.data.geometry.cells))

            self.task_to_cell[task_id] = i
            self.task_to_level[task_id] = step
            self.task_to_type[task_id] = "REDUCE"
            self.task_to_direction[task_id] = -1
            self.direction_to_task["REDUCE"][-1].append(task_id)
            self.type_to_task["REDUCE"].append(task_id)
            self.level_to_task[step]["REDUCE"].append(task_id)

            read_blocks = []
            write_blocks = []

            # Read/write shared block (material properties, etc)
            shared_block = self.data.get_shared_block(Cell(i))
            read_blocks.append(shared_block)
            write_blocks.append(shared_block)

            for d in range(len(self.data.directions)):
                # Read interior block for this step
                interior_block = self.data.get_block_at_direction(Cell(i), d)
                read_blocks.append(interior_block)

                # Read all edges, skip blocks that don't exist
                edges = self.data.geometry.cell_edges[i]
                for edge in edges:
                    block = self.data.get_block_at_direction((Cell(i), Edge(edge)), d)
                    if isinstance(block, int):
                        read_blocks.append(block)
                    # else:
                    #     print("Block does not exist", block, Cell(i), Edge(edge), d)

            self.add_read_data(task_id, read_blocks)
            self.add_write_data(task_id, write_blocks)

    def _build_graph(self):
        types = ["SWEEP", "REDUCE"]

        def make_type_dict():
            return defaultdict(list)

        self.task_to_cell = {}
        self.task_to_level = {}
        self.task_to_type = {}
        self.type_to_task = defaultdict(list)
        self.level_to_task = defaultdict(make_type_dict)
        self.direction_to_task = defaultdict(make_type_dict)
        self.task_to_direction = {}

        for s in range(self.config.steps):
            for d, direction in enumerate(self.data.directions):
                self._build_sweep_tasks(step=s, direction=d)

            if self.config.reduce:
                self._build_reduction_tasks(step=s)

        self.fill_data_flow_dependencies()

    def __init__(self, geometry: Geometry, config: SweepConfig = SweepConfig()):
        super(SweepGraph, self).__init__()
        data = SweepData.from_mesh(geometry, config)
        self.data = data
        self.config = config
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

    def permute_locations(
        self, location_map: dict[int, int], permutation_idx: Optional[int] = None
    ):
        return self.data.permute_locations(location_map, permutation_idx)


class SweepVariant(VariantBuilder):
    @staticmethod
    def build_variant(arch: DeviceType, task: TaskTuple) -> Optional[VariantTuple]:
        memory_usage = 0
        vcu_usage = 1
        expected_time = 1000
        if arch == DeviceType.GPU:
            return VariantTuple(arch, memory_usage, vcu_usage, expected_time)
        else:
            return None


class PartitionMapper:
    def __init__(
        self,
        mapper: Optional[Self] = None,
        cell_to_mapping: Optional[dict] = None,
        level_start: int = 0,
    ):
        if mapper is not None:
            assert isinstance(mapper, PartitionMapper), (
                "Mapper must be of type PartitionMapper, is " + str(type(mapper))
            )
            self.cell_to_mapping = mapper.cell_to_mapping

        elif cell_to_mapping is not None:
            self.cell_to_mapping = cell_to_mapping
        else:
            self.cell_to_mapping = {}

        self.level_start = level_start

    def set_mapping_dict(self, cell_to_mapping):
        self.cell_to_mapping = cell_to_mapping

    def map_tasks(self, simulator: "SimulatorDriver") -> list[fastsim.Action]:
        candidates = torch.zeros((1), dtype=torch.int64)
        simulator.simulator.get_mappable_candidates(candidates)
        global_task_id = candidates[0].item()
        local_id = 0
        graph = simulator.input.graph
        assert isinstance(graph, SweepGraph)
        level = graph.task_to_level[global_task_id]

        cell_id = graph.task_to_cell[global_task_id]

        device = self.cell_to_mapping[cell_id]

        if level < self.level_start:
            device = np.random.randint(1, 4)

        # print(global_task_id, cell_id, device)
        state = simulator.simulator.get_state()
        mapping_priority = state.get_mapping_priority(global_task_id)
        return [fastsim.Action(local_id, device, mapping_priority, mapping_priority)]


class LevelPartitionMapper:
    def __init__(
        self, mapper: Optional[Self] = None, level_cell_mapping: Optional[dict] = None
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
        candidates = torch.zeros((1), dtype=torch.int64)
        simulator.simulator.get_mappable_candidates(candidates)
        global_task_id = candidates[0].item()
        local_id = 0
        graph = simulator.input.graph
        assert isinstance(graph, SweepGraph)
        level = graph.task_to_level[global_task_id]
        cell_id = graph.task_to_cell[global_task_id]
        device = self.level_cell_mapping[level][cell_id]
        state = simulator.simulator.get_state()
        mapping_priority = state.get_mapping_priority(global_task_id)
        return [fastsim.Action(local_id, device, mapping_priority, mapping_priority)]


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
    def task_observation(
        self, output: TensorDict, task_ids: Optional[torch.Tensor] = None
    ):
        graph = self.simulator.input.graph
        if task_ids is None:
            n_candidates = output["aux"]["candidates"]["count"][0]
            task_ids = output["aux"]["candidates"]["idx"][:n_candidates]
            output["nodes"]["tasks"]["attr"][:n_candidates, -1] = 1
        _, count = self.get_bidirectional_neighborhood(
            task_ids, output["nodes"]["tasks"]["glb"]
        )
        output["nodes"]["tasks"]["count"][0] = count
        self.get_task_features(
            output["nodes"]["tasks"]["glb"][:count], output["nodes"]["tasks"]["attr"]
        )
        for i, id in enumerate(output["nodes"]["tasks"]["glb"][:count]):
            id = int(id)
            cell_id = graph.task_to_cell[id]
            centroid = graph.data.geometry.cell_points[
                graph.data.geometry.cells[cell_id]
            ].mean(axis=0)
            centroid = np.round(centroid, 2)
            output["nodes"]["tasks"]["attr"][i][-3] = centroid[0] / 4
            output["nodes"]["tasks"]["attr"][i][-2] = centroid[1] / 4


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
        task_device_feature_extractor = (
            self.task_device_feature_factory.create(state)
            if self.task_device_feature_factory is not None
            else None
        )
        data_device_feature_extractor = (
            self.data_device_feature_factory.create(state)
            if self.data_device_feature_factory is not None
            else None
        )

        return XYExternalObserver(
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


class XYObserverFactory(XYExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec):
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        # task_feature_factory.add(fastsim.InDegreeTaskFeature)
        # task_feature_factory.add(fastsim.OutDegreeTaskFeature)
        # task_feature_factory.add(fastsim.TaskStateFeature)
        task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(
            fastsim.EmptyTaskFeature, 3
        )  # 2 for x, y position, last for whether it is mapped

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.DataSizeFeature)
        # data_feature_factory.add(fastsim.DataMappedLocationsFeature)

        device_feature_factory = FeatureExtractorFactory()
        # device_feature_factory.add(fastsim.DeviceArchitectureFeature)
        device_feature_factory.add(fastsim.DeviceIDFeature)
        # device_feature_factory.add(fastsim.DeviceMemoryFeature)
        device_feature_factory.add(fastsim.DeviceTimeFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.TaskTaskSharedDataFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.TaskDataRelativeSizeFeature)
        # task_data_feature_factory.add(fastsim.TaskDataUsageFeature)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = None

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


class XYMinimalObserverFactory(XYExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec):
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        # task_feature_factory.add(fastsim.InDegreeTaskFeature)
        # task_feature_factory.add(fastsim.OutDegreeTaskFeature)
        # task_feature_factory.add(fastsim.TaskStateFeature)
        # task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(
            fastsim.EmptyTaskFeature, 3
        )  # 2 for x, y position, last for whether it is mapped

        data_feature_factory = FeatureExtractorFactory()
        data_feature_factory.add(fastsim.DataSizeFeature)
        # data_feature_factory.add(fastsim.DataMappedLocationsFeature)

        device_feature_factory = FeatureExtractorFactory()
        # device_feature_factory.add(fastsim.DeviceArchitectureFeature)
        device_feature_factory.add(fastsim.DeviceIDFeature)
        # device_feature_factory.add(fastsim.DeviceMemoryFeature)
        device_feature_factory.add(fastsim.DeviceTimeFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.TaskTaskSharedDataFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        task_data_feature_factory.add(fastsim.TaskDataRelativeSizeFeature)
        # task_data_feature_factory.add(fastsim.TaskDataUsageFeature)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = None

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


class XYDataObserverFactory(XYExternalObserverFactory):
    def __init__(self, spec: fastsim.GraphSpec):
        graph_extractor_t = fastsim.GraphExtractor
        task_feature_factory = FeatureExtractorFactory()
        # task_feature_factory.add(fastsim.InDegreeTaskFeature)
        # task_feature_factory.add(fastsim.OutDegreeTaskFeature)
        # task_feature_factory.add(fastsim.TaskStateFeature)
        task_feature_factory.add(fastsim.OneHotMappedDeviceTaskFeature)
        task_feature_factory.add(
            fastsim.EmptyTaskFeature, 3
        )  # 2 for x, y position, last for whether it is mapped

        data_feature_factory = FeatureExtractorFactory()
        # data_feature_factory.add(fastsim.DataSizeFeature)
        data_feature_factory.add(fastsim.DataMappedLocationsFeature)

        device_feature_factory = FeatureExtractorFactory()
        # device_feature_factory.add(fastsim.DeviceArchitectureFeature)
        device_feature_factory.add(fastsim.DeviceIDFeature)
        # device_feature_factory.add(fastsim.DeviceMemoryFeature)
        device_feature_factory.add(fastsim.DeviceTimeFeature)

        task_task_feature_factory = EdgeFeatureExtractorFactory()
        task_task_feature_factory.add(fastsim.TaskTaskSharedDataFeature)

        task_data_feature_factory = EdgeFeatureExtractorFactory()
        # task_data_feature_factory.add(fastsim.TaskDataRelativeSizeFeature)
        task_data_feature_factory.add(fastsim.TaskDataUsageFeature)

        task_device_feature_factory = EdgeFeatureExtractorFactory()
        task_device_feature_factory.add(fastsim.TaskDeviceDefaultEdgeFeature)

        data_device_feature_factory = None

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
