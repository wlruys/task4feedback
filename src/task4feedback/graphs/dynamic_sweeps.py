from .mesh.base import *
from .mesh.partition import *
from .base import *
from typing import Callable, Optional, Self
from collections import defaultdict
from .sweep import *


@dataclass
class DynamicSweepConfig(SweepConfig):
    start_workload: int = 1000
    upper_workload: int = 3000
    lower_workload: int = 500
    step_size: int = 2000
    correlation_scale: float = 0.1
    data_compute_relation: Callable[[int], int] = lambda x: x


class DynamicSweepData(SweepData):
    @staticmethod
    def from_mesh(
        geometry: Geometry,
        config: SweepConfig = SweepConfig(),
        workload: DynamicWorkload = None,
    ):
        return DynamicSweepData(geometry, config, workload)

    def __init__(
        self,
        geometry: Geometry,
        config: DynamicSweepConfig = DynamicSweepConfig(),
        workload: DynamicWorkload = None,
    ):
        super(SweepData, self).__init__(geometry, DataBlocks(), GeometryIDMap())
        self.workload = workload
        self.config = config
        self._create_blocks(directions=config.directions)

    def _create_blocks(
        self,
        directions: Optional[np.ndarray] = None,
    ):
        if directions is None:
            directions = np.array([[-1, -1], [1, 1]])

        directions = directions.astype(np.float64)

        for i, direction in enumerate(directions):
            directions[i] = direction / np.linalg.norm(direction)

        self.directions = directions

        self.centroid_projections = []
        dim = len(directions) + 1

        base_workload = self.config.start_workload

        for i, direction in enumerate(directions):
            self.centroid_projections.append(np.zeros(len(self.geometry.cells)))

        for k in range(self.config.steps + 1):
            # Loop over cells
            for cell in range(len(self.geometry.cells)):
                workload = self.workload.get_workload(k)[cell]
                workload_ratio = workload / base_workload
                new_data_ratio = self.config.data_compute_relation(workload_ratio)
                new_data_size = int(self.config.interior_size * new_data_ratio)

                # Create 1 shared block per cell
                self.add_block(
                    DataKey(Cell(cell), k * dim + 0),
                    size=new_data_size,
                    location=0,
                )

            for i, direction in enumerate(directions):
                # print(f"Creating blocks in direction {i}: {direction}")

                # Loop over cells
                for cell in range(len(self.geometry.cells)):
                    workload = self.workload.get_workload(k)[cell]
                    workload_ratio = workload / base_workload
                    new_data_ratio = self.config.data_compute_relation(workload_ratio)
                    new_data_size = int(self.config.interior_size * new_data_ratio)

                    # Create 1 block per cell per direction
                    self.add_block(
                        DataKey(Cell(cell), k * dim + i + 1),
                        size=new_data_size,
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

                            workload = self.workload.get_workload(k)[cell]
                            workload_ratio = workload / base_workload
                            new_data_ratio = self.config.data_compute_relation(
                                workload_ratio
                            )
                            new_data_size = int(
                                self.config.boundary_size * new_data_ratio
                            )

                            self.add_block(
                                DataKey(Edge(edge), (Cell(cell), k * dim + i + 1)),
                                size=new_data_size,
                                location=0,
                            )

    def idx_at_step(self, step: int) -> int:
        return step * (len(self.directions) + 1)


class DynamicSweepGraph(SweepGraph):
    def __init__(self, geometry: Geometry, config: DynamicSweepConfig):
        self.workload = DynamicWorkload(geometry)
        self.workload.generate_initial_mass(
            distribution=lambda x: 1.0, average_workload=config.start_workload
        )
        self.workload.generate_workload(
            num_levels=config.steps + 1,
            step_size=config.step_size,
            scale=config.correlation_scale,
            lower_bound=config.lower_workload,
            upper_bound=config.upper_workload,
        )
        super(SweepGraph, self).__init__()
        self.config = config
        self.data = DynamicSweepData.from_mesh(geometry, config, self.workload)
        self._build_graph()
        self._apply_workload_variant()

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

            # Read shared block (material properties, etc)
            shared_block_in = self.data.get_shared_block(Cell(i), step=step)
            read_blocks.append(shared_block_in)

            shared_block_out = self.data.get_shared_block(Cell(i), step=step + 1)
            write_blocks.append(shared_block_out)

            for d in range(len(self.data.directions)):
                # Read interior block for this step
                interior_block = self.data.get_block_at_direction(Cell(i), d, step=step)
                read_blocks.append(interior_block)

                # Read all edges, skip blocks that don't exist
                edges = self.data.geometry.cell_edges[i]
                for edge in edges:
                    block = self.data.get_block_at_direction(
                        (Cell(i), Edge(edge)), d, step=step
                    )
                    if isinstance(block, int):
                        read_blocks.append(block)

                    next_block = self.data.get_block_at_direction(
                        (Cell(i), Edge(edge)), d, step=step + 1
                    )
                    if isinstance(next_block, int):
                        write_blocks.append(next_block)

            self.add_read_data(task_id, read_blocks)
            self.add_write_data(task_id, write_blocks)

    def _apply_workload_variant(self):
        task_to_level = self.task_to_level
        task_to_cell = self.task_to_cell
        task_to_type = self.task_to_type
        task_to_direction = self.task_to_direction

        class DynamicSweepVariant(SweepVariant):
            @staticmethod
            def build_variant(
                arch: DeviceType, task: TaskTuple
            ) -> Optional[VariantTuple]:
                memory_usage = 0
                vcu_usage = 1
                level = task_to_level[task.id]
                cell = task_to_cell[task.id]
                direction = task_to_direction[task.id]
                task_type = task_to_type[task.id]

                workload = self.workload.get_workload(level)[cell]
                workload = int(workload)

                if arch == DeviceType.GPU:
                    return VariantTuple(arch, memory_usage, vcu_usage, workload)
                else:
                    return None

        self.apply_variant(DynamicSweepVariant)
