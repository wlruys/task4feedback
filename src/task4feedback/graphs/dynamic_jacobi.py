from .mesh.base import *
from .mesh.partition import *
from .base import *
from typing import Callable, Optional, Self
from collections import defaultdict
from .jacobi import *


@dataclass
class DynamicJacobiConfig(JacobiConfig):
    workload_args: dict = field(default_factory=dict)
    workload_class: Type = TrajectoryWorkload
    data_compute_relation: Callable[[int], int] = lambda x: x
    start_workload: int = 1000


class DynamicJacobiData(JacobiData):
    @staticmethod
    def from_mesh(
        geometry: Geometry, config: DynamicJacobiConfig, workload: DynamicWorkload
    ) -> Self:
        data = DynamicJacobiData(geometry, config, workload)
        return data

    def __init__(
        self,
        geometry: Geometry,
        config: DynamicJacobiConfig = DynamicJacobiConfig(),
        workload: DynamicWorkload = None,
    ):
        self.workload = workload
        super().__init__(geometry, config)

    def idx_at_step(self, step: int) -> int:
        return step

    def _create_blocks(self):
        interior_size = self.config.interior_size
        boundary_size = int(self.config.boundary_interior_ratio * interior_size)

        base_workload = self.config.start_workload

        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 2 data blocks per cell
            for i in range(self.config.steps + 1):
                workload = self.workload.get_workload(i)[cell]
                workload_ratio = workload / base_workload
                new_data_ratio = self.config.data_compute_relation(workload_ratio)
                new_data_size = int(interior_size * new_data_ratio)

                self.add_block(DataKey(Cell(cell), i), size=new_data_size, location=0)

            # Create 2 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                for i in range(self.config.steps + 1):
                    workload = self.workload.get_workload(i)[cell]
                    workload_ratio = workload / base_workload
                    new_data_ratio = self.config.data_compute_relation(workload_ratio)
                    new_data_size = int(boundary_size * new_data_ratio)

                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), i)),
                        size=new_data_size,
                        location=0,
                    )


class DynamicJacobiGraph(JacobiGraph):
    def __init__(self, geometry: Geometry, config: DynamicJacobiConfig):
        workload_class = config.workload_class
        self.workload = workload_class(geometry)
        self.workload.generate_initial_mass(
            distribution=lambda x: 1.0, average_workload=config.start_workload
        )
        self.workload.generate_workload(config.steps, **config.workload_args)
        super(JacobiGraph, self).__init__()
        self.config = config
        self.data = DynamicJacobiData.from_mesh(geometry, config, self.workload)
        self._build_graph(retire_data=True)
        self._apply_workload_variant()

    def _apply_workload_variant(self):
        task_to_level = self.task_to_level
        task_to_cell = self.task_to_cell

        class DynamicJacobiVariant(JacobiVariant):
            @staticmethod
            def build_variant(
                arch: DeviceType, task: TaskTuple
            ) -> Optional[VariantTuple]:
                memory_usage = 0
                vcu_usage = 1
                level = task_to_level[task.id]
                cell = task_to_cell[task.id]

                workload = self.workload.get_workload(level)[cell]
                workload = int(workload)

                if arch == DeviceType.GPU:
                    return VariantTuple(arch, memory_usage, vcu_usage, workload)
                else:
                    return None

        self.apply_variant(DynamicJacobiVariant)
