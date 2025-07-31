from .mesh.base import *
from .mesh.partition import *
from .base import *
from typing import Callable, Optional, Self
from collections import defaultdict
from .jacobi import *
from .base import register_graph


@dataclass
class DynamicJacobiConfig(JacobiConfig):
    workload_args: dict = field(default_factory=dict)
    workload_class: Type = TrajectoryWorkload
    start_workload: int = 1000
    level_chunks: int = 1


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
        self.config: DynamicJacobiConfig

    def idx_at_step(self, step: int) -> int:
        return step

    def _create_blocks(self):
        interior_data = []
        boundary_data = []
        step_data_sum = [0 for _ in range(self.config.steps)]
        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 2 data blocks per cell
            for i in range(self.config.steps + 1):
                workload = self.workload.get_workload(i)[cell]
                new_data_size = int(
                    workload * self.config.interior_compute_ratio * self.config.d2d_bw
                )

                self.add_block(DataKey(Cell(cell), i), size=new_data_size, location=0)
                assert (
                    new_data_size > 0 or i == self.config.steps
                ), "Interior data size must be positive "
                if new_data_size > 0:
                    interior_data.append(new_data_size)
                    step_data_sum[i] += new_data_size

            # Create 8 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                for i in range(self.config.steps + 1):
                    workload = self.workload.get_workload(i)[cell]
                    new_data_size = int(
                        workload
                        * self.config.d2d_bw
                        * self.config.boundary_compute_ratio
                    )

                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), i)),
                        size=new_data_size,
                        location=0,
                    )
                    assert (
                        new_data_size > 0 or i == self.config.steps
                    ), "Boundary data size must be positive"
                    if new_data_size > 0:
                        boundary_data.append(new_data_size)
                        step_data_sum[i] += new_data_size
        self.data_stat = {
            "interior_average": sum(interior_data) / len(interior_data),
            "interior_minimum": min(interior_data),
            "interior_maximum": max(interior_data),
            "boundary_average": sum(boundary_data) / len(boundary_data),
            "boundary_minimum": min(boundary_data),
            "boundary_maximum": max(boundary_data),
            "average_step_data": sum(step_data_sum) / len(step_data_sum),
        }

    def reset_data_size(self):
        """
        Reset the data size of all blocks to a new trajectory.
        """
        interior_data = []
        boundary_data = []
        step_data_sum = [0 for _ in range(self.config.steps)]
        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 2 data blocks per cell
            for i in range(self.config.steps + 1):
                workload = self.workload.get_workload(i)[cell]
                new_data_size = int(
                    workload * self.config.interior_compute_ratio * self.config.d2d_bw
                )

                self.blocks.set_size(
                    self.map.get_block(DataKey(Cell(cell), i)), new_data_size
                )
                assert (
                    new_data_size > 0 or i == self.config.steps
                ), "Interior data size must be positive "
                if new_data_size > 0:
                    interior_data.append(new_data_size)
                    step_data_sum[i] += new_data_size

            # Create 8 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                for i in range(self.config.steps + 1):
                    workload = self.workload.get_workload(i)[cell]
                    new_data_size = int(
                        workload
                        * self.config.d2d_bw
                        * self.config.boundary_compute_ratio
                    )

                    if workload > 0:
                        new_data_size = max(new_data_size, 1)

                    self.blocks.set_size(
                        self.map.get_block(DataKey(Edge(edge), (Cell(cell), i))),
                        new_data_size,
                    )

                    assert (
                        new_data_size > 0 or i == self.config.steps
                    ), "Boundary data size must be positive"
                    if new_data_size > 0:
                        boundary_data.append(new_data_size)
                        step_data_sum[i] += new_data_size
        self.data_stat = {
            "interior_average": sum(interior_data) / len(interior_data),
            "interior_minimum": min(interior_data),
            "interior_maximum": max(interior_data),
            "boundary_average": sum(boundary_data) / len(boundary_data),
            "boundary_minimum": min(boundary_data),
            "boundary_maximum": max(boundary_data),
            "average_step_data": sum(step_data_sum) / len(step_data_sum),
        }


class DynamicJacobiGraph(JacobiGraph):
    def __init__(
        self,
        geometry: Geometry,
        config: DynamicJacobiConfig,
        variant: Optional[VariantBuilder] = None,
    ):
        workload_class = config.workload_class
        self.workload = workload_class(geometry)
        self.workload.generate_initial_mass(
            distribution=lambda x: 1.0, average_workload=config.start_workload
        )
        print(config.workload_args)
        self.workload.generate_workload(config.steps, **config.workload_args)
        super(JacobiGraph, self).__init__()
        self.config = config
        self.data: DynamicJacobiData = DynamicJacobiData.from_mesh(
            geometry, config, self.workload
        )
        self._build_graph(retire_data=True)
        self._apply_workload_variant()
        self.finalize()

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

                workload = int(self.workload.get_workload(level)[cell])

                if arch == DeviceType.GPU:
                    return VariantTuple(arch, memory_usage, vcu_usage, workload)
                else:
                    return None

        self.apply_variant(DynamicJacobiVariant)

    def randomize_workload(self, seed: int = 0):
        if self.workload.random:
            self.workload.generate_workload(
                self.config.steps, seed=seed, **self.config.workload_args
            )
            self.data.workload = self.workload
            self.data.reset_data_size()
        else:
            return


register_graph(DynamicJacobiGraph, DynamicJacobiConfig)
