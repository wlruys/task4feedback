from .mesh.base import *
from .mesh.partition import *
from .base import *
from typing import Callable, Optional, Self
from collections import defaultdict
from .jacobi import *
from .base import register_graph
from ..interface.types import _bytes_to_readable
from dataclasses import dataclass, field


@dataclass
class DynamicJacobiConfig(JacobiConfig):
    workload: DynamicWorkload = TrajectoryWorkload()
    workload_args: dict = field(
        default_factory=lambda: {},
        metadata={"description": "Arguments for the workload generation."},
    )
    steps: int = 10
    level_chunks: int = 1


class DynamicJacobiData(JacobiData):
    @staticmethod
    def from_mesh(
        geometry: Geometry, config: DynamicJacobiConfig, workload: DynamicWorkload, system: Optional[System] = None
    ) -> Self:
        data = DynamicJacobiData(geometry, config, workload, system=system)
        return data

    def __init__(
        self,
        geometry: Geometry,
        config: DynamicJacobiConfig = DynamicJacobiConfig(),
        workload: DynamicWorkload = None,
        system: Optional[System] = None,
    ):
        self.workload = workload
        self.cell_to_interior_elems = {}
        super().__init__(geometry, config, system=system)
        self.config: DynamicJacobiConfig

    def idx_at_step(self, step: int) -> int:
        return step
    
    def get_workload(self):
        return self.workload 

    def _create_blocks(self, system: System):
        interior_data = []
        boundary_data = []
        step_data_sum = [0 for _ in range(self.config.steps)]
        compute_time = []

        interiors_per_level = self.geometry.get_num_cells()
        edges_per_level = self.geometry.get_num_edges()

        y = sympy.symbols("y", real=True, positive=True)
        equation = interiors_per_level * y - self.config.level_memory / self.config.bytes_per_element 
        # equation = interiors_per_level * y + self.config.boundary_width * edges_per_level * (y)**self.config.boundary_complexity - self.config.level_memory / self.config.bytes_per_element
        solution = sympy.solve(equation, y)
        y_value = solution[0].evalf()
        interior_elem = int(y_value)
        #print("ERROR: ", interior_elem * interiors_per_level * self.config.bytes_per_element - self.config.level_memory)
        boundary_elem = interior_elem**(self.config.boundary_complexity) * self.config.boundary_width
        interior_size = interior_elem * self.config.bytes_per_element
        boundary_size = boundary_elem * self.config.bytes_per_element

        self.interior_elem = interior_elem
        self.boundary_elem = boundary_elem

        if self.config.interior_time is not None:
            assert(system is not None)
            interior_size = system.fastest_bandwidth * self.config.interior_time
            interior_elem = int(interior_size / self.config.bytes_per_element)

        if self.config.boundary_time is not None:
            assert(system is not None)
            boundary_size = system.fastest_bandwidth * self.config.boundary_time
            boundary_elem = int(boundary_size / self.config.bytes_per_element)

        interior_size = int(interior_size)
        boundary_size = int(boundary_size)

        print(f"Total (per-level) Interior Size: {_bytes_to_readable(interior_size * interiors_per_level)}")
        print(f"Communication time for reference interior size: {interior_size / system.fastest_bandwidth:.2f} {_bytes_to_readable(interior_size)} {interior_elem} elements")
        print(f"Communication time for reference boundary size: {boundary_size / system.fastest_bandwidth:.2f} {_bytes_to_readable(boundary_size)} {int(boundary_elem)} elements")
        print(f"Compute time for reference interior: {interior_elem ** self.config.arithmetic_complexity * self.config.arithmetic_intensity / (system.fastest_flops / 1e6):.2f}")
        print(f"Memory time for reference interior: {(interior_size * self.config.memory_intensity) / (system.fastest_gmbw / 1e6):.2f}")

        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create data blocks per cell for each level
            for i in range(self.config.steps + 1):
                workload = self.workload.get_scaled_cell_workload(i, cell)

                cell_interior_elem = int(interior_elem * workload)

                if self.config.boundary_time is None:
                    cell_boundary_elem = int(cell_interior_elem**(self.config.boundary_complexity) * self.config.boundary_width * workload)
                else:
                    cell_boundary_elem = int(boundary_elem * workload)

                self.cell_to_interior_elems[(cell, i)] = cell_interior_elem
                interior_size = cell_interior_elem * self.config.bytes_per_element
                boundary_size = cell_boundary_elem * self.config.bytes_per_element

                centroid = self.geometry.get_centroid(cell)
                centroid_x = centroid[0]
                centroid_y = centroid[1]

                self.add_block(DataKey(Cell(cell), i), size=interior_size, location=0, x=centroid_x, y=centroid_y)

                assert (
                    interior_size > 0 or i == self.config.steps
                ), "Interior data size must be positive "
                if interior_size > 0:
                    interior_data.append(interior_size)
                    step_data_sum[i] += interior_size
                    compute_time.append(max((interior_size * self.config.memory_intensity) / (system.fastest_gmbw / 1e6),int(interior_size / self.config.bytes_per_element) ** self.config.arithmetic_complexity * self.config.arithmetic_intensity / (system.fastest_flops / 1e6) ))

            # Create data blocks per edge for each level
            for edge in self.geometry.cell_edges[cell]:
                for i in range(self.config.steps + 1):
                    workload = self.workload.get_scaled_cell_workload(i, cell)
                    cell_interior_elem = int(interior_elem * workload)

                    if self.config.boundary_time is None:
                        cell_boundary_elem = int(cell_interior_elem**(self.config.boundary_complexity) * self.config.boundary_width * workload)
                    else:
                        cell_boundary_elem = int(boundary_elem * workload)

                    interior_size = cell_interior_elem * self.config.bytes_per_element
                    boundary_size = cell_boundary_elem * self.config.bytes_per_element

                    if workload > 0:
                        boundary_size = max(boundary_size, 1)

                    edge_center = self.geometry.get_edge_center(edge)
                    edge_x = edge_center[0]
                    edge_y = edge_center[1]

                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), i)),
                        size=boundary_size,
                        location=0,
                        x=edge_x,
                        y=edge_y,
                    )
                    assert (
                        boundary_size > 0 or i == self.config.steps
                    ), "Boundary data size must be positive"
                    if boundary_size > 0:
                        boundary_data.append(boundary_size)
                        step_data_sum[i] += boundary_size
        self.data_stat = {
            "interior_average": sum(interior_data) / len(interior_data),
            "interior_minimum": min(interior_data),
            "interior_maximum": max(interior_data),
            "boundary_average": sum(boundary_data) / len(boundary_data),
            "boundary_minimum": min(boundary_data),
            "boundary_maximum": max(boundary_data),
            "average_step_data": sum(step_data_sum) / len(step_data_sum),
            "interior_average_comm": sum(interior_data) / len(interior_data) / system.fastest_bandwidth,
            "boundary_average_comm": sum(boundary_data) / len(boundary_data) / system.fastest_bandwidth,
            "compute_average":  sum(compute_time) / len(compute_time),
        }
        print(f"Average Step Size: {int(self.data_stat['average_step_data']/1e9):,}GB")

    def reset_data_size(self, system: System):
        """
        Reset the data size of all blocks to a new trajectory.
        """
        interior_data = []
        boundary_data = []
        step_data_sum = [0 for _ in range(self.config.steps)]
        compute_time = []
        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 2 data blocks per cell
            for i in range(self.config.steps + 1):
                workload = self.workload.get_scaled_cell_workload(i, cell)
                cell_interior_elem = int(self.interior_elem * workload)
                interior_size = cell_interior_elem * self.config.bytes_per_element
                interior_size = int(interior_size)


                self.blocks.set_size(
                    self.map.get_block(DataKey(Cell(cell), i)), interior_size
                )
                assert (
                    interior_size > 0 or i == self.config.steps
                ), "Interior data size must be positive "
                if interior_size > 0:
                    interior_data.append(interior_size)
                    step_data_sum[i] += interior_size
                    compute_time.append(max((interior_size * self.config.memory_intensity) / (system.fastest_gmbw / 1e6),int(interior_size / self.config.bytes_per_element) ** self.config.arithmetic_complexity * self.config.arithmetic_intensity / (system.fastest_flops / 1e6) ))

            # Create 2 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                for i in range(self.config.steps + 1):
                    workload = self.workload.get_scaled_cell_workload(i, cell)
                    cell_interior_elem = int(self.interior_elem * workload)

                    if self.config.boundary_time is None:
                        cell_boundary_elem = int(
                            cell_interior_elem**(self.config.boundary_complexity)
                            * self.config.boundary_width
                            * workload
                        )
                    else:
                        cell_boundary_elem = int(self.boundary_elem * workload)

                    boundary_size = cell_boundary_elem * self.config.bytes_per_element
                    boundary_size = int(boundary_size)

                    if workload > 0:
                        boundary_size = max(boundary_size, 1)

                    self.blocks.set_size(
                        self.map.get_block(DataKey(Edge(edge), (Cell(cell), i))),
                        boundary_size,
                    )
                    
                    assert (
                        boundary_size > 0 or i == self.config.steps
                    ), "Boundary data size must be positive"
                    if boundary_size > 0:
                        boundary_data.append(boundary_size)
                        step_data_sum[i] += boundary_size

        self.data_stat = {
            "interior_average": sum(interior_data) / len(interior_data),
            "interior_minimum": min(interior_data),
            "interior_maximum": max(interior_data),
            "boundary_average": sum(boundary_data) / len(boundary_data),
            "boundary_minimum": min(boundary_data),
            "boundary_maximum": max(boundary_data),
            "average_step_data": sum(step_data_sum) / len(step_data_sum),
            "interior_average_comm": sum(interior_data) / len(interior_data) / system.fastest_bandwidth,
            "boundary_average_comm": sum(boundary_data) / len(boundary_data) / system.fastest_bandwidth,
            "compute_average":  sum(compute_time) / len(compute_time),
        }


class DynamicJacobiGraph(JacobiGraph):
    def __init__(
        self,
        geometry: Geometry,
        config: DynamicJacobiConfig,
        system: Optional[System]=None,
        variant: Optional[VariantBuilder] = None,
    ):
        self.workload = config.workload 
        self.workload.set_geometry(geometry)
        self.workload.generate_initial_mass(distribution=lambda x: 1.0)
        self.workload.generate_workload(config.steps, **config.workload_args)
        super(JacobiGraph, self).__init__() #Call base ComputeDataGraph constructor (not JacobiGraph constructor)
        self.config = config
        self.data: DynamicJacobiData = DynamicJacobiData.from_mesh(
            geometry, config, self.workload, system=system
        )

        assert(system is not None), "System must be provided for DynamicJacobiGraph"
        self._build_graph(retire_data=True, system=system)
        self._apply_workload_variant(system)
        self.finalize()

    def _apply_workload_variant(self, system: System):
        task_to_level = self.task_to_level
        task_to_cell = self.task_to_cell

       #print("Building custom variant for system", system)


        class DynamicJacobiVariant(JacobiVariant):
            @staticmethod
            def build_variant(
                arch: DeviceType, task: TaskTuple
            ) -> Optional[VariantTuple]:
                memory_usage = self.config.task_internal_memory
                vcu_usage = self.config.vcu_usage 

                level = task_to_level[task.id]
                cell = task_to_cell[task.id]

                if system.get_flops(arch) == 0:
                    return None 
                
                workload = self.workload.get_scaled_cell_workload(level, cell)
                
                if self.config.task_time is not None:
                    expected_time = workload * self.config.task_time
                    expected_time = int(expected_time)
                else:
                    interior_elem = self.data.cell_to_interior_elems[(cell, level)]
                    expected_work = interior_elem ** self.config.arithmetic_complexity * self.config.arithmetic_intensity
                    expected_time = int(expected_work / system.get_flop_ms(arch))
                    expected_memory = interior_elem * self.config.bytes_per_element * self.config.memory_intensity
                    expected_time = max(expected_time, expected_memory / system.get_gmbw_ms(arch))
                    expected_time = int(max(expected_time, 1))

                if arch == DeviceType.GPU:
                    return VariantTuple(arch, memory_usage, vcu_usage, expected_time)
                else:
                    return None

        self.apply_variant(DynamicJacobiVariant)

    def randomize_workload(self, system, seed: int = 0):
        if self.workload.random:
            self.workload.generate_workload(
                self.config.steps, seed=seed, **self.config.workload_args
            )
            self.data.workload = self.workload
            self.data.reset_data_size(system)
        
    def get_workload(self) -> DynamicWorkload:
        return self.workload


register_graph(DynamicJacobiGraph, DynamicJacobiConfig)
