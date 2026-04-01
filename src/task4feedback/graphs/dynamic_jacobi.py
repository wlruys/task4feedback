import math
from dataclasses import dataclass, field
from typing import Self

import sympy

from ..interface.types import VariantTuple, _bytes_to_readable
from ..interface.wrappers import System, TaskTuple
from ..logging import training
from .base import (
    DataKey,
    DeviceType,
    DynamicWorkload,
    TrajectoryWorkload,
    VariantBuilder,
    register_graph,
)
from .jacobi import JacobiConfig, JacobiData, JacobiGraph, JacobiVariant
from .mesh.base import Cell, Edge, Geometry


@dataclass
class DynamicJacobiConfig(JacobiConfig):
    workload: DynamicWorkload = field(default_factory=TrajectoryWorkload)
    workload_args: dict = field(
        default_factory=lambda: {},
        metadata={"description": "Arguments for the workload generation."},
    )
    steps: int = 10
    level_chunks: int = 1
    r_interior: float = None
    r_boundary: float = None


class DynamicJacobiData(JacobiData):
    @staticmethod
    def from_mesh(
        geometry: Geometry,
        config: DynamicJacobiConfig,
        workload: DynamicWorkload,
        system: System | None = None,
    ) -> Self:
        return DynamicJacobiData(geometry, config, workload, system=system)

    def __init__(
        self,
        geometry: Geometry,
        config: DynamicJacobiConfig | None = None,
        workload: DynamicWorkload | None = None,
        system: System | None = None,
    ):
        self.config = DynamicJacobiConfig() if config is None else config
        self.workload = workload
        self.cell_to_interior_elems: dict[tuple[int, int], int] = {}

        # Filled during _create_blocks
        self.interior_elem: int | None = None
        self.boundary_elem: int | None = None

        super().__init__(geometry, self.config, system=system)

    def idx_at_step(self, step: int) -> int:
        return step

    def get_workload(self):
        return self.workload

    # -----------------------------
    # Core sizing helpers
    # -----------------------------
    def _solve_reference_interior_elems(self) -> int:
        """Solve interiors_per_level * y = level_memory / bytes_per_element."""
        interiors_per_level = self.geometry.get_num_cells()
        y = sympy.symbols("y", real=True, positive=True)
        equation = interiors_per_level * y - (
            self.config.level_memory / self.config.bytes_per_element
        )
        solution = sympy.solve(equation, y)
        return int(solution[0].evalf())

    def _derive_reference_elems_and_intensity(self, system: System) -> None:
        """
        Sets:
          self.interior_elem, self.boundary_elem
        and may adjust:
          config.arithmetic_complexity, config.arithmetic_intensity
        """
        self.interior_elem = self._solve_reference_interior_elems()

        # Default reference sizes from model
        if (self.config.r_interior, self.config.r_boundary) == (None, None):
            # boundary_elem derived from interior_elem unless overridden by boundary_time
            self.boundary_elem = (
                self.interior_elem**self.config.boundary_complexity
            ) * self.config.boundary_width

            # Override via time-based sizing if requested
            if self.config.interior_time is not None:
                interior_size = system.fastest_bandwidth * self.config.interior_time
                self.interior_elem = int(interior_size / self.config.bytes_per_element)

            if self.config.boundary_time is not None:
                boundary_size = system.fastest_bandwidth * self.config.boundary_time
                self.boundary_elem = int(boundary_size / self.config.bytes_per_element)

        else:
            # Ratio mode: boundary derived from r_boundary/r_interior, and adjust intensity.
            self.boundary_elem = (
                self.interior_elem
                * self.config.boundary_width
                * self.config.r_boundary
                / self.config.r_interior
            )

            # Force arithmetic model to match ratio-mode semantics
            self.config.arithmetic_complexity = 1.0
            self.config.arithmetic_intensity = (
                system.fastest_flops
                / 1e6
                / system.fastest_bandwidth
                / self.config.r_interior
                * self.config.bytes_per_element
            )

    def _cell_elems(self, step: int, cell: int) -> tuple[int, int, float]:
        """
        Returns (cell_interior_elems, cell_boundary_elems, workload_scalar).
        Mirrors original logic:
          - interior scales linearly with workload
          - boundary scales either from boundary_time (linear) or from complexity model
        """
        w = self.workload.get_scaled_cell_workload(step, cell)
        if w == 0:
            w = self.workload.get_scaled_cell_workload(step - 1, cell)
        interior_elems = int(self.interior_elem * w)

        if self.config.boundary_time is None and (
            self.config.r_interior,
            self.config.r_boundary,
        ) == (None, None):
            boundary_elems = int(
                (interior_elems**self.config.boundary_complexity)
                * self.config.boundary_width
                * w
            )
        else:
            boundary_elems = int(self.boundary_elem * w)

        return interior_elems, boundary_elems, w

    def _compute_block_times(
        self, interior_size_bytes: int, interior_elems: int, system: System
    ) -> float:
        """Original compute-time model: max(memory-time, arithmetic-time)."""
        mem_time = (interior_size_bytes * self.config.memory_intensity) / (
            system.fastest_gmbw / 1e6
        )
        arith_time = (
            (interior_elems**self.config.arithmetic_complexity)
            * self.config.arithmetic_intensity
            / (system.fastest_flops / 1e6)
        )
        return max(mem_time, arith_time)

    def _print_reference_info(self, system: System) -> None:
        interiors_per_level = self.geometry.get_num_cells()

        interior_size = int(self.interior_elem * self.config.bytes_per_element)
        boundary_size = int(self.boundary_elem * self.config.bytes_per_element)

        print(
            f"Total (per-level) Interior Size: "
            f"{_bytes_to_readable(interior_size * interiors_per_level)}"
        )
        print(f"Fastest bw: {system.fastest_bandwidth / 1e3:.2f} GB/s")
        print(
            "Communication time for reference interior size: "
            f"{interior_size / system.fastest_bandwidth:.2f} "
            f"{_bytes_to_readable(interior_size)} {self.interior_elem} elements"
        )
        print(
            "Communication time for reference boundary size: "
            f"{boundary_size / system.fastest_bandwidth:.2f} "
            f"{_bytes_to_readable(boundary_size)} {int(self.boundary_elem)} elements"
        )
        print(
            "Compute time for reference interior: "
            f"{(self.interior_elem**self.config.arithmetic_complexity) * self.config.arithmetic_intensity / (system.fastest_flops / 1e6):.2f}"
        )
        print(
            "Memory time for reference interior: "
            f"{(interior_size * self.config.memory_intensity) / (system.fastest_gmbw / 1e6):.2f}"
        )

    def _finalize_stats(
        self,
        interior_sizes: list[int],
        boundary_sizes: list[int],
        step_data_sum: list[int],
        compute_times: list[float],
        system: System,
    ) -> None:
        self.data_stat = {
            "interior_average": sum(interior_sizes) / len(interior_sizes),
            "interior_minimum": min(interior_sizes),
            "interior_maximum": max(interior_sizes),
            "boundary_average": sum(boundary_sizes) / len(boundary_sizes),
            "boundary_minimum": min(boundary_sizes),
            "boundary_maximum": max(boundary_sizes),
            "average_step_data": sum(step_data_sum) / len(step_data_sum),
            "average_step_data+ghost": (sum(step_data_sum) + sum(boundary_sizes))
            / len(step_data_sum),
            "interior_average_comm": (sum(interior_sizes) / len(interior_sizes))
            / system.fastest_bandwidth,
            "boundary_average_comm": (sum(boundary_sizes) / len(boundary_sizes))
            / system.fastest_bandwidth,
            "compute_average": sum(compute_times) / len(compute_times),
        }

    # -----------------------------
    # Public build/reset methods
    # -----------------------------
    def _create_blocks(self, system: System):
        interior_sizes: list[int] = []
        boundary_sizes: list[int] = []
        step_data_sum = [0 for _ in range(self.config.steps + 1)]
        compute_times: list[float] = []

        self._derive_reference_elems_and_intensity(system)
        self._print_reference_info(system)

        num_cells = len(self.geometry.cells)

        for cell in range(num_cells):
            centroid_x, centroid_y = self.geometry.get_centroid(cell)

            # Interior blocks (Cell, step)
            for step in range(self.config.steps + 1):
                interior_elems, boundary_elems, w = self._cell_elems(step, cell)
                self.cell_to_interior_elems[(cell, step)] = interior_elems

                interior_size = int(interior_elems * self.config.bytes_per_element)
                # Keep original behavior: ensure non-empty interior blocks have at least 1000 bytes
                interior_size = max(interior_size, 1)

                self.add_block(
                    DataKey(Cell(cell), step),
                    size=interior_size,
                    location=0,
                    x=centroid_x,
                    y=centroid_y,
                )

                assert interior_size > 0 or step == self.config.steps, (
                    "Interior data size must be positive "
                )
                if interior_size > 0:
                    interior_sizes.append(interior_size)
                    step_data_sum[step] += interior_size
                    compute_times.append(
                        self._compute_block_times(interior_size, interior_elems, system)
                    )

            # Boundary blocks (Edge, (Cell, step))
            for edge in self.geometry.cell_edges[cell]:
                edge_x, edge_y = self.geometry.get_edge_center(edge)

                for step in range(self.config.steps + 1):
                    interior_elems, boundary_elems, w = self._cell_elems(step, cell)
                    boundary_size = int(boundary_elems * self.config.bytes_per_element)

                    boundary_size = max(boundary_size, 1)

                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), step)),
                        size=boundary_size,
                        location=0,
                        x=edge_x,
                        y=edge_y,
                    )

                    assert boundary_size > 0 or step == self.config.steps, (
                        "Boundary data size must be positive"
                    )
                    if boundary_size > 1:
                        boundary_sizes.append(boundary_size)
                        step_data_sum[step] += boundary_size

        self._finalize_stats(
            interior_sizes, boundary_sizes, step_data_sum, compute_times, system
        )

    def reset_data_size(self, system: System):
        """
        Reset the data size of all blocks to a new trajectory.
        Uses the same sizing logic as _create_blocks, but only updates sizes.
        """
        interior_sizes: list[int] = []
        boundary_sizes: list[int] = []
        step_data_sum = [0 for _ in range(self.config.steps + 1)]
        compute_times: list[float] = []

        # NOTE: we assume interior_elem/boundary_elem are already set from _create_blocks().

        num_cells = len(self.geometry.cells)

        for cell in range(num_cells):
            for step in range(self.config.steps + 1):
                interior_elems, _, _ = self._cell_elems(step, cell)
                self.cell_to_interior_elems[(cell, step)] = interior_elems

                interior_size = int(interior_elems * self.config.bytes_per_element)

                self.blocks.set_size(
                    self.map.get_block(DataKey(Cell(cell), step)), interior_size
                )

                assert interior_size > 0 or step == self.config.steps, (
                    "Interior data size must be positive "
                )
                if interior_size > 0:
                    interior_sizes.append(interior_size)
                    step_data_sum[step] += interior_size
                    compute_times.append(
                        self._compute_block_times(interior_size, interior_elems, system)
                    )

            for edge in self.geometry.cell_edges[cell]:
                for step in range(self.config.steps + 1):
                    _, boundary_elems, w = self._cell_elems(step, cell)
                    boundary_size = int(boundary_elems * self.config.bytes_per_element)

                    if w > 0:
                        boundary_size = max(boundary_size, 1)

                    self.blocks.set_size(
                        self.map.get_block(DataKey(Edge(edge), (Cell(cell), step))),
                        boundary_size,
                    )

                    assert boundary_size > 0 or step == self.config.steps, (
                        "Boundary data size must be positive"
                    )
                    if boundary_size > 0:
                        boundary_sizes.append(boundary_size)
                        step_data_sum[step] += boundary_size

        self._finalize_stats(
            interior_sizes, boundary_sizes, step_data_sum, compute_times, system
        )


class DynamicJacobiGraph(JacobiGraph):
    def __init__(
        self,
        geometry: Geometry,
        config: DynamicJacobiConfig,
        system: System | None = None,
        variant: VariantBuilder | None = None,
    ):
        self.workload = config.workload
        self.workload.set_geometry(geometry)
        self.workload.generate_initial_mass(distribution=lambda x: 1.0)
        self.workload.generate_workload(config.steps, **config.workload_args)
        super(
            JacobiGraph, self
        ).__init__()  # Call base ComputeDataGraph constructor (not JacobiGraph constructor)
        self.reference_partition = self._build_reference_partition(config, system)
        self.config = config
        self.data: DynamicJacobiData = DynamicJacobiData.from_mesh(
            geometry, config, self.workload, system=system
        )

        assert system is not None, "System must be provided for DynamicJacobiGraph"
        self._build_graph(retire_data=True, system=system)
        self._apply_workload_variant(system)
        self.finalize()

    def _apply_workload_variant(self, system: System):
        task_to_level = self.task_to_level
        task_to_cell = self.task_to_cell

        # print("Building custom variant for system", system)

        class DynamicJacobiVariant(JacobiVariant):
            @staticmethod
            def build_variant(arch: DeviceType, task: TaskTuple) -> VariantTuple | None:
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
                    expected_work = (
                        interior_elem**self.config.arithmetic_complexity
                        * self.config.arithmetic_intensity
                    )
                    expected_time = int(expected_work / system.get_flop_ms(arch))
                    expected_memory = (
                        interior_elem
                        * self.config.bytes_per_element
                        * self.config.memory_intensity
                    )
                    expected_time = max(
                        expected_time, expected_memory / system.get_gmbw_ms(arch)
                    )
                    expected_time = int(max(expected_time, 1))

                # print(f"Task {task.id} (Cell {cell}, Level {level}): Workload={workload:.2f}, Expected Time={expected_time}ms on {arch.name}")

                if arch == DeviceType.GPU:
                    return VariantTuple(arch, memory_usage, vcu_usage, expected_time)
                else:
                    return None

        self.apply_variant(DynamicJacobiVariant)

    def randomize_workload(self, system, seed: int = 0):
        if self.workload.random:
            training.info(f"Randomizing workload with seed {seed}")
            self.workload.generate_workload(
                self.config.steps, seed=seed, **self.config.workload_args
            )
            self.data.workload = self.workload
            self.data.reset_data_size(system)
            self._apply_workload_variant(system)
            if self.is_finalized:
                assert self.static_graph is not None
                self.static_graph.update_variants(self.graph)

    def load_workload(self, system, state):
        if self.workload.random:
            self.workload.level_workload = state
            self.data.workload = self.workload
            self.data.reset_data_size(system)
            self._apply_workload_variant(system)
            if self.is_finalized:
                assert self.static_graph is not None
                self.static_graph.update_variants(self.graph)

    def get_workload(self) -> DynamicWorkload:
        return self.workload


register_graph(DynamicJacobiGraph, DynamicJacobiConfig)
