from .mesh.base import Geometry, Cell, Edge
from ..interface import DataBlocks, Graph
from .base import DataGeometry, DataKey, GeometryIDMap
from dataclasses import dataclass

# def create_jacobi_graph(edge_dict, cell)
# array of cell to points
# dict of cell to neighbor (cells)
# dict of cell to edges

# create 2 data blocks per cell  (data id to cell or edge id)
# create 2 data blocks per edge
# loop over cells for each
# create task that reads from cell-block (it % 2) and writes to edge-block (it+1 % 2)

# def create_jacobi_graph(geom: Geometry, num_iterations: int):
#     blocks = DataBlocks()
#     for i in range(num_iterations):
#         blocks.add_block()


class JacobiData(DataGeometry):
    @staticmethod
    def from_mesh(geometry: Geometry):
        return JacobiData(geometry)

    def _create_blocks(self):
        # Loop over cells
        for cell in range(len(self.geometry.cells)):
            # Create 2 data blocks per cell
            for i in range(2):
                self.add_block(DataKey(Cell(cell), i), size=0, location=0)

            # Create 2 data blocks per edge
            for edge in self.geometry.cell_edges[cell]:
                for i in range(2):
                    self.add_block(
                        DataKey(Edge(edge), (Cell(cell), i)), size=0, location=0
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
            for edge in self.geometry.cell_edges[obj.id]:
                id_list = self.map.key_to_block.get_leaves(DataKey(Edge(edge), (obj,)))
                for i in id_list:
                    self.blocks.set_location(i, location)

    def set_locations_from_list(self, location_list: list[int]):
        for i, location in enumerate(location_list):
            self.set_location(Cell(i), location)


class JacobiGraph(Graph):
    def _build_graph(self):
        for i in range(self.num_iterations):
            for j, (cell, edges) in enumerate(self.data.geometry.cell_edges.items()):
                # Create task that:
                # -reads all of its block (interior and edges) and the edges of its neighbors
                # -writes to blocks of its self (interior and edges)

                idx = self.data.idx_at_step(i)

                name = f"Task(Cell({cell}), {i})"
                task_id = self.add_task(name, j)

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
        super().__init__()
        self.data = JacobiData.from_mesh(geometry)
        self.num_iterations = num_iterations
        self._build_graph()

    def get_data(self):
        return self.data

    def get_num_iterations(self):
        return self.num_iterations
