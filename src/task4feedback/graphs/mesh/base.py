import numpy as np
import gmsh
import pygmsh
import meshio
from dataclasses import dataclass
from typing import Optional

GMSH_INITIALIZED = False


@dataclass(frozen=True, eq=True, order=True, slots=True)
class Cell:
    """
    Data class representing a cell in a mesh.
    """

    id: int


@dataclass(frozen=True, eq=True, order=True, slots=True)
class Edge:
    """
    Data class representing an edge in a mesh.
    """

    id: int


def geom_to_int(obj: Cell | Edge) -> int:
    return obj.id


def geom_to_int_list(obj_list: list[Cell | Edge]) -> list[int]:
    return [geom_to_int(obj) for obj in obj_list]


def filter_geometry_list(
    obj_list: list[Edge | Cell], to_int: bool = False
) -> tuple[list[Cell], list[Edge]] | tuple[list[int], list[int]]:
    edge_list = []
    cell_list = []

    for obj in obj_list:
        if isinstance(obj, Edge):
            if to_int:
                edge_list.append(obj.id)
            else:
                edge_list.append(obj)
        elif isinstance(obj, Cell):
            if to_int:
                cell_list.append(obj.id)
            else:
                cell_list.append(obj)

    return cell_list, edge_list


@dataclass(frozen=True, eq=True, order=True, slots=True)
class Geometry:
    """
    Data class representing a mesh geometry.
    """

    cells: np.array
    cell_points: np.array
    cell_neighbors: dict
    cell_edges: dict
    edges: np.array
    vertex_edge_dict: dict
    edge_cell_dict: dict
    centroids: np.array
    bounds: Optional[np.array] = None

    def get_centroid(self, cell, round_out=None):
        cell_verticies = np.asarray(
            self.cell_points[self.cells[cell]][:, :2], dtype=np.float64
        )
        centroid = np.mean(cell_verticies, axis=0)

        if round_out is not None:
            centroid = np.round(centroid, round_out)
        return centroid

    def get_max_coordinate(self, direction=0):
        """
        Get the maximum coordinate of the mesh in a given direction.
        """
        if direction == 0:
            return np.max(self.cell_points[:, 0])
        elif direction == 1:
            return np.max(self.cell_points[:, 1])
        else:
            raise ValueError("Direction must be 0 (x) or 1 (y).")

    def get_min_coordinate(self, direction=0):
        """
        Get the minimum coordinate of the mesh in a given direction.
        """
        if direction == 0:
            return np.min(self.cell_points[:, 0])
        elif direction == 1:
            return np.min(self.cell_points[:, 1])
        else:
            raise ValueError("Direction must be 0 (x) or 1 (y).")

    def get_normal_to_edge(self, edge, cell, round_in=None, round_out=None):
        """
        Get the normal vector to the edge, where cell is the interior.
        """

        # Get the vertices of the edge
        v1, v2 = self.edges[edge]

        # Get the coordinates of the vertices
        x1, y1 = self.cell_points[int(v1)]
        x2, y2 = self.cell_points[int(v2)]

        if round_in is not None:
            x1 = np.round(x1, round_in)
            y1 = np.round(y1, round_in)
            x2 = np.round(x2, round_in)
            y2 = np.round(y2, round_in)

        # Put in higher precision
        x1 = np.float64(x1)
        y1 = np.float64(y1)
        x2 = np.float64(x2)
        y2 = np.float64(y2)

        # Calculate the normal vector (perpendicular to the edge)
        normal = np.array([-(y2 - y1), x2 - x1], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        # print("normal", normal)

        # Check if the normal is pointing towards the cell
        # Calculate the centroid of the cell
        centroid = self.get_centroid(cell, round_out=round_in)

        # Calculate the vector from the centroid to the edge
        edge_vector = np.array([x1, y1], dtype=np.float64) - np.asarray(
            centroid, dtype=np.float64
        )
        edge_vector /= np.linalg.norm(edge_vector)

        # Calculate the dot product
        dot_product = np.dot(normal, edge_vector)

        # If the dot product is negative, flip the normal vector
        if dot_product < 0:
            normal = -normal

        if round_out is not None:
            normal = np.round(normal, round_out)

        return normal


def initialize_gmsh():
    global GMSH_INITIALIZED
    if not GMSH_INITIALIZED:
        if not gmsh.is_initialized():
            gmsh.initialize()
    GMSH_INITIALIZED = True


def finalize_gmsh():
    global GMSH_INITIALIZED
    if GMSH_INITIALIZED:
        if gmsh.is_initialized():
            gmsh.finalize()
    GMSH_INITIALIZED = False


def generate_quad_mesh(L=1.0, n=4):
    """
    Generate a 2D rectangular mesh with quadrilateral elements.

    Returns:
    - meshio.Mesh object
    """
    nx = L / n
    gmsh.option.set_number("Mesh.RecombineAll", 1)
    gmsh.option.set_number("Mesh.Algorithm", 8)  # 8 = Delaunay
    with pygmsh.geo.Geometry() as geom:
        rectangle = geom.add_rectangle(0.0, L, 0.0, L, 0.0, mesh_size=nx)
        for curve in rectangle.curves:
            geom.set_transfinite_curve(curve, n + 1, "Progression", 1.0)
        geom.set_transfinite_surface(rectangle.surface, "Left", tuple(rectangle.curves))

        mesh = geom.generate_mesh(dim=2)
    return mesh


def generate_tri_mesh(L=1.0, n=4):
    """
    Generate a 2D rectangular mesh with triangular elements.
    Returns:
    - meshio.Mesh object
    """
    nx = L / n
    with pygmsh.geo.Geometry() as geom:
        geom.add_rectangle(0.0, L, 0.0, L, 0.0, mesh_size=nx)
        # geom.set_recombined_surfaces([rectangle.surface])
        mesh = geom.generate_mesh(dim=2)
    return mesh


def get_cells(mesh):
    """
    Parameters:
    - mesh: meshio.Mesh object

    Returns:
    - NumPy array of cell vertex indices
    """
    for cell_block in mesh.cells:
        if cell_block.type in ["quad", "quad4"]:
            return np.array(cell_block.data)

    # Return non quadrilateral cells
    for cell_block in mesh.cells:
        if cell_block.type in ["triangle", "triangle3"]:
            return np.array(cell_block.data)
    raise ValueError("No quads or triangles found in the mesh.")


def get_cell_neighbors(cells):
    """
    Parameters:
    - cells: NumPy array of cells containing cell vertex indices

    Returns:
    - Dictionary mapping each cell index to its neighboring cell indices.
    """
    N, n = cells.shape
    # Build array of edges (each edge is represented in sorted order)
    cells_shifted = np.roll(cells, shift=-1, axis=1)
    edges = np.stack((cells, cells_shifted), axis=2)
    edges.sort(axis=2)
    edges_flat = edges.reshape(-1, 2)

    # Map each edge to its cell id
    cell_ids = np.repeat(np.arange(N), n)

    # Find unique edges and get counts
    unique_edges, inverse, counts = np.unique(
        edges_flat, axis=0, return_inverse=True, return_counts=True
    )

    cell_neighbors = {i: set() for i in range(N)}
    for ue_index, count in enumerate(counts):
        if count > 1:
            indices = np.where(inverse == ue_index)[0]
            cells_sharing_edge = np.unique(cell_ids[indices])
            for ci in cells_sharing_edge:
                cell_neighbors[ci].update(cells_sharing_edge[cells_sharing_edge != ci])
    return cell_neighbors


def extract_unique_edges(cells):
    """
    Returns:
    - dictionary mapping edge tuple (cell, cell) to an edge index
    - edge index to numPy array of unique (cell, cell) edges.
    """
    N, n = cells.shape
    cells_shifted = np.roll(cells, shift=-1, axis=1)
    edges = np.stack((cells, cells_shifted), axis=2)
    edges.sort(axis=2)
    edges_flat = edges.reshape(-1, 2)
    cell_ids = np.repeat(np.arange(N), n)
    unique_edges, inverse_indicies = np.unique(edges_flat, axis=0, return_inverse=True)
    edge_dict = {tuple(edge): i for i, edge in enumerate(unique_edges)}

    edge_to_cells = {i: [] for i in range(len(unique_edges))}
    for flat_idx, edge_id in enumerate(inverse_indicies):
        cell = cell_ids[flat_idx]
        edge_to_cells[edge_id].append(cell)

    return edge_dict, unique_edges, edge_to_cells


def get_cell_edges(cells, edge_dict=None):
    """
    Compute a dictionary mapping each cell index to its edge IDs.

    Parameters:
    - cells: NumPy array of cells containing cell vertex indices
    - edge_dict: Optional dictionary mapping edge tuples to edge IDs

    Returns:
    - Dictionary mapping cell IDs to lists of edge IDs
    """
    # Get edge dictionary if not provided
    if edge_dict is None:
        edge_dict, _ = extract_unique_edges(cells)

    N, n = cells.shape
    # Generate the edges for each cell (pairs of adjacent vertices)
    cells_shifted = np.roll(cells, shift=-1, axis=1)
    edges = np.stack((cells, cells_shifted), axis=2)
    edges.sort(axis=2)  # Ensure canonical edge representation (v1 < v2)

    # Create cell-to-edge mapping
    cell_edges = {}

    # For each cell, find its edges and their IDs
    for i in range(N):
        edge_ids = []
        for j in range(n):
            edge_tuple = tuple(edges[i, j])
            edge_id = edge_dict[edge_tuple]
            edge_ids.append(edge_id)
        cell_edges[i] = edge_ids

    return cell_edges


def get_centroids(cells, points, round_decimals: Optional[int] = 3):
    """
    Parameters:
        - cells: NumPy array of shape (N, n) containing cell vertex indices
        - points: NumPy array of shape (num_points, 2) containing vertex coordinates
    """

    # Compute the centroid of each cell
    centroids = np.zeros((len(cells), 2))

    for i, cell in enumerate(cells):
        centroid = np.mean(points[cell][:2], axis=0)
        centroids[i] = centroid

    if round_decimals is not None:
        centroids = np.round(centroids, round_decimals)

    return centroids


def build_geometry(mesh):
    """
    Parameters:
    - mesh: meshio.Mesh object

    Returns:
    - Geometry object containing the following fields:
        - cells: NumPy array of cell vertex indices
        - cell_points: NumPy array of cell centroid coordinates
        - cell_neighbors: Dictionary mapping cell indices to neighboring cell indices
        - cell_edges: Dictionary mapping cell indices to edge indices
        - edges: NumPy array of unique edge vertex indices
        - edge_dict: Dictionary mapping edge vertex tuples to edge indices
    """
    cells = get_cells(mesh)
    points = mesh.points[:, :2]
    cell_neighbors = get_cell_neighbors(cells)
    edge_dict, edges, edge_to_cells = extract_unique_edges(cells)
    cell_edges = get_cell_edges(cells, edge_dict)
    centroids = get_centroids(cells, points)

    # Get bounding box for mesh.io object
    bounds = np.array(
        [
            np.min(points[:, 0]),
            np.max(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 1]),
        ]
    )

    return Geometry(
        cells=cells,
        cell_points=points,
        cell_neighbors=cell_neighbors,
        cell_edges=cell_edges,
        edges=edges,
        vertex_edge_dict=edge_dict,
        edge_cell_dict=edge_to_cells,
        centroids=centroids,
        bounds=bounds,
    )


def save_gmsh(filename, mesh):
    """
    Parameters:
    - filename: str
    - mesh: meshio.Mesh object
    """
    meshio.write(filename, mesh)


def load_gmsh(filename):
    """
    Parameters:
    - filename: str

    Returns:
    - meshio.Mesh object
    """
    return meshio.read(filename)
