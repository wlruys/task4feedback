import pymetis
from .base import Geometry
import numpy as np
from collections import defaultdict


def metis_partition(cells, cell_neighbors, nparts=2):
    """
    Partition the mesh cells using METIS minimal edge cut partitioning.

    Returns:
      parts: a list where the i-th element is the partition id assigned to cell i.
    """
    # Build the adjacency list as required by METIS
    graph = []
    for i in range(len(cells)):
        graph.append(list(cell_neighbors[i]))
    # Compute the partitioning
    edgecuts, parts = pymetis.part_graph(nparts=nparts, adjacency=graph)
    return parts


def metis_geometry_partition(geometry: Geometry, nparts=2, round=2, direction=None):
    graph = []
    for i in range(len(geometry.cells)):
        neighbors = geometry.cell_neighbors[i]
        graph.append(list(neighbors))

    # Compute the partitioning
    edgecuts, parts = pymetis.part_graph(nparts=nparts, adjacency=graph)
    return parts


def bin_partition(geometry: Geometry, round=2, direction=None, n_parts=2):
    if direction is None:
        direction = 0

    # Get the projection of all centroids
    projection = np.zeros(len(geometry.cells))
    for i in range(len(geometry.cells)):
        # Get the centroid
        centroid = geometry.get_centroid(i, round_out=round)

        projection[i] = np.dot(centroid, direction)

    # Divide the projection into n_parts
    min_val = projection.min()
    max_val = projection.max()
    delta = (max_val - min_val) / n_parts
    partition_vector = [0] * len(geometry.cells)

    for i in range(len(geometry.cells)):
        projection = projection[i]
        for j in range(n_parts):
            if (
                projection >= min_val + j * delta
                and projection < min_val + (j + 1) * delta
            ):
                partition_vector[i] = j
                break
    return partition_vector


def ij_partition(geometry: Geometry, round=2):
    """
    Partition column-wise based on the geometry
    """
    # Get the max and min of the geometry
    # min = geometry.get_min_coordinate(direction)
    # max = geometry.get_max_coordinate(direction)

    row_map = defaultdict(list)
    col_map = defaultdict(list)
    ij_map = defaultdict(list)

    for i in range(len(geometry.cells)):
        # Get the centroid
        centroid = geometry.get_centroid(i, round_out=round)

        row_map[centroid[0]].append(i)
        col_map[centroid[1]].append(i)
        ij_map[(centroid[0], centroid[1])].append(i)

    row_keys = list(row_map.keys())
    col_keys = list(col_map.keys())
    row_keys.sort()
    col_keys.sort()

    return row_map, col_map, row_keys, col_keys, ij_map


def row_partition(geometry: Geometry, round=2, n_parts=4):
    row_map, col_map, row_keys, col_keys, ij_map = ij_partition(geometry, round=round)
    n_rows = len(row_keys)

    partition_vector = [0] * len(geometry.cells)

    rows_per_part = n_rows // n_parts

    for i in range(n_parts):
        start = i * rows_per_part
        end = (i + 1) * rows_per_part
        if i == n_parts - 1:
            end = n_rows

        for j in range(start, end):
            for k in row_map[row_keys[j]]:
                partition_vector[k] = i

    return partition_vector


def col_partition(geometry: Geometry, round=2, n_parts=4):
    row_map, col_map, row_keys, col_keys, ij_map = ij_partition(geometry, round=round)
    n_cols = len(col_keys)

    partition_vector = [0] * len(geometry.cells)

    cols_per_part = n_cols // n_parts

    for i in range(n_parts):
        start = i * cols_per_part
        end = (i + 1) * cols_per_part
        if i == n_parts - 1:
            end = n_cols

        for j in range(start, end):
            for k in col_map[col_keys[j]]:
                partition_vector[k] = i

    return partition_vector


def row_cyclic(geometry: Geometry, round=2, n_parts=4):
    row_map, col_map, row_keys, col_keys, ij_map = ij_partition(geometry, round=round)
    n_rows = len(row_keys)

    partition_vector = [0] * len(geometry.cells)

    rows_per_part = n_rows // n_parts

    for i in range(n_rows):
        for j in range(len(row_map[row_keys[i]])):
            partition_vector[row_map[row_keys[i]][j]] = i % n_parts

    return partition_vector


def col_cyclic(geometry: Geometry, round=2, n_parts=4):
    row_map, col_map, row_keys, col_keys, ij_map = ij_partition(geometry, round=round)
    n_cols = len(col_keys)

    partition_vector = [0] * len(geometry.cells)

    cols_per_part = n_cols // n_parts

    for i in range(n_cols):
        for j in range(len(col_map[col_keys[i]])):
            partition_vector[col_map[col_keys[i]][j]] = i % n_parts

    return partition_vector


def block_cyclic(
    geometry: Geometry,
    round=2,
    n_row_parts=2,
    n_col_parts=2,
    parts_per_column=2,
    parts_per_row=2,
):
    # With default parameters this partitions the mesh into 4 blocks of size 2x2

    _, _, row_keys, col_keys, ij_map = ij_partition(geometry, round=round)
    n_rows = len(row_keys)
    n_cols = len(col_keys)

    partition_vector = [0] * len(geometry.cells)

    for i in range(n_rows):
        for j in range(n_cols):
            rv = row_keys[i]
            cv = col_keys[j]

            # Calculate the block indices
            block_row = i // parts_per_row
            block_col = j // parts_per_column

            # Calculate the partition ID using a cyclic distribution
            row_part = block_row % n_row_parts
            col_part = block_col % n_col_parts
            part_id = row_part * n_col_parts + col_part

            # Assign the partition ID to all cells in this (i,j) position
            cells = ij_map[(rv, cv)]
            for c in cells:
                partition_vector[c] = part_id

    return partition_vector
