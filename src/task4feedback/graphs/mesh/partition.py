import pymetis


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
