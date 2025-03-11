from .base import Geometry
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional


def plot_edges(ax, points, edge_array, color="k", linewidth=1, alpha=0.5):
    lines = points[edge_array]  # shape: (num_edges, 2, 2)
    collection = LineCollection(
        lines, colors=color, linewidths=linewidth, zorder=2, alpha=alpha
    )
    ax.add_collection(collection)


def plot_vertices(ax, points, color="red", markersize=4, alpha=1.0):
    ax.scatter(
        points[:, 0], points[:, 1], color=color, s=markersize, zorder=3, alpha=alpha
    )


def plot_cells(
    ax, points, cells, facecolor="lightblue", edgecolor="black", alpha=0.5, label=False
):
    polys = points[cells]  # shape: (num_cells, vertices, 2)
    collection = PolyCollection(
        polys, facecolors=facecolor, edgecolors=edgecolor, alpha=alpha, zorder=1
    )
    ax.add_collection(collection)

    if label:
        for i, poly in enumerate(polys):
            centroid = np.mean(poly, axis=0)
            ax.text(
                centroid[0], centroid[1], f"{i}", ha="center", va="center", zorder=4
            )


@dataclass
class MeshPlotConfig:
    face_color: str = "lightblue"
    edge_color: str = "black"
    vertex_color: str = "red"
    vertex_size: int = 4
    edge_linewidth: int = 1
    alpha: float = 0.5


def create_mesh_plot(
    geometry: Geometry,
    config: Optional[MeshPlotConfig] = None,
    title="Mesh Plot",
    label_cells=False,
):
    """
    Return a figure and axis with the complete mesh plot with cells, edges, and vertices.
    Used as a base canvas for further plotting.
    """
    points, cells, unique_edges = geometry.cell_points, geometry.cells, geometry.edges

    if config is None:
        config = MeshPlotConfig()

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_cells(
        ax,
        points,
        cells,
        facecolor=config.face_color,
        edgecolor=config.edge_color,
        alpha=config.alpha,
        label=label_cells,
    )
    plot_edges(
        ax,
        points,
        unique_edges,
        color=config.edge_color,
        linewidth=config.edge_linewidth,
    )
    plot_vertices(ax, points, color=config.vertex_color, markersize=config.vertex_size)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.autoscale_view()

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def shade_partitioning(
    ax, points, cells, partition_vector, cmap=None, edgecolor="black", alpha=0.6
):
    """
    Shade each cell according to its partition membership.

    Parameters:
      ax              : The matplotlib axis.
      points          : Array of node coordinates.
      cells           : NumPy array of cell connectivity.
      partition_vector: List/array of partition id for each cell.
      cmap            : A matplotlib colormap (default 'tab10').
      edgecolor       : Color for cell edges.
      alpha           : Transparency for cell shading.

    Returns:
      The PolyCollection object added to the axis.
    """
    if cmap is None:
        cmap = plt.get_cmap("tab10")
    nparts = max(partition_vector) + 1
    # Create a list of face colors based on partition id
    facecolors = [cmap(i / max(1, nparts - 1)) for i in range(nparts)]
    polys = points[cells]  # shape: (num_cells, vertices, 2)
    colors = [facecolors[partition_vector[i]] for i in range(len(cells))]
    collection = PolyCollection(
        polys, facecolors=colors, edgecolors=edgecolor, alpha=alpha, zorder=5
    )
    ax.add_collection(collection)
    return collection


def highlight_cells_edges(
    ax, points, cells, unique_edges, cell_highlights, edge_highlights, z_order=8
):
    """
    Highlight cells and edges.

    Parameters:
      ax             : The matplotlib axis.
      points         : Array of node coordinates.
      cells          : NumPy array of cell connectivity.
      unique_edges   : NumPy array of unique edge definitions (pairs of vertex indices).
      cell_highlights: dict mapping color to list of cell indices.
      edge_highlights: dict mapping color to list of edge indices.

    Returns:
      A list of matplotlib artist objects for later removal.
    """
    artists = []
    # Highlight cells with custom colors (always drawn on top)
    for color, cell_ids in cell_highlights.items():
        for cid in cell_ids:
            poly_coords = points[cells[cid]]
            patch = Polygon(
                poly_coords,
                closed=True,
                fill=True,
                edgecolor=color,
                linewidth=3,
                alpha=0.5,
                zorder=z_order,
            )
            ax.add_patch(patch)
            artists.append(patch)

    # Highlight edges with custom colors
    for color, edge_ids in edge_highlights.items():
        for eid in edge_ids:
            # Each eid refers to an index in unique_edges
            edge = unique_edges[eid]
            v1, v2 = points[edge[0]], points[edge[1]]
            (line,) = ax.plot(
                [v1[0], v2[0]],
                [v1[1], v2[1]],
                color=color,
                linewidth=3,
                zorder=z_order - 1,
            )
            artists.append(line)
    return artists


def random_color(map: Optional[dict] = None):
    """
    Generate a random color string.

    Parameters:
      map: Optional dictionary to avoid generating the same color twice.

    Returns:
      A random color string in the format '#RRGGBB'.
    """
    if map is None:
        map = {}
    while True:
        color = "#" + "%06x" % np.random.randint(0, 0xFFFFFF)
        if color not in map:
            map[color] = True
            return color


def highlight_boundary(ax, geom, side_highlights, zorder=6, h=0.2):
    """
    Highlight each with different cell sides.

    Parameters:
      ax              : The matplotlib axis.
      side_highlights : Dict mapping (edge, cell) index to color.
      linewidth       : Line width for the highlighted segments.
      zorder          : Drawing order for the highlighted segments.
      h               : Width of the highlighted region.

    Returns:
      A list of matplotlib artist objects representing the drawn segments.
    """
    artists = []
    color_list = []
    points = geom.cell_points
    edges = geom.edges
    cells = geom.cells

    # preallocate numpy array for all verts we will need to plot
    c = 0
    for edge in side_highlights:
        for cell in side_highlights[edge]:
            c += 1
    verts = np.zeros((c, 4, 2))
    count = 0

    for eid in side_highlights:
        # Unpack vertex indices and cell ids
        v1_idx, v2_idx = edges[eid]

        # Retrieve vertex coordinates
        v1 = points[int(v1_idx)]
        v2 = points[int(v2_idx)]

        for cell in side_highlights[eid]:
            color = side_highlights[eid][cell]

            # Shade a region parallel to the edge but closer to the center of the cell
            mid_x = (v1[0] + v2[0]) / 2
            mid_y = (v1[1] + v2[1]) / 2

            centroid = np.mean(points[geom.cells[cell]][:, :2], axis=0)
            diff = (centroid - np.array([mid_x, mid_y])) * h

            p1 = np.array([v1[0] + diff[0], v1[1] + diff[1]])
            p2 = np.array([v2[0] + diff[0], v2[1] + diff[1]])

            verts[count][:] = np.array(
                [[v1[0], v1[1]], [p1[0], p1[1]], [p2[0], p2[1]], [v2[0], v2[1]]]
            )
            color_list.append(color)
            count += 1

    collection = PolyCollection(verts, facecolors=color_list)
    ax.add_collection(collection)
    artists.append(collection)

    return artists


def animate_highlights(
    fig, ax, geom, highlight_sequence, interval=500, repeat=False, z_order=8
):
    highlight_artists = []

    points, cells, unique_edges = geom.cell_points, geom.cells, geom.edges

    def update(frame):
        nonlocal highlight_artists
        # Remove previous highlight artists
        for art in highlight_artists:
            art.remove()
        highlight_artists.clear()

        # Cycle through highlight_sets
        cell_highlights, edge_highlights, boundary_highlights = highlight_sequence[
            frame % len(highlight_sequence)
        ]

        print(f"Frame: {frame}")
        print(f"Cell Highlights: {cell_highlights}")
        print(f"Edge Highlights: {edge_highlights}")
        new_artists = highlight_cells_edges(
            ax, points, cells, unique_edges, cell_highlights, edge_highlights, z_order
        )
        new_artists_2 = highlight_boundary(
            ax, geom, boundary_highlights, zorder=z_order
        )
        highlight_artists.extend(new_artists)
        highlight_artists.extend(new_artists_2)
        return highlight_artists

    # Create the animation (update every 1000ms)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(highlight_sequence),
        interval=interval,
        blit=False,
        repeat=False,
    )

    return ani
