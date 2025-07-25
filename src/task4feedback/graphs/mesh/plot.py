from .base import Geometry
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .base import Cell, Edge
from ..base import DataBlocks, DataKey
from collections import defaultdict
from task4feedback.fastsim2 import TaskState
import task4feedback.fastsim2 as fastsim
import copy
from ..base import EnvironmentState

import wandb
import os


device_to_color = [
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "orange",
    "cyan",
]


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
    figsize=(8, 8),
    title="Mesh Plot",
    label_cells=False,
):
    """
    Return a figure and axis with the complete mesh plot with cells, edges, and vertices.
    """
    points, cells, unique_edges = geometry.cell_points, geometry.cells, geometry.edges

    if config is None:
        config = MeshPlotConfig()

    fig, ax = plt.subplots(figsize=figsize)
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


def shade_geometry_by_partition(
    ax,
    geometry,
    partition_vector,
    cmap=None,
    edgecolor="black",
    alpha=0.6,
    z_order=5,
):
    """
    Shade the geometry with a color map.

    Parameters:
      ax        : The matplotlib axis.
      geometry  : Geometry object containing points and cells.
      cmap      : A matplotlib colormap (default 'tab10').
      edgecolor : Color for cell edges.
      alpha     : Transparency for cell shading.

    Returns:
      The PolyCollection object added to the axis.
    """
    points, cells = geometry.cell_points, geometry.cells
    return shade_partitioning(
        ax,
        points,
        cells,
        partition_vector,
        cmap=cmap,
        edgecolor=edgecolor,
        alpha=alpha,
        z_order=z_order,
    )


def shade_partitioning(
    ax,
    points,
    cells,
    partition_vector,
    cmap=None,
    edgecolor="black",
    alpha=0.6,
    z_order=5,
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
    nparts = max(partition_vector) + 1
    if cmap is None:
        cmap = plt.get_cmap("tab10")
        facecolors = [cmap(i / max(1, nparts - 1)) for i in range(nparts)]

    if isinstance(cmap, list):
        facecolors = [cmap[i] for i in range(len(cmap))]

    # Create a list of face colors based on partition id
    polys = points[cells]  # shape: (num_cells, vertices, 2)
    colors = [facecolors[partition_vector[i]] for i in range(len(cells))]
    collection = PolyCollection(polys, facecolors=colors, alpha=alpha, zorder=z_order)
    ax.add_collection(collection)

    return collection


def label_cells(ax, points, cells, cell_labels, z_order=8):
    """
    Highlight cells and edges, and add labels at the center of each highlighted cell.

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
    for label, cell_ids in cell_labels.items():
        for cid in cell_ids:
            poly_coords = points[cells[cid]]
            centroid = poly_coords.mean(axis=0)
            text = ax.text(
                centroid[0],
                centroid[1],
                f"{label}",
                fontsize=25,
                ha="center",
                va="center",
                color="black",
                zorder=z_order,
            )
            artists.append(text)

    return artists


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
                fill=False,
                edgecolor=color,
                linewidth=6,
                alpha=0.8,
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

    # preallocate numpy array for all verts 
    c = 0
    for edge in side_highlights:
        for cell in side_highlights[edge]:
            c += 1
    verts = np.zeros((c, 4, 2))
    count = 0

    for eid in side_highlights:
        # Unpack vertex indices and cell ids
        v1_idx, v2_idx = edges[eid]
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

    collection = PolyCollection(verts, facecolors=color_list, alpha=0.8)
    ax.add_collection(collection)
    artists.append(collection)

    return artists


def animate_highlights(
    fig,
    ax,
    geom,
    highlight_sequence,
    interval=500,
    repeat=False,
    z_order=8,
    device_to_color=None,
):
    highlight_artists = []

    points, cells, unique_edges = geom.cell_points, geom.cells, geom.edges

    def update(frame):
        nonlocal highlight_artists
        
        for text in ax.texts:
            text.set_visible(False)

        for art in highlight_artists:
            art.remove()
        highlight_artists.clear()

        # Cycle through highlight_sets
        (
            cell_highlights,
            edge_highlights,
            boundary_highlights,
            cell_labels,
            partition,
        ) = highlight_sequence[frame % len(highlight_sequence)]

        new_artists = highlight_cells_edges(
            ax, points, cells, unique_edges, cell_highlights, edge_highlights, z_order
        )
        new_artists_2 = highlight_boundary(
            ax, geom, boundary_highlights, zorder=z_order + 6
        )
        new_artists_3 = label_cells(ax, points, cells, cell_labels, z_order=z_order)
        shade_collection = shade_partitioning(
            ax,
            points,
            cells,
            partition,
            cmap=device_to_color,
            edgecolor="black",
            alpha=0.5,
            z_order=z_order - 1,
        )
        highlight_artists.append(shade_collection)
        highlight_artists.extend(new_artists)
        highlight_artists.extend(new_artists_2)
        # highlight_artists.extend(new_artists_3)
        return highlight_artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(highlight_sequence),
        interval=interval,
        blit=False,
        repeat=False,
    )

    return ani


def animate_state_list(graph, state_list, figsize=(8, 8)):
    geom = graph.data.geometry
    fig, ax = create_mesh_plot(geom, figsize=figsize)
    highlight_sequence = []
    last_level_label = {}
    last_partition = graph.get_cell_locations(as_dict=False)
    s = 0
    for state in state_list:
        cell_highlights = defaultdict(list)
        cell_labels = defaultdict(list)

        for state_type, tasks in state.compute_tasks_by_state.items():
            for task in tasks:
                cell_id = graph.task_to_cell[task]

                if cell_id < len(graph.data.geometry.cells):
                    if state_type == fastsim.TaskState.LAUNCHED:
                        mapped_device = state.compute_task_mapping_dict[task]
                        cell_highlights[device_to_color[mapped_device]].append(cell_id)

                        label = graph.task_to_level[task]
                        if hasattr(graph, "task_to_direction"):
                            direction = graph.task_to_direction[task]
                            label = f"{label} ({direction})"

                        last_level_label[cell_id] = label
                        last_partition[cell_id] = mapped_device

        edge_highlights = defaultdict(lambda: list())
        boundary_highlights = defaultdict(lambda: dict())

        for state_type, tasks in state.data_tasks_by_state.items():
            for task in tasks:
                if state_type == fastsim.TaskState.LAUNCHED:
                    block_id = state.data_task_block[task]
                    obj = graph.data.get_key(block_id)
                    if isinstance(obj, Cell) or isinstance(obj, Edge):
                        continue
                    elif isinstance(obj, DataKey):
                        edge = obj.object
                        if isinstance(edge, Edge):
                            cell = obj.id[0]
                            edge_id = edge.id
                            cell_id = cell.id
                            target_device = state.data_task_mapping_dict[task]
                            boundary_highlights[edge_id].update(
                                {cell_id: device_to_color[target_device]}
                            )

        for cellid, level in last_level_label.items():
            cell_labels[level].append(cellid)
        highlight_sequence.append(
            (
                cell_highlights,
                edge_highlights,
                boundary_highlights,
                cell_labels,
                copy.deepcopy(last_partition),
            )
        )

    # Update title with time information
    def update_title():
        # Create update_title.frame attribute if it doesn't exist
        if not hasattr(update_title, "frame"):
            update_title.frame = 0
        update_title.frame += 1
        update_title.frame %= len(state_list)
        ax.set_title(f"Simulation Time: {state_list[update_title.frame].time}μs")

    update_title.frame = 0
    ani = animate_highlights(
        fig, ax, geom, highlight_sequence, device_to_color=device_to_color
    )
    ani.event_source.add_callback(update_title)
    return ani


def make_mesh_graph_animation(
    graph,
    state_list,
    title="mesh_animation",
    figsize=(8, 8),
    show=True,
    folder=None,
    dpi=None,
    bitrate=None,
):
    if folder is None:
        if wandb is None or wandb.run is None or wandb.run.dir is None:
            folder = "."
        else:
            folder = wandb.run.dir

    if not os.path.exists(folder):
        os.makedirs(folder)
    title = os.path.join(folder, title)

    title = f"{title}.mp4"
    ani = animate_state_list(graph, state_list, figsize=figsize)
    try:
        ani.save(title, writer="ffmpeg", fps=30, dpi=dpi, bitrate=bitrate)
    except Exception as e:
        print(f"Error saving animation: {e}")
    if show:
        plt.show()
    return ani


def animate_mesh_graph(
    env,
    time_interval=1000,
    show=True,
    title="mesh_animation",
    folder=None,
    figsize=(8, 8),
    dpi=300,
    bitrate=300,
):
    current_time = env.simulator.time
    state_list = []
    for t in range(0, current_time, time_interval):
        state = EnvironmentState.from_env(env, t)
        state_list.append(state)

    return make_mesh_graph_animation(
        env.simulator.input.graph,
        state_list,
        title=title,
        show=show,
        folder=None,
        figsize=figsize,
        dpi=dpi,
        bitrate=bitrate,
    )
