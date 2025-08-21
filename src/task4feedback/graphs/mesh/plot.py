from .base import Geometry
from matplotlib import colors as mcolors
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

def shade_partitioning(ax, points, cells, partition_vector, cmap=None,
                       edgecolor="black", alpha=0.6, z_order=5):
    nparts = (max(partition_vector) + 1) if len(partition_vector) else 0
    polys = points[cells]

    if cmap is None:
        cmap = plt.get_cmap("tab10")

    # Allow str, Colormap, or list of colors
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if isinstance(cmap, list):
        facecolors = [mcolors.to_rgba(c) for c in cmap]
    else:  # assume Matplotlib Colormap
        k = max(1, nparts)  # avoid div by zero
        facecolors = [cmap(i / max(1, k - 1)) for i in range(k)]

    # Safe indexing (clip) in case partition ids exceed palette length
    max_idx = len(facecolors) - 1 if facecolors else 0
    colors = [facecolors[min(int(partition_vector[i]), max_idx)]
              for i in range(len(cells))]

    collection = PolyCollection(polys, facecolors=colors, edgecolors=edgecolor,
                                alpha=alpha, zorder=z_order)
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
    interval=None,
    z_order=8,
    device_to_color=None,
    video_seconds=15,
):
    points, cells, unique_edges = geom.cell_points, geom.cells, geom.edges

    polys = points[cells]
    n_cells = len(cells)

    if device_to_color is None:
        cmap = plt.get_cmap("tab10")
        device_rgba = [cmap(i / max(1, 9)) for i in range(10)]
    elif isinstance(device_to_color, list):
        device_rgba = [mcolors.to_rgba(c) for c in device_to_color]
    else:  
        cmap = device_to_color
        device_rgba = [cmap(i / max(1, 9)) for i in range(10)]

    shade_collection = PolyCollection(
        polys,
        facecolors=[(0, 0, 0, 0)] * n_cells,  # updated per frame
        edgecolors="none",
        alpha=0.5,
        zorder=z_order - 1,
    )
    ax.add_collection(shade_collection)

    active_outlines = []
    for i in range(n_cells):
        patch = Polygon(
            polys[i],
            closed=True,
            fill=False,
            edgecolor="none",
            linewidth=6,
            alpha=0.8,
            zorder=z_order,
        )
        patch.set_visible(False)
        ax.add_patch(patch)
        active_outlines.append(patch)

    pair_indices = {}
    verts_list = []
    colors_list = []

    h = 0.2
    for eid, cell_list in geom.edge_cell_dict.items():
        v1_idx, v2_idx = unique_edges[eid]
        v1 = points[int(v1_idx)]
        v2 = points[int(v2_idx)]
        mid = (v1 + v2) / 2.0
        for cid in cell_list:
            centroid = polys[cid].mean(axis=0)
            diff = (centroid - mid) * h
            p1 = v1 + diff
            p2 = v2 + diff
            verts_list.append(np.array([v1, p1, p2, v2]))
            colors_list.append((0, 0, 0, 0))
            pair_indices[(eid, cid)] = len(verts_list) - 1

    if verts_list:
        boundary_collection = PolyCollection(
            np.array(verts_list),
            facecolors=colors_list,
            alpha=0.8, 
            zorder=z_order + 6,
        )
        ax.add_collection(boundary_collection)
    else:
        boundary_collection = None

    text_objects = []
    for i in range(n_cells):
        centroid = polys[i].mean(axis=0)
        txt = ax.text(
            centroid[0],
            centroid[1],
            "",
            fontsize=25,
            ha="center",
            va="center",
            color="black",
            zorder=z_order,
        )
        txt.set_visible(False)
        text_objects.append(txt)

    def update(frame):
        (
            cell_highlights,
            edge_highlights,
            boundary_highlights,
            cell_labels,
            partition,
        ) = highlight_sequence[frame % len(highlight_sequence)]

        # 1) Update shaded partition colors (device of last task that used cell)
        shade_colors = [device_rgba[partition[i]] for i in range(n_cells)]
        shade_collection.set_facecolors(shade_colors)

        for patch in active_outlines:
            patch.set_visible(False)
        for color, cell_ids in cell_highlights.items():
            for cid in cell_ids:
                if 0 <= cid < n_cells:
                    patch = active_outlines[cid]
                    patch.set_edgecolor(color)
                    patch.set_visible(True)

        #Update boundary transfer highlights as colored slanted quads
        if boundary_collection is not None:
            fc = boundary_collection.get_facecolors()
            # set all to transparent
            if len(fc) > 0:
                fc[:] = (0, 0, 0, 0)
            for eid, celldict in boundary_highlights.items():
                for cid, color in celldict.items():
                    idx = pair_indices.get((eid, cid))
                    if idx is not None and 0 <= idx < len(fc):
                        fc[idx] = mcolors.to_rgba(color)
            boundary_collection.set_facecolors(fc)

        #Update labels to show the label of the last task that used the cell
        for txt in text_objects:
            txt.set_visible(False)
        for label, cell_ids in cell_labels.items():
            for cid in cell_ids:
                if 0 <= cid < n_cells:
                    t = text_objects[cid]
                    t.set_text(f"{label}")
                    t.set_visible(True)

        return [] 

    if interval is None:
        interval = int(video_seconds * 1000 / len(highlight_sequence))

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(highlight_sequence),
        interval=interval,
        blit=False,
        repeat=False,
    )

    return ani


def animate_state_list(graph, state_list, figsize=(8, 8), video_seconds=15, durations: bool = False):
    geom = graph.data.geometry
    fig, ax = create_mesh_plot(geom, figsize=figsize)
    highlight_sequence = []
    last_level_label = {}
    last_compute_duration = {}
    last_data_duration = {}
    last_partition = graph.get_cell_locations(as_dict=False)
    s = 0

    def fixed_scale(value, min=0.0, max=2000.0):
        if max <= min:
            return 0
        v = np.clip(float(value), min, max)
        return int(round((v - min) / (max - min) * 10))

    for state in state_list:
        cell_highlights = defaultdict(list)
        cell_labels = defaultdict(list)

        for state_type, tasks in state.compute_tasks_by_state.items():
            for task in tasks:
                cell_id = graph.task_to_cell[task]

                if cell_id < len(graph.data.geometry.cells):

                    if state_type == fastsim.TaskState.COMPLETED:
                        label = graph.task_to_level[task]
                        mapped_device = state.compute_task_mapping_dict[task]
                        if hasattr(graph, "task_to_direction"):
                            direction = graph.task_to_direction[task]
                            label = f"{label} ({direction})"
                        last_level_label[cell_id] = label
                        last_partition[cell_id] = mapped_device
                        last_compute_duration[cell_id] = state.compute_task_durations[task]
                    elif state_type == fastsim.TaskState.LAUNCHED:
                        mapped_device = state.compute_task_mapping_dict[task]
                        cell_highlights[device_to_color[mapped_device]].append(cell_id)

                        label = graph.task_to_level[task]
                        if hasattr(graph, "task_to_direction"):
                            direction = graph.task_to_direction[task]
                            label = f"{label} ({direction})"

                        last_level_label[cell_id] = label
                        last_partition[cell_id] = mapped_device
                        last_compute_duration[cell_id] = state.compute_task_durations[task]


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


        if durations:
            for cellid, duration in last_compute_duration.items():
                color_value = fixed_scale(duration, min=0.0, max=2000.0)
                cell_labels[color_value].append(cellid)
                #print(f"Cell {cellid} duration {duration} color value {color_value}")
                #cell_highlights[f"#{color_value:02x}00{255 - color_value:02x}"].append(cellid)
        else:
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
        fig, ax, geom, highlight_sequence, device_to_color=device_to_color, video_seconds=video_seconds
    )
    ani.event_source.add_callback(update_title)
    return ani

def animate_data_flow_durations_state_list(
    graph,
    state_list,
    figsize=(8, 8),
    video_seconds=15,
    show_labels: bool = False,
    # colormaps for edge flows and cell aggregate
    inbound_cmap_name: str = "Blues",
    outbound_cmap_name: str = "Oranges",
    aggregate_cmap_name: str = "Greys",
    # robust scaling
    robust: bool = True,
    robust_low: float = 5.0,
    robust_high: float = 95.0,
    # optional explicit bounds (microseconds); if None, inferred robustly
    in_min_us: float | None = None,
    in_max_us: float | None = None,
    out_min_us: float | None = None,
    out_max_us: float | None = None,
    agg_min_us: float | None = None,
    agg_max_us: float | None = None,
):
    """
    Animate non-virtual data-task durations going INTO and OUT OF each cell.

    For each frame (EnvironmentState snapshot):
      - Determine the current owner device of each cell from compute tasks.
      - Classify each COMPLETED, non-virtual data task tied to (edge, cell):
          inbound  if target_device == owner_device
          outbound if source_device == owner_device and target_device != owner_device
      - Keep the most recent per-cell inbound/outbound durations (μs).
      - Cell interior shows aggregate traffic (in+out) via grayscale bins (0..10).
      - Boundary quads near edges show inbound (Blues) and outbound (Oranges)
        with intensity scaled by duration.
      - Optionally label cells with "in: Xμs / out: Yμs".

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    import matplotlib.colors as _mcolors
    from collections import defaultdict as _dd

    geom = graph.data.geometry
    fig, ax = create_mesh_plot(geom, figsize=figsize)

    n_cells = len(geom.cells)
    # Owner device per cell (kept up to date per frame)
    last_partition = graph.get_cell_locations(as_dict=False)

    # Helpers ---------------------------------------------------------------
    def _get_edge_cell_from_block(block_id):
        """Return (edge_id, cell_id) if block_id refers to a DataKey for an Edge, else None."""
        obj = graph.data.get_key(block_id)
        if isinstance(obj, DataKey):
            edge = obj.object
            if isinstance(edge, Edge):
                cell = obj.id[0]  # convention from your existing code
                return edge.id, cell.id
        return None

    # Track most recent durations per cell (μs) over timeline when constructing frames
    last_in_us = {}   # cell_id -> duration_us
    last_out_us = {}  # cell_id -> duration_us

    # First pass: gather durations to compute robust scales ------------------
    all_in_us = []
    all_out_us = []
    all_agg_us = []

    # We emulate the per-frame update rules to estimate realistic scales
    for state in state_list:
        # Update owner devices from compute tasks (LAUNCHED/COMPLETED)
        for state_type, tasks in state.compute_tasks_by_state.items():
            if state_type in (fastsim.TaskState.LAUNCHED, fastsim.TaskState.COMPLETED):
                for task in tasks:
                    cid = graph.task_to_cell[task]
                    if 0 <= cid < n_cells:
                        last_partition[cid] = state.compute_task_mapping_dict[task]

        # Inspect COMPLETED non-virtual data tasks and classify into/out-of cell
        for state_type, tasks in state.data_tasks_by_state.items():
            if state_type != fastsim.TaskState.COMPLETED:
                continue
            for dt in tasks:
                if state.data_task_virtual.get(dt, False):
                    continue  # ignore virtual transfers
                pair = _get_edge_cell_from_block(state.data_task_block.get(dt))
                if pair is None:
                    continue
                edge_id, cell_id = pair
                if not (0 <= cell_id < n_cells):
                    continue
                owner = last_partition[cell_id]
                tgt = state.data_task_mapping_dict.get(dt)
                src = state.data_task_source_device.get(dt)
                dur = float(state.data_task_durations.get(dt, 0.0))

                if tgt == owner:
                    last_in_us[cell_id] = dur
                    all_in_us.append(dur)
                elif src == owner and tgt != owner:
                    last_out_us[cell_id] = dur
                    all_out_us.append(dur)

        # Aggregate for scale (sum of current last known)
        for cid in range(n_cells):
            agg = (last_in_us.get(cid, 0.0) + last_out_us.get(cid, 0.0))
            if agg > 0:
                all_agg_us.append(agg)

    def _robust_bounds(arr, lo, hi):
        if len(arr) == 0:
            return (0.0, 1.0)
        a = _np.asarray(arr, dtype=float)
        if robust and len(a) >= 3:
            return float(_np.percentile(a, lo)), float(_np.percentile(a, hi))
        return float(_np.min(a)), float(_np.max(a))

    in_lo, in_hi   = _robust_bounds(all_in_us,   robust_low, robust_high) if (in_min_us is None or in_max_us is None) else (float(in_min_us), float(in_max_us))
    out_lo, out_hi = _robust_bounds(all_out_us,  robust_low, robust_high) if (out_min_us is None or out_max_us is None) else (float(out_min_us), float(out_max_us))
    agg_lo, agg_hi = _robust_bounds(all_agg_us,  robust_low, robust_high) if (agg_min_us is None or agg_max_us is None) else (float(agg_min_us), float(agg_max_us))

    # Guard bounds
    def _fix(lo, hi):
        if not _np.isfinite(lo): lo = 0.0
        if not _np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        return lo, hi
    in_lo, in_hi = _fix(in_lo, in_hi)
    out_lo, out_hi = _fix(out_lo, out_hi)
    agg_lo, agg_hi = _fix(agg_lo, agg_hi)

    # Binners and color mappers ---------------------------------------------
    def _bin(value, lo, hi, bins=10):
        if value is None:
            return 0
        v = float(_np.clip(value, lo, hi))
        x = (v - lo) / (hi - lo)
        b = int(round(x * bins))
        return max(1, min(b, bins))

    inbound_cmap  = _plt.get_cmap(inbound_cmap_name)
    outbound_cmap = _plt.get_cmap(outbound_cmap_name)
    aggregate_cmap = _plt.get_cmap(aggregate_cmap_name)

    # partition bins -> grayscale colors for aggregate (index 0 = no data)
    agg_colors = ["#BFBFBF"] + [_mcolors.to_hex(aggregate_cmap(i / 9.0)) for i in range(10)]

    # Prepare animation frames ----------------------------------------------
    highlight_sequence = []
    # Reset state for actual frame construction
    last_partition = graph.get_cell_locations(as_dict=False)
    last_in_us.clear()
    last_out_us.clear()

    for state in state_list:
        # Update owner devices from compute tasks
        for state_type, tasks in state.compute_tasks_by_state.items():
            if state_type in (fastsim.TaskState.LAUNCHED, fastsim.TaskState.COMPLETED):
                for task in tasks:
                    cid = graph.task_to_cell[task]
                    if 0 <= cid < n_cells:
                        last_partition[cid] = state.compute_task_mapping_dict[task]

        # Build edge (boundary) highlights with directional colors
        boundary_highlights = _dd(dict)

        # Update last seen inbound/outbound durations for cells from COMPLETED, non-virtual data tasks
        for state_type, tasks in state.data_tasks_by_state.items():
            if state_type != fastsim.TaskState.COMPLETED:
                continue
            for dt in tasks:
                if state.data_task_virtual.get(dt, False):
                    continue
                pair = _get_edge_cell_from_block(state.data_task_block.get(dt))
                if pair is None:
                    continue
                edge_id, cell_id = pair
                if not (0 <= cell_id < n_cells):
                    continue
                owner = last_partition[cell_id]
                tgt = state.data_task_mapping_dict.get(dt)
                src = state.data_task_source_device.get(dt)
                dur = float(state.data_task_durations.get(dt, 0.0))

                if tgt == owner:
                    last_in_us[cell_id] = dur
                    # inbound edge color
                    frac = (min(max(dur, in_lo), in_hi) - in_lo) / (in_hi - in_lo) if in_hi > in_lo else 0.0
                    boundary_highlights[edge_id][cell_id] = _mcolors.to_hex(inbound_cmap(frac))
                elif src == owner and tgt != owner:
                    last_out_us[cell_id] = dur
                    # outbound edge color
                    frac = (min(max(dur, out_lo), out_hi) - out_lo) / (out_hi - out_lo) if out_hi > out_lo else 0.0
                    boundary_highlights[edge_id][cell_id] = _mcolors.to_hex(outbound_cmap(frac))

        # Aggregate intensity per cell for interior fill
        partition_bins = _np.zeros(n_cells, dtype=int)
        for cid in range(n_cells):
            agg = last_in_us.get(cid, 0.0) + last_out_us.get(cid, 0.0)
            if agg > 0:
                partition_bins[cid] = _bin(agg, agg_lo, agg_hi, bins=10)

        # Optional labels
        cell_labels = _dd(list)
        if show_labels:
            for cid in range(n_cells):
                inv = last_in_us.get(cid, 0.0)
                outv = last_out_us.get(cid, 0.0)
                if inv > 0 or outv > 0:
                    lbl = f"in:{inv:.0f}μs\nout:{outv:.0f}μs"
                    cell_labels[lbl].append(cid)

        # We keep compute-task cell outlines subdued (no special cell_highlights)
        cell_highlights = _dd(list)
        edge_highlights = _dd(list)  # not used; boundary_highlights carries flow info

        highlight_sequence.append(
            (
                cell_highlights,
                edge_highlights,
                boundary_highlights,
                cell_labels,
                partition_bins.copy(),  # reused by animate_highlights as "partition"
            )
        )

    # Title updater with scales
    def _update_title():
        if not hasattr(_update_title, "frame"):
            _update_title.frame = 0
        _update_title.frame = (_update_title.frame + 1) % len(state_list)
        t_us = state_list[_update_title.frame].time
        ax.set_title(
            "Data Flow Durations (non-virtual) — time {}μs\n"
            "Inbound scale: [{:.1f},{:.1f}] μs   "
            "Outbound scale: [{:.1f},{:.1f}] μs   "
            "Aggregate (fill) scale: [{:.1f},{:.1f}] μs".format(
                t_us, in_lo, in_hi, out_lo, out_hi, agg_lo, agg_hi
            )
        )

    _update_title.frame = 0

    ani = animate_highlights(
        fig,
        ax,
        geom,
        highlight_sequence,
        device_to_color=agg_colors,   # interior fill mapping for aggregate bins
        video_seconds=video_seconds,
    )
    ani.event_source.add_callback(_update_title)
    return ani

def animate_durations_state_list(
    graph,
    state_list,
    figsize=(8, 8),
    video_seconds=15,
    cmap_name: str = "viridis",
    min_ms: Optional[float] = None,
    max_ms: Optional[float] = None,
    robust: bool = True,
    robust_low: float = 5.0,
    robust_high: float = 95.0,
):
    """
    Animate a mesh where each cell's fill color encodes the duration (in microseconds) of the
    most recent *completed* compute task that finished there.

    Design:
      - We DO NOT modify any existing functions. We reuse `animate_highlights` by
        mapping duration bins (0..10) into its 'partition' channel.
      - Bin 0 is a neutral gray ("no data yet"). Bins 1..10 span the chosen duration range.
      - Duration scaling defaults to robust percentiles (p5..p95) across all frames,
        unless `min_ms`/`max_ms` are provided.

    Parameters
    ----------
    graph : object
        Must expose `data.geometry`, `task_to_cell`, and optionally `task_to_direction`, `task_to_level`.
    state_list : list[EnvironmentState]
        Sequence of states sampled over time (see `animate_mesh_graph`).
    figsize : tuple
        Figure size passed to `create_mesh_plot`.
    video_seconds : int
        Target animation duration in seconds (used to compute frame interval).
    cmap_name : str
        Matplotlib colormap name used for bins 1..10 (bin 0 is neutral gray).
    min_ms, max_ms : Optional[float]
        Explicit lower/upper bounds (in microseconds) for duration scaling.
        If None, they are inferred (robust percentiles by default).
    robust : bool
        If True and bounds not provided, use percentiles.
    robust_low, robust_high : float
        Percentile bounds (in %) for robust scaling when `robust` is True.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    from collections import defaultdict as _dd
    import matplotlib.colors as _mcolors

    geom = graph.data.geometry
    fig, ax = create_mesh_plot(geom, figsize=figsize)

    n_cells = len(geom.cells)

    # 1) Collect most-recent completed durations per cell over time to estimate scale.
    #    We'll also build the per-frame highlight payloads.
    #    We'll track last known duration per cell (microseconds).
    last_duration_us = {}  # cell_id -> duration_us
    all_durations_us = []

    # helper: pull completed duration updates from a frame
    def _update_durations_from_state(state):
        for state_type, tasks in state.compute_tasks_by_state.items():
            if state_type == fastsim.TaskState.COMPLETED:
                for task in tasks:
                    cell_id = graph.task_to_cell[task]
                    if 0 <= cell_id < n_cells:
                        dur = state.compute_task_durations[task]
                        # ensure float
                        dur = float(dur)
                        last_duration_us[cell_id] = dur
                        all_durations_us.append(dur)

    # Pre-scan to get robust bounds if needed
    for st in state_list:
        _update_durations_from_state(st)

    # Determine scaling bounds
    if min_ms is None or max_ms is None:
        if len(all_durations_us) == 0:
            # Fallback: arbitrary small range to avoid div-by-zero; everything will be "no data"
            lo, hi = 0.0, 1.0
        else:
            arr = _np.asarray(all_durations_us, dtype=float)
            if robust and len(arr) >= 3:
                lo = _np.percentile(arr, robust_low)
                hi = _np.percentile(arr, robust_high)
            else:
                lo, hi = float(_np.min(arr)), float(_np.max(arr))
        # convert to microseconds limits already; variables are in microseconds throughout
        min_us = lo if min_ms is None else float(min_ms)
        max_us = hi if max_ms is None else float(max_ms)
    else:
        min_us = float(min_ms)
        max_us = float(max_ms)

    # Guard against degenerate bounds
    if not _np.isfinite(min_us): min_us = 0.0
    if not _np.isfinite(max_us) or max_us <= min_us:
        max_us = min_us + 1.0

    def _bin_duration_us(u: Optional[float], bins: int = 10) -> int:
        """
        Map duration (microseconds) to integer bin in [1..bins].
        Return 0 for "no data".
        """
        if u is None:
            return 0
        # clip and scale
        v = float(_np.clip(u, min_us, max_us))
        # normalize to [0,1]
        x = (v - min_us) / (max_us - min_us)
        # map to 1..bins inclusive
        b = int(round(x * bins))
        b = max(1, min(b, bins))
        return b

    # Build colormap list: index 0 = neutral gray, 1..10 from the chosen cmap
    base_cmap = _plt.get_cmap(cmap_name)
    duration_colors = ["#BFBFBF"]  # bin 0 ("no data")
    duration_colors.extend([_mcolors.to_hex(base_cmap(i / 9.0)) for i in range(10)])  # 1..10

    # Re-initialize for per-frame construction
    last_duration_us.clear()
    highlight_sequence = []

    # We'll render labels showing the *binned* value (optional); here we leave labels off by default
    for state in state_list:
        # update durations from COMPLETED tasks
        _update_durations_from_state(state)

        # cell_highlights: outline launched compute cells (optional); keep minimal
        cell_highlights = _dd(list)
        for state_type, tasks in state.compute_tasks_by_state.items():
            if state_type == fastsim.TaskState.LAUNCHED:
                for task in tasks:
                    cell_id = graph.task_to_cell[task]
                    if 0 <= cell_id < n_cells:
                        # outline with a dim color (same neutral gray) to avoid fighting the fill
                        cell_highlights[duration_colors[0]].append(cell_id)

        # No edge-specific overlays for this animation (keep boundary highlights empty)
        edge_highlights = _dd(list)
        boundary_highlights = _dd(dict)

        # Labels are optional/noisy; keep them empty by default.
        # If you want labels, you could put the *binned* index or the raw duration here.
        cell_labels = _dd(list)

        # Build the 'partition' vector where each entry is the duration bin (0..10)
        partition_bins = _np.zeros(n_cells, dtype=int)
        for cid in range(n_cells):
            d = last_duration_us.get(cid, None)
            partition_bins[cid] = _bin_duration_us(d, bins=10)

        # NOTE: We pass a *copy* because animate_highlights mutates the per-frame 'partition'
        highlight_sequence.append(
            (
                cell_highlights,
                edge_highlights,
                boundary_highlights,
                cell_labels,
                partition_bins.copy(),
            )
        )

    # Prepare title updater to show time and the scale mapping
    def _update_title():
        if not hasattr(_update_title, "frame"):
            _update_title.frame = 0
        _update_title.frame = (_update_title.frame + 1) % len(state_list)
        t_us = state_list[_update_title.frame].time
        ax.set_title(
            f"Task Duration Heatmap — time {t_us}μs\n"
            f"Scale: bin 1={min_us:.1f}μs → bin 10={max_us:.1f}μs (0 = no data)"
        )

    _update_title.frame = 0

    ani = animate_highlights(
        fig,
        ax,
        geom,
        highlight_sequence,
        device_to_color=duration_colors,  # <- drives duration colormap
        video_seconds=video_seconds,
    )
    ani.event_source.add_callback(_update_title)
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
    video_seconds=15,
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
    ani = animate_state_list(graph, state_list, figsize=figsize, video_seconds=video_seconds)
    #ani = animate_durations_state_list(graph, state_list, figsize=figsize, video_seconds=video_seconds)
    #ani = animate_data_flow_durations_state_list(graph, state_list, figsize=figsize, video_seconds=video_seconds)
    try:
        ani.save(title, writer="ffmpeg", dpi=dpi, bitrate=bitrate)
    except Exception as e:
        print(f"Error saving animation: {e}")
    if show:
        plt.show()
    return ani


def animate_mesh_graph(
    env,
    time_interval=500,
    show=True,
    title="mesh_animation",
    folder=None,
    figsize=(8, 8),
    dpi=300,
    bitrate=300,
    video_seconds=15,
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
        folder=folder,
        figsize=figsize,
        dpi=dpi,
        bitrate=bitrate,
        video_seconds=video_seconds,
    )
