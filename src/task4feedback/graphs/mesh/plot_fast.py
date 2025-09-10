# anim_ultra.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, colors as mcolors
from matplotlib.collections import PolyCollection

# --- Global render perf hints ---
mpl.rcParams['path.simplify'] = True
mpl.rcParams['agg.path.chunksize'] = 10000  # chunk long paths


# ===============================
# Axis / canvas helpers
# ===============================
def _create_axes(geom, figsize=(8, 8)):
    """Minimal canvas with correct extents and equal aspect."""
    fig, ax = plt.subplots(figsize=figsize)
    pts = np.asarray(geom.cell_points, dtype=float)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    dx, dy = (xmax - xmin), (ymax - ymin)
    pad = 0.03
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


# --- Helpers (unchanged API) -----------------------------------------------

def _xy_points(geom):
    """Return float XY array from geom.cell_points, robust to extra columns."""
    pts = np.asarray(geom.cell_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("geom.cell_points must be (N,>=2).")
    return pts[:, :2].copy()  # force contiguous float XY

def _create_axes(geom, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    pts = _xy_points(geom)
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    dx, dy = (xmax - xmin), (ymax - ymin)
    pad = 0.03
    if dx <= 0: dx = 1.0
    if dy <= 0: dy = 1.0
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    return fig, ax

def _ensure_edges_and_edge_cells(geom):
    """As before, but unchanged; builds geom.edges and geom.edge_cell_dict if missing."""
    if hasattr(geom, "edges") and len(getattr(geom, "edges", [])) > 0 \
       and hasattr(geom, "edge_cell_dict") and isinstance(geom.edge_cell_dict, dict):
        return np.asarray(geom.edges, dtype=int), geom.edge_cell_dict

    edge_key_to_id = {}
    edge_cell_dict = {}
    next_id = 0
    for cid, cell in enumerate(geom.cells):
        v = np.asarray(cell, dtype=int)
        if v.size < 2: continue
        a, b = v, np.roll(v, -1)
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
        for u, w in pairs:
            key = (int(u), int(w))
            eid = edge_key_to_id.get(key)
            if eid is None:
                eid = next_id
                edge_key_to_id[key] = eid
                next_id += 1
            edge_cell_dict.setdefault(eid, []).append(cid)
    edges = np.empty((len(edge_key_to_id), 2), dtype=int)
    for (u, w), eid in edge_key_to_id.items():
        edges[eid] = (u, w)
    geom.edges = edges
    geom.edge_cell_dict = edge_cell_dict
    return edges, edge_cell_dict

# ===============================
# Geometry preprocessing
# ===============================
def _ensure_edges_and_edge_cells(geom) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Ensure we have:
      - edges: (n_edges, 2) int vertex indices
      - edge_cell_dict: {edge_id: [cell_id, ...]}
    Works for arbitrary polygons.
    """
    # If already present and valid, return
    if hasattr(geom, "edges") and len(getattr(geom, "edges", [])) > 0 \
       and hasattr(geom, "edge_cell_dict") and isinstance(geom.edge_cell_dict, dict):
        return np.asarray(geom.edges, dtype=int), geom.edge_cell_dict

    # Build from polygons
    edge_key_to_id: Dict[Tuple[int, int], int] = {}
    edge_cell_dict: Dict[int, List[int]] = {}
    next_id = 0

    for cid, cell in enumerate(geom.cells):
        verts = np.asarray(cell, dtype=int)
        if verts.size < 2:
            continue
        a, b = verts, np.roll(verts, -1)  # polygon edges
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
        for u, v in pairs:
            key = (int(u), int(v))
            eid = edge_key_to_id.get(key)
            if eid is None:
                eid = next_id
                edge_key_to_id[key] = eid
                next_id += 1
            edge_cell_dict.setdefault(eid, []).append(cid)

    # Materialize edges array
    edges = np.empty((len(edge_key_to_id), 2), dtype=int)
    for (u, v), eid in edge_key_to_id.items():
        edges[eid, 0] = u
        edges[eid, 1] = v

    # Cache on geom for reuse
    geom.edges = edges
    geom.edge_cell_dict = edge_cell_dict
    return edges, edge_cell_dict


def _precompute_boundary_quads(geom, width: float = 0.14) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Precompute slanted quads near each (edge, cell) pair:
    returns (verts: (N,4,2) float, pair_indices: {(eid,cid): idx})
    """
    edges, edge_cell_dict = _ensure_edges_and_edge_cells(geom)
    points = np.asarray(geom.cell_points, dtype=float)
    polys = [points[np.asarray(c, dtype=int)] for c in geom.cells]

    verts_list: List[np.ndarray] = []
    pair_indices: Dict[Tuple[int, int], int] = {}
    h = float(width)

    for eid, cell_list in edge_cell_dict.items():
        vi, vj = edges[eid]
        v1 = points[int(vi)]
        v2 = points[int(vj)]
        mid = (v1 + v2) / 2.0
        for cid in cell_list:
            centroid = polys[cid].mean(axis=0)
            diff = (centroid - mid) * h
            p1 = v1 + diff
            p2 = v2 + diff
            verts_list.append(np.array([v1, p1, p2, v2], dtype=float))
            pair_indices[(int(eid), int(cid))] = len(verts_list) - 1

    if not verts_list:
        return np.zeros((0, 4, 2), dtype=float), {}
    return np.stack(verts_list, axis=0), pair_indices


# ===============================
# Environment state (Tier-B arrays)
# ===============================
@dataclass(slots=True)
class EnvironmentStateLite:
    time: int
    # compute
    ct_state: np.ndarray       # (nC,) int8
    ct_device: np.ndarray      # (nC,) int32
    ct_duration_us: np.ndarray # (nC,) float32
    # data
    dt_state: np.ndarray
    dt_device: np.ndarray
    dt_source: np.ndarray
    dt_virtual: np.ndarray     # bool
    dt_block: np.ndarray       # int64 (opaque id)
    dt_duration_us: np.ndarray # float32


@dataclass(slots=True)
class _EnvStateCache:
    nC: int
    nD: int
    ct_cell_id: np.ndarray     # (nC,) int32
    data_edge_id: np.ndarray   # (nD,) int32 (−1 if not edge-bound)
    data_cell_id: np.ndarray   # (nD,) int32 (−1 if unknown)


def _extract_edge_cell_from_block(data_store, block_id) -> Tuple[int, int]:
    """
    Duck-typed extractor; returns (-1,-1) if block doesn't bind an Edge→Cell.
    Avoids hard dependency on project-specific classes.
    """
    try:
        obj = data_store.get_key(block_id)
        edge = getattr(obj, "object", None)
        cell0 = getattr(obj, "id", [None])[0]
        if edge is None or cell0 is None:
            return -1, -1
        eid = int(getattr(edge, "id", edge))
        cid = int(getattr(cell0, "id", cell0))
        return eid, cid
    except Exception:
        return -1, -1


def _get_env_cache(env) -> _EnvStateCache:
    cache = getattr(env, "_es_cache_ultra", None)
    if cache is not None:
        return cache
    sim = env.simulator
    tr = sim.state.get_task_runtime()
    sg = sim.state.get_tasks()
    graph = sim.input.graph

    nC = tr.get_n_compute_tasks()
    nD = tr.get_n_data_tasks()

    # compute task -> cell id (assume indexable)
    ct_cell_id = np.empty(nC, dtype=np.int32)
    for i in range(nC):
        ct_cell_id[i] = int(graph.task_to_cell[i])

    # data task -> (edge_id, cell_id) (static)
    data_edge_id = np.full(nD, -1, dtype=np.int32)
    data_cell_id = np.full(nD, -1, dtype=np.int32)
    store = graph.data
    for i in range(nD):
        block_id = sg.get_data_id(i)
        eid, cid = _extract_edge_cell_from_block(store, block_id)
        data_edge_id[i] = eid
        data_cell_id[i] = cid

    cache = _EnvStateCache(nC=nC, nD=nD, ct_cell_id=ct_cell_id,
                           data_edge_id=data_edge_id, data_cell_id=data_cell_id)
    env._es_cache_ultra = cache
    return cache


@dataclass(slots=True)
class _EnvArrays:
    ct_state: np.ndarray
    ct_device: np.ndarray
    ct_duration: np.ndarray
    dt_state: np.ndarray
    dt_device: np.ndarray
    dt_source: np.ndarray
    dt_virtual: np.ndarray
    dt_block: np.ndarray
    dt_duration: np.ndarray


def _ensure_env_arrays(env, cache: _EnvStateCache) -> _EnvArrays:
    arrs = getattr(env, "_es_arrays_ultra", None)
    if arrs is not None:
        return arrs
    nC, nD = cache.nC, cache.nD
    arrs = _EnvArrays(
        ct_state   = np.empty(nC, dtype=np.int8),
        ct_device  = np.empty(nC, dtype=np.int32),
        ct_duration= np.empty(nC, dtype=np.float32),
        dt_state   = np.empty(nD, dtype=np.int8),
        dt_device  = np.empty(nD, dtype=np.int32),
        dt_source  = np.empty(nD, dtype=np.int32),
        dt_virtual = np.empty(nD, dtype=bool),
        dt_block   = np.empty(nD, dtype=np.int64),
        dt_duration= np.empty(nD, dtype=np.float32),
    )
    env._es_arrays_ultra = arrs
    return arrs


def environment_state_from_env_arrays(env, time: Optional[int] = None) -> EnvironmentStateLite:
    """Fast, reusable array snapshot (Tier-B)."""
    sim = env.simulator
    if time is None:
        time = sim.time
    st = sim.state
    tr = st.get_task_runtime()
    sg = st.get_tasks()

    cache = _get_env_cache(env)
    arrs = _ensure_env_arrays(env, cache)

    # Localize bound methods (reduces Python lookup overhead)
    ct_state_at = tr.get_compute_task_state_at_time
    ct_dev      = tr.get_compute_task_mapped_device
    ct_dur      = tr.get_compute_task_duration

    dt_state_at = tr.get_data_task_state_at_time
    dt_dev      = tr.get_data_task_mapped_device
    dt_src      = tr.get_data_task_source_device
    dt_virt     = tr.is_data_task_virtual
    dt_dur      = tr.get_data_task_duration
    dt_block    = sg.get_data_id

    # Fill compute arrays
    for i in range(cache.nC):
        arrs.ct_state[i]    = ct_state_at(i, time)
        arrs.ct_device[i]   = ct_dev(i)
        arrs.ct_duration[i] = ct_dur(i)

    # Fill data arrays
    for i in range(cache.nD):
        v = dt_virt(i)
        arrs.dt_state[i]    = dt_state_at(i, time)
        arrs.dt_device[i]   = dt_dev(i)
        arrs.dt_source[i]   = dt_src(i)
        arrs.dt_virtual[i]  = v
        arrs.dt_block[i]    = dt_block(i)
        arrs.dt_duration[i] = 0.0 if v else dt_dur(i)

    return EnvironmentStateLite(
        time=int(time),
        ct_state=arrs.ct_state,
        ct_device=arrs.ct_device,
        ct_duration_us=arrs.ct_duration,
        dt_state=arrs.dt_state,
        dt_device=arrs.dt_device,
        dt_source=arrs.dt_source,
        dt_virtual=arrs.dt_virtual,
        dt_block=arrs.dt_block,
        dt_duration_us=arrs.dt_duration,
    )



# --- FIXED: always build boundary quads from XY only ------------------------

def _precompute_boundary_quads(geom, width: float = 0.14):
    """
    Precompute slanted quads near each (edge, cell) pair using XY only.
    Returns (verts: (K,4,2) float, pair_indices: {(eid,cid): idx})
    """
    edges, edge_cell_dict = _ensure_edges_and_edge_cells(geom)
    pts2 = _xy_points(geom)  # enforce 2D
    polys = [pts2[np.asarray(c, dtype=int)] for c in geom.cells]

    verts_list = []
    pair_indices = {}
    h = float(width)

    for eid, cell_list in edge_cell_dict.items():
        vi, vj = edges[eid]
        v1 = pts2[int(vi)]
        v2 = pts2[int(vj)]
        mid = (v1 + v2) / 2.0
        for cid in cell_list:
            centroid = polys[cid].mean(axis=0)
            diff = (centroid - mid) * h
            p1 = v1 + diff
            p2 = v2 + diff
            verts_list.append(np.array([v1, p1, p2, v2], dtype=float))
            pair_indices[(int(eid), int(cid))] = len(verts_list) - 1

    if not verts_list:
        return np.zeros((0, 4, 2), dtype=float), {}
    return np.stack(verts_list, axis=0), pair_indices

# --- FIXED & ROBUST: animate_highlights_ultra -------------------------------

def animate_highlights_ultra(
    geom,
    frame_builder,            # -> (outline_dev_per_cell, pair_idx, pair_dev, partition, hud_text)
    n_frames: int,
    *,
    figsize=(8, 8),
    video_seconds: int = 15,
    device_palette=None,      # None -> tab10; list[str] or Colormap accepted
    boundary_width: float = 0.14,
    use_blit: bool = True,
    diff_updates: bool = True,
    boundary_precompute=None, # (verts, pair_indices) or None
    draw_wire: bool = True,   # show a thin base wireframe for context
):
    import numpy as _np
    from matplotlib.collections import PolyCollection as _PC
    from matplotlib import colors as _mcolors

    fig, ax = _create_axes(geom, figsize)
    pts2 = _xy_points(geom)  # <- enforce XY here
    polys = [pts2[_np.asarray(c, dtype=int)] for c in geom.cells]
    n_cells = len(polys)

    # Palette
    if device_palette is None:
        cmap = plt.get_cmap("tab10")
        palette = _np.asarray([cmap(i / max(1, 9)) for i in range(10)], dtype=float)
    elif hasattr(device_palette, "__call__"):
        palette = _np.asarray([device_palette(i / max(1, 9)) for i in range(10)], dtype=float)
    else:
        palette = _np.asarray([_mcolors.to_rgba(c) for c in device_palette], dtype=float)
    npal = max(1, palette.shape[0])

    # Optional base wireframe (very cheap, gives structure even if fills transparent)
    if draw_wire:
        wire = _PC(polys, facecolors="none", edgecolors="#DDDDDD", linewidths=0.6,
                   antialiased=False, zorder=6)
        ax.add_collection(wire)

    # Fill collection
    shade = _PC(polys, facecolors=_np.zeros((n_cells, 4), dtype=float),
                edgecolors="none", antialiased=False, zorder=7)
    ax.add_collection(shade)
    shade_fc = shade.get_facecolors()
    if not isinstance(shade_fc, _np.ndarray) or shade_fc.shape != (n_cells, 4):
        buf = _np.zeros((n_cells, 4), dtype=float); shade.set_facecolors(buf); shade_fc = buf

    # Outline collection
    outlines = _PC(polys, facecolors=_np.zeros((n_cells, 4), dtype=float),
                   edgecolors=_np.zeros((n_cells, 4), dtype=float),
                   linewidths=_np.zeros((n_cells,), dtype=float),
                   antialiased=False, zorder=9)
    ax.add_collection(outlines)
    outline_ec = outlines.get_edgecolors()
    outline_lw = outlines.get_linewidths()
    if not isinstance(outline_ec, _np.ndarray) or outline_ec.shape != (n_cells, 4):
        buf = _np.zeros((n_cells, 4), dtype=float); outlines.set_edgecolors(buf); outline_ec = buf
    if not isinstance(outline_lw, _np.ndarray) or outline_lw.shape != (n_cells,):
        buf = _np.zeros((n_cells,), dtype=float); outlines.set_linewidths(buf); outline_lw = buf

    # Boundary quads
    if boundary_precompute is None:
        verts, pair_indices = _precompute_boundary_quads(geom, width=boundary_width)
    else:
        verts, pair_indices = boundary_precompute
    if verts.size:
        boundary = _PC(verts, facecolors=_np.zeros((len(verts), 4), dtype=float),
                       edgecolors="none", linewidths=0.0, antialiased=False, zorder=12)
        ax.add_collection(boundary)
        boundary_fc = boundary.get_facecolors()
        if not isinstance(boundary_fc, _np.ndarray) or boundary_fc.shape != (len(verts), 4):
            buf = _np.zeros((len(verts), 4), dtype=float); boundary.set_facecolors(buf); boundary_fc = buf
    else:
        boundary = None
        boundary_fc = None

    # HUD
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, ha="left", va="top",
                  fontsize=10, color="#222", zorder=99,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, boxstyle="round,pad=0.2"))

    # Blitting
    background = None
    def _cache_bg(_evt=None):
        nonlocal background
        fig.canvas.draw()
        if use_blit:
            background = fig.canvas.copy_from_bbox(ax.bbox)
    cid = fig.canvas.mpl_connect("resize_event", _cache_bg)

    fps = max(1, round(n_frames / max(1, video_seconds)))
    interval = int(max(1, 1000 * video_seconds / max(1, n_frames)))
    prev_part = None
    prev_boundary_active = _np.empty((0,), dtype=int)

    def _init():
        if boundary_fc is not None and len(boundary_fc):
            boundary_fc[:] = 0.0; boundary.set_facecolors(boundary_fc)
        outline_ec[:] = 0.0; outlines.set_edgecolors(outline_ec)
        outline_lw[:] = 0.0; outlines.set_linewidths(outline_lw)
        _cache_bg()
        return (shade, outlines, boundary, hud)

    def _update(i):
        nonlocal prev_part, prev_boundary_active

        outline_dev, pair_idx, pair_dev, partition, hud_text = frame_builder(i)

        # --- Interior fill (MAKE prev_part A COPY) ---
        idx = np.asarray(partition, dtype=np.int64) % npal
        if diff_updates and prev_part is not None and prev_part.shape == idx.shape:
            changed = (idx != prev_part)
            if np.any(changed):
                shade_fc[changed, :] = palette[idx[changed], :]
                shade.set_facecolors(shade_fc)
        else:
            shade_fc[:, :] = palette[idx, :]
            shade.set_facecolors(shade_fc)
        prev_part = idx.copy()             # <<< critical: snapshot, not a view

        # --- Outlines (unchanged logic) ---
        outline_lw[:] = 0.0
        if outline_dev is not None:
            dev = np.asarray(outline_dev, dtype=np.int64)
            active = (dev >= 0) & (dev < 1_000_000_000)
            if np.any(active):
                aidx = np.nonzero(active)[0]
                outline_ec[aidx, :] = palette[dev[aidx] % npal, :]
                outlines.set_edgecolors(outline_ec)
                outline_lw[aidx] = 2.2
                outlines.set_linewidths(outline_lw)
        else:
            outlines.set_linewidths(outline_lw)

        # --- Boundary strips (ALSO snapshot the active ids) ---
        if boundary_fc is not None:
            if prev_boundary_active.size:
                boundary_fc[prev_boundary_active, :] = 0.0
            if pair_idx is not None and len(pair_idx):
                pi = np.asarray(pair_idx, dtype=int)
                pd = np.asarray(pair_dev, dtype=int)
                boundary_fc[pi, :] = palette[pd % npal, :]
                prev_boundary_active = pi.copy()    # <<< snapshot
            else:
                prev_boundary_active = np.empty((0,), dtype=int)
            boundary.set_facecolors(boundary_fc)

        hud.set_text(hud_text or f"frame {i+1}/{n_frames}")

        if use_blit:
            fig.canvas.restore_region(background)
            ax.draw_artist(shade); ax.draw_artist(outlines)
            if boundary is not None: ax.draw_artist(boundary)
            ax.draw_artist(hud)
            fig.canvas.blit(ax.bbox)
        return (shade, outlines, boundary, hud)
    ani = animation.FuncAnimation(
        fig, _update, init_func=_init,
        frames=n_frames, interval=interval, blit=bool(use_blit),
        repeat=False, save_count=n_frames, cache_frame_data=False,
    )
    ani._fig_ref = fig
    ani._fps = fps
    ani._disconnect = lambda: fig.canvas.mpl_disconnect(cid)
    ani._pair_indices = pair_indices
    return ani

# ===============================
# Env → animation wiring (arrays)
# ===============================
def animate_env_ultra_arrays(
    env,
    *,
    time_interval: int = 500,
    video_seconds: int = 15,
    figsize=(8, 8),
    device_palette=None,
    boundary_width: float = 0.14,
    use_blit: bool = True,
    diff_updates: bool = True,
    stride: int = 1,
):
    """
    Build a streamed animation for `env` (no state list). Semantics:
      - Fill: last device touching a cell (LAUNCHED or COMPLETED)
      - Outline: cells with LAUNCHED compute this frame
      - Boundary strips: LAUNCHED data tasks this frame (colored by target device)
    """
    if time_interval <= 0:
        raise ValueError("time_interval must be positive.")

    geom = env.simulator.input.graph.data.geometry
    graph = env.simulator.input.graph
    cache = _get_env_cache(env)

    # Sample timeline (with stride)
    current_time = int(env.simulator.time)
    times = np.arange(0, current_time, time_interval * max(1, stride), dtype=np.int64)
    n_frames = max(1, len(times))

    # Persistent owner per cell
    n_cells = len(geom.cells)
    locs = np.asarray(graph.get_cell_locations(as_dict=False), dtype=np.int32)
    last_partition = np.zeros((n_cells,), dtype=np.int32)
    last_partition[:min(n_cells, locs.size)] = locs[:min(n_cells, locs.size)]

    # Precompute boundary quads once (and reuse mapping)
    boundary_pre = _precompute_boundary_quads(geom, width=boundary_width)
    pair_indices = boundary_pre[1]  # (eid,cid) -> idx

    # TaskState enum values (assumed IntEnum-like)
    TS_LAUNCHED  = int(env.simulator.state.get_task_runtime().TaskState.LAUNCHED) \
                   if hasattr(env.simulator.state.get_task_runtime(), "TaskState") \
                   else 1
    TS_COMPLETED = int(env.simulator.state.get_task_runtime().TaskState.COMPLETED) \
                   if hasattr(env.simulator.state.get_task_runtime(), "TaskState") \
                   else 2

    def frame_builder(i: int):
        t = int(times[i])
        s = environment_state_from_env_arrays(env, t)

        # Compute events
        mL = (s.ct_state == TS_LAUNCHED)
        mC = (s.ct_state == TS_COMPLETED)

        # COMPLETED updates owner persistently
        if np.any(mC):
            idx = np.nonzero(mC)[0]
            cells = cache.ct_cell_id[idx]
            last_partition[cells] = s.ct_device[idx]

        # LAUNCHED outlines (and also update owner)
        outline_dev_per_cell = np.full((n_cells,), -1, dtype=np.int32)
        if np.any(mL):
            idx = np.nonzero(mL)[0]
            cells = cache.ct_cell_id[idx]
            devs  = s.ct_device[idx]
            outline_dev_per_cell[cells] = devs
            last_partition[cells] = devs

        # Data launches → boundary strips
        mDL = (s.dt_state == TS_LAUNCHED)
        if np.any(mDL):
            di = np.nonzero(mDL)[0]
            eids = cache.data_edge_id[di]
            cids = cache.data_cell_id[di]
            devs = s.dt_device[di]
            valid = (eids >= 0) & (cids >= 0) & (cids < n_cells)
            eids, cids, devs = eids[valid], cids[valid], devs[valid]
            if eids.size:
                # Map (eid,cid) → pair index
                keys = list(zip(eids.tolist(), cids.tolist()))
                pidx = np.fromiter((pair_indices.get(k, -1) for k in keys), count=len(keys), dtype=int)
                ok = (pidx >= 0)
                pair_idx = pidx[ok]
                pair_dev = devs[ok]
            else:
                pair_idx = np.empty((0,), dtype=int)
                pair_dev = np.empty((0,), dtype=int)
        else:
            pair_idx = np.empty((0,), dtype=int)
            pair_dev = np.empty((0,), dtype=int)

        hud_text = f"t = {t} μs"
        return outline_dev_per_cell, pair_idx, pair_dev, last_partition, hud_text

    ani = animate_highlights_ultra(
        geom,
        frame_builder,
        n_frames=n_frames,
        figsize=figsize,
        video_seconds=video_seconds,
        device_palette=device_palette,
        boundary_width=boundary_width,
        use_blit=use_blit,
        diff_updates=diff_updates,
        boundary_precompute=boundary_pre,
    )
    return ani


# ===============================
# Saving utility
# ===============================
def save_animation_ultra(ani, path: str, dpi: Optional[int] = None, bitrate: Optional[int] = None):
    """Fast ffmpeg save with cleanup."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fps = getattr(ani, "_fps", 10)
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=bitrate if bitrate is not None else -1,
            extra_args=[
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-g", "24",
                "-threads", str(os.cpu_count() or 2),
                "-bf", "2",
                "-refs", "2",
                "-x264-params", "sync-lookahead=0",
            ],
        )
        ani.save(path, writer=writer, dpi=dpi)
    finally:
        try:
            if hasattr(ani, "_disconnect"):
                ani._disconnect()
        except Exception:
            pass
        try:
            if hasattr(ani, "event_source") and ani.event_source:
                ani.event_source.stop()
        except Exception:
            pass
        fig = getattr(ani, "_fig_ref", None)
        try:
            if fig is not None:
                plt.close(fig)
            else:
                plt.close("all")
        except Exception:
            pass
        del ani


# ===============================
# One-call convenience
# ===============================
def animate_mesh_graph_ultra(
    env,
    *,
    out_path: Optional[str] = None,
    time_interval: int = 500,
    video_seconds: int = 15,
    figsize=(8, 8),
    device_palette=None,
    boundary_width: float = 0.14,
    dpi: int = 150,
    bitrate: int = 6000,
    show: bool = False,
):
    """
    Produce and (optionally) save the animation for `env`.
    """
    ani = animate_env_ultra_arrays(
        env,
        time_interval=time_interval,
        video_seconds=video_seconds,
        figsize=figsize,
        device_palette=device_palette,
        boundary_width=boundary_width,
        use_blit=False,
        diff_updates=False,
        stride=1,
    )
    if out_path:
        save_animation_ultra(ani, out_path, dpi=dpi, bitrate=bitrate)
    if show:
        try:
            plt.show()
        finally:
            plt.close("all")
    return ani