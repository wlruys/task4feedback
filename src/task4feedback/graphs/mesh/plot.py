from enum import IntEnum
from .base import Geometry
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal 
from .base import Cell, Edge
from ..base import DataBlocks, DataKey
from collections import defaultdict
from task4feedback.fastsim2 import TaskState
import task4feedback.fastsim2 as fastsim
import copy
from ..base import EnvironmentState
from dataclasses import dataclass, field
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import wandb
import os
from pathlib import Path

OKABE_ITO_COLORS = [
    "#56B4E9",  
    "#E69F00",  
    "#009E73",  
    "#0072B2",
    "#CC79A7",    
    "#D55E00",  
    "#F0E442",  
    "#000000",
]

TRANSPARENT = np.array([0, 0, 0, 0], dtype=np.float32)
WHITE = np.array([1, 1, 1, 1], dtype=np.float32)
BLACK = np.array([0, 0, 0, 1], dtype=np.float32)

device_to_color = OKABE_ITO_COLORS.copy()

def create_okabe_ito_cmap():
    return mcolors.ListedColormap(OKABE_ITO_COLORS, name='okabe_ito')

def _auto_text_color(rgb: np.ndarray) -> np.ndarray:
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    return np.where(lum > 0.55, "black", "white")

@dataclass(slots=True)
class ColorConfig:
    # Device partition colors
    device_colors: Optional[list[str]] = field(default_factory=lambda: OKABE_ITO_COLORS.copy())         
    device_cmap: str | mcolors.Colormap = "tab20"     
    unknown_color: str = "#8a8a8a"                     
    running_color: str = "#808080"                     

    # Duration shading
    duration_cmap: str | mcolors.Colormap = "viridis" 
    duration_mode: Literal["overlay", "to_white", "duration_only"] = "to_white"
    duration_gamma: float = 0.4              
    duration_alpha: float = 0.65                       
    duration_percentile_max: float = 0.98             
    ema_tau_frames: int = 5    

@dataclass(slots=True)
class PlotConfig:
    fontsize: float = 25.0
    use_labels: bool = False
    use_blit: bool = False
    use_duration_shading: bool = True
    dpi: int = 300 
    bitrate: int = 300 
    figsize: tuple[float, float] = (8.0, 8.0)
    video_seconds: int = 30
    n_frames: int = 100


class PercentileEMANormalizer:
    def __init__(self, p: float = 0.98, tau: int = 30, eps: float = 1):
        self.p = float(p)
        self.tau = int(tau)
        self.eps = float(eps)
        self._vmax = None

    def update_and_get(self, x: np.ndarray) -> float:
        if x.size == 0:
            return (self._vmax if self._vmax is not None else 1.0)
        m = float(np.quantile(x, self.p)) if np.any(np.isfinite(x)) else 1.0
        if self._vmax is None:
            self._vmax = max(m, self.eps)
        else:
            self._vmax += (m - self._vmax) / max(self.tau, 1)
            self._vmax = max(self._vmax, self.eps)
        return self._vmax                

def _get_cmap(name_or_obj) -> mcolors.Colormap:
    return name_or_obj if isinstance(name_or_obj, mcolors.Colormap) else cm.get_cmap(name_or_obj)

def _build_device_palette(n_devices: int, cfg: ColorConfig) -> np.ndarray:
    unknown = np.array(mcolors.to_rgba(cfg.unknown_color), dtype=np.float32)
    if cfg.device_colors is not None and len(cfg.device_colors) > 0:
        listed = mcolors.ListedColormap(cfg.device_colors)
        base = listed(np.linspace(0, 1, max(n_devices, len(cfg.device_colors)))).astype(np.float32)
    else:
        cmap = _get_cmap(cfg.device_cmap)
        base = cmap(np.linspace(0, 1, max(n_devices, getattr(cmap, "N", n_devices)))).astype(np.float32)

    palette = np.empty((n_devices + 1, 4), dtype=np.float32)
    palette[0] = unknown
    palette[1:1+n_devices] = base[:n_devices]
    return palette

@dataclass(slots=True)
class EnvStaticState:
    n_compute_tasks: int
    n_data_tasks: int
    ct_duration_us: np.ndarray
    ct_device: np.ndarray
    ct_cell: np.ndarray
    ct_launch_time: np.ndarray
    ct_complete_time: np.ndarray

    dt_device: np.ndarray
    dt_source: np.ndarray
    dt_virtual: np.ndarray
    dt_block: np.ndarray
    dt_duration_us: np.ndarray
    dt_launch_time: np.ndarray 
    dt_complete_time: np.ndarray

@dataclass(slots=True)
class EnvDynamicState:
    ct_state: np.ndarray       
    dt_state: np.ndarray
    partition: np.ndarray 
    last_duration: np.ndarray
    last_label: list[str]
    ct_running: np.ndarray 
    ct_changed: np.ndarray 
    time: int
    last_time: int
    last_cell_update: np.ndarray

def get_total_work(static_state: EnvStaticState, device: int) -> float:
    mask = (static_state.ct_device == device)
    return float(np.sum(static_state.ct_duration_us[mask])) if np.any(mask) else 0.0

def get_total_work_in_interval(static_state: EnvStaticState, device: int, start_time: int, end_time: int) -> float:
    mask = (static_state.ct_device == device) & (static_state.ct_launch_time >= start_time) & (static_state.ct_complete_time <= end_time)
    return float(np.sum(static_state.ct_duration_us[mask])) if np.any(mask) else 0.0

def get_instantaneous_work(static_state: EnvStaticState, device: int, start_time: int, end_time: int) -> float:
    mask = (static_state.ct_device == device) & (static_state.ct_launch_time < end_time) & (static_state.ct_complete_time > start_time)
    total = 0.0
    for launch, complete, duration in zip(static_state.ct_launch_time[mask], static_state.ct_complete_time[mask], static_state.ct_duration_us[mask]):
        overlap_start = max(launch, start_time)
        overlap_end = min(complete, end_time)
        if overlap_start < overlap_end:
            overlap_duration = (overlap_end - overlap_start) / (complete - launch) * duration
            total += overlap_duration
    return float(total)

def get_total_in_communication(static_state: EnvStaticState, device: int) -> float:
    mask = (static_state.dt_device == device) & (~static_state.dt_virtual)
    return float(np.sum(static_state.dt_duration_us[mask])) if np.any(mask) else 0.0

def get_total_in_communication_in_interval(static_state: EnvStaticState, device: int, start_time: int, end_time: int) -> float:
    mask = (static_state.dt_device == device) & (~static_state.dt_virtual) & (static_state.dt_launch_time >= start_time) & (static_state.dt_complete_time <= end_time)
    return float(np.sum(static_state.dt_duration_us[mask])) if np.any(mask) else 0.0

def get_instantaneous_in_communication(static_state: EnvStaticState, device: int, start_time: int, end_time: int) -> float:
    mask = (static_state.dt_device == device) & (~static_state.dt_virtual) & (static_state.dt_launch_time < end_time) & (static_state.dt_complete_time > start_time)
    total = 0.0
    for launch, complete, duration in zip(static_state.dt_launch_time[mask], static_state.dt_complete_time[mask], static_state.dt_duration_us[mask]):
        overlap_start = max(launch, start_time)
        overlap_end = min(complete, end_time)
        if overlap_start < overlap_end:
            overlap_duration = (overlap_end - overlap_start) / (complete - launch) * duration
            total += overlap_duration
    return float(total)

def get_total_out_communication(static_state: EnvStaticState, device: int) -> float:
    mask = (static_state.dt_source == device) & (~static_state.dt_virtual)
    return float(np.sum(static_state.dt_duration_us[mask])) if np.any(mask) else 0.0

def get_total_out_communication_in_interval(static_state: EnvStaticState, device: int, start_time: int, end_time: int) -> float:
    mask = (static_state.dt_source == device) & (~static_state.dt_virtual) & (static_state.dt_launch_time >= start_time) & (static_state.dt_complete_time <= end_time)
    return float(np.sum(static_state.dt_duration_us[mask])) if np.any(mask) else 0.0


def get_instantaneous_out_communication(static_state: EnvStaticState, device: int, start_time: int, end_time: int) -> float:
    mask = (static_state.dt_source == device) & (~static_state.dt_virtual) & (static_state.dt_launch_time < end_time) & (static_state.dt_complete_time > start_time)
    total = 0.0
    for launch, complete, duration in zip(static_state.dt_launch_time[mask], static_state.dt_complete_time[mask], static_state.dt_duration_us[mask]):
        overlap_start = max(launch, start_time)
        overlap_end = min(complete, end_time)
        if overlap_start < overlap_end:
            overlap_duration = (overlap_end - overlap_start) / (complete - launch) * duration
            total += overlap_duration
    return float(total)

@dataclass(slots=True)
class LoadBalanceResult:
    load_balance: float
    in_comm_balance: float
    out_comm_balance: float
    total_work: float
    total_in_communication: float
    total_out_communication: float

def compute_load_balance(static_state: EnvStaticState, start_time: Optional[int] = None, end_time: Optional[int] = None) -> LoadBalanceResult:
    
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = max(np.max(static_state.ct_complete_time), np.max(static_state.dt_complete_time)) + 1

    compute_devices = set(static_state.ct_device[static_state.ct_device >= 0])
    data_devices = set(static_state.dt_device[static_state.dt_device >= 0]) | set(static_state.dt_source[static_state.dt_source >= 0])
    n_compute_devices = len(compute_devices)
    n_data_devices = len(data_devices)
    n_devices = max(compute_devices | data_devices) + 1 if (compute_devices | data_devices) else 0

    if n_devices <= 0:
        return LoadBalanceResult(
            load_balance=1.0,
            in_comm_balance=1.0,
            out_comm_balance=1.0,
            total_work=0.0,
            total_in_communication=0.0,
            total_out_communication=0.0,
        )
    
    work = np.zeros((n_devices,), dtype=np.float64)
    in_comm = np.zeros((n_devices,), dtype=np.float64)
    out_comm = np.zeros((n_devices,), dtype=np.float64)
    for device in range(n_devices):
        work[device] = get_instantaneous_work(static_state, device, start_time, end_time)
        in_comm[device] = get_instantaneous_in_communication(static_state, device, start_time, end_time)
        out_comm[device] = get_instantaneous_out_communication(static_state, device, start_time, end_time)

    

    avg_work = float(np.sum(work))/n_compute_devices if np.any(work) else 0.0
    avg_in_comm = float(np.sum(in_comm))/n_data_devices if np.any(in_comm) else 0.0
    avg_out_comm = float(np.sum(out_comm))/n_data_devices if np.any(out_comm) else 0.0
    max_work = float(np.max(work)) if np.any(work) else 0.0
    max_in_comm = float(np.max(in_comm)) if np.any(in_comm) else 0.0
    max_out_comm = float(np.max(out_comm)) if np.any(out_comm) else 0.0

    load_balance = (avg_work / max_work) if max_work > 0 else 1.0
    in_comm_balance = (avg_in_comm / max_in_comm) if max_in_comm > 0 else 1.0
    out_comm_balance = (avg_out_comm / max_out_comm) if max_out_comm > 0 else 1.0
    total_work = float(np.sum(work))
    total_in_communication = float(np.sum(in_comm))
    total_out_communication = float(np.sum(out_comm))

    return LoadBalanceResult(
        load_balance=load_balance,
        in_comm_balance=in_comm_balance,
        out_comm_balance=out_comm_balance,
        total_work=total_work,
        total_in_communication=total_in_communication,
        total_out_communication=total_out_communication,
    )

def load_balance_over_time(static_state: EnvStaticState, interval: int) -> list[LoadBalanceResult]:
    n_devices = max(np.max(static_state.ct_device), np.max(static_state.dt_device), np.max(static_state.dt_source)) + 1

    if n_devices <= 0:
        return []

    start_time = 0
    end_time = max(np.max(static_state.ct_complete_time), np.max(static_state.dt_complete_time)) + 1
    results = []
    for t in range(start_time, end_time, interval):
        lb = compute_load_balance(static_state, t, min(t + interval, end_time))
        results.append(lb)
    return results

def plot_load_balance_over_time(env, interval: int):
    static_state, dynamic_state = _build_state(env)
    results = load_balance_over_time(static_state, interval)

    times = [i * interval for i in range(len(results))]
    load_balances = [r.load_balance for r in results]
    in_comm_balances = [r.in_comm_balance for r in results]
    out_comm_balances = [r.out_comm_balance for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(times, load_balances, label='Load Balance', color='blue')
    plt.plot(times, in_comm_balances, label='In Communication Balance', color='orange')
    plt.plot(times, out_comm_balances, label='Out Communication Balance', color='green')
    plt.xlabel('Time (us)')
    plt.ylabel('Balance Ratio')
    plt.title('Load Balance Over Time')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()


@dataclass(slots=True)
class IdleType:
    compute: bool = True
    in_comm: bool = True
    out_comm: bool = True
    


def get_total_idle_time(static_state: EnvStaticState, idle_type: Optional[IdleType]=None, simulation_end_time: Optional[int] = None) -> np.ndarray:
    def _collect(device_count: int, idle_type: IdleType) -> list[list[tuple[int, int]]]:
        intervals = [[] for _ in range(device_count)]
        if idle_type.compute:
            for launch, finish, device in zip(static_state.ct_launch_time, static_state.ct_complete_time, static_state.ct_device):
                if device >= 0 and launch >= 0 and finish > launch:
                    intervals[device].append((int(launch), int(finish)))
        if idle_type.in_comm:
            for launch, finish, device, is_virtual in zip(static_state.dt_launch_time, static_state.dt_complete_time, static_state.dt_device, static_state.dt_virtual):
                if device >= 0 and not is_virtual and launch >= 0 and finish > launch:
                    intervals[device].append((int(launch), int(finish)))
        if idle_type.out_comm:
            for launch, finish, device, is_virtual in zip(static_state.dt_launch_time, static_state.dt_complete_time, static_state.dt_source, static_state.dt_virtual):
                if device >= 0 and not is_virtual and launch >= 0 and finish > launch:
                    intervals[device].append((int(launch), int(finish)))
        return intervals
    
    if idle_type is None:
        idle_type = IdleType(compute=True, in_comm=True, out_comm=True)
    
    compute_devices = set(static_state.ct_device[static_state.ct_device >= 0])
    data_devices = set(static_state.dt_device[static_state.dt_device >= 0]) | set(static_state.dt_source[static_state.dt_source >= 0])
    n_devices = max(compute_devices | data_devices) + 1 if (compute_devices | data_devices) else 0

    if n_devices <= 0:
        return np.zeros(0, dtype=np.float64)

    busy_intervals = _collect(n_devices, idle_type)
    total_end = int(simulation_end_time) if simulation_end_time is not None else max((end for dev in busy_intervals for _, end in dev), default=0)
    idle = np.full(n_devices, float(total_end), dtype=np.float64)

    for device, intervals in enumerate(busy_intervals):
        if not intervals:
            continue
        intervals.sort()
        busy = 0
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                busy += cur_end - cur_start
                cur_start, cur_end = start, end
        busy += cur_end - cur_start
        idle[device] = max(0.0, float(total_end - busy))
    return idle

def get_idle_from_env(env) -> np.ndarray:
    static_state, _ = _build_state(env)
    return get_total_idle_time(static_state, simulation_end_time=env.simulator.time if env.simulator is not None else None)


def _build_state(env) -> tuple[EnvStaticState, EnvDynamicState]:
    assert(env.simulator is not None)
    sim = env.simulator
    simulator_state = sim.state
    task_runtime = simulator_state.get_task_runtime()
    static_graph = simulator_state.get_tasks()

    graph = env.get_graph()

    n_compute_tasks = task_runtime.get_n_compute_tasks()
    n_data_tasks = task_runtime.get_n_data_tasks()

    ct_state = np.full((n_compute_tasks,), -1, dtype=np.int8)
    dt_state = np.full((n_data_tasks,), -1, dtype=np.int8)

    ct_device = np.full((n_compute_tasks,), -1, dtype=np.int32)
    ct_duration_us = np.zeros((n_compute_tasks,), dtype=np.float32)
    ct_launch_time = np.full((n_compute_tasks,), -1, dtype=np.int64)
    ct_complete_time = np.full((n_compute_tasks,), -1, dtype=np.int64)

    ct_cell = np.full((n_compute_tasks,), -1, dtype=np.int64)
    ct_changed = np.zeros((n_compute_tasks,), dtype=bool)

    dt_device = np.full((n_data_tasks,), -1, dtype=np.int32)
    dt_source = np.full((n_data_tasks,), -1, dtype=np.int32)
    dt_virtual = np.zeros((n_data_tasks,), dtype=bool)
    dt_block = np.full((n_data_tasks,), -1, dtype=np.int64)
    dt_duration_us = np.zeros((n_data_tasks,), dtype=np.float32)
    dt_launch_time = np.full((n_data_tasks,), -1, dtype=np.int64)
    dt_complete_time = np.full((n_data_tasks,), -1, dtype=np.int64)

    for i in range(n_compute_tasks):
        ct_device[i] = task_runtime.get_compute_task_mapped_device(i)
        ct_duration_us[i] = task_runtime.get_compute_task_duration(i)
        ct_cell[i] = graph.task_to_cell[i]
        ct_launch_time[i] = task_runtime.get_compute_task_launched_time(i)
        ct_complete_time[i] = task_runtime.get_compute_task_completed_time(i)

    for i in range(n_data_tasks):
        dt_device[i] = task_runtime.get_data_task_mapped_device(i)
        dt_source[i] = task_runtime.get_data_task_source_device(i)
        dt_virtual[i] = task_runtime.is_data_task_virtual(i)
        dt_block[i] = static_graph.get_data_id(i)
        dt_duration_us[i] = task_runtime.get_data_task_duration(i) if not dt_virtual[i] else 0.0
        dt_launch_time[i] = task_runtime.get_data_task_launched_time(i)
        dt_complete_time[i] = task_runtime.get_data_task_completed_time(i)

    static_state = EnvStaticState(
        n_compute_tasks=n_compute_tasks,
        n_data_tasks=n_data_tasks,
        ct_duration_us=ct_duration_us,
        ct_cell=ct_cell,
        ct_launch_time=ct_launch_time,
        ct_complete_time=ct_complete_time,
        ct_device=ct_device,
        dt_device=dt_device,
        dt_source=dt_source,
        dt_virtual=dt_virtual,
        dt_launch_time=dt_launch_time,
        dt_complete_time=dt_complete_time,
        dt_block=dt_block,    
        dt_duration_us=dt_duration_us)

    geom = env.get_graph().data.geometry
    n_cells = len(geom.cells)
    partition = np.full((n_cells,), -1, dtype=np.int32)
    last_duration = np.full((n_cells,), 0.0, dtype=np.float32)
    last_label = [""] * n_cells
    ct_running = np.zeros((n_compute_tasks,), dtype=bool)

    last_cell_update = np.full((n_cells,), -1, dtype=np.int64)

    dynamic_state = EnvDynamicState(
        last_time = -1,
        time = 0,
        ct_state=ct_state,
        dt_state=dt_state,
        partition=partition,
        last_duration=last_duration,
        last_label=last_label,
        ct_running=ct_running,
        ct_changed=ct_changed,
        last_cell_update=last_cell_update,
    )

    return static_state, dynamic_state

def _update_dynamic_state(env, time: int, static_state: EnvStaticState, dynamic_state: EnvDynamicState, gather_data_tasks: bool = True):
    assert(env.simulator is not None)
    sim = env.simulator
    simulator_state = sim.state
    task_runtime = simulator_state.get_task_runtime()
    assert(time <= sim.time)

    n_compute_tasks = static_state.n_compute_tasks
    n_data_tasks = static_state.n_data_tasks

    prev_state = dynamic_state.ct_state.copy()

    dynamic_state.ct_state.fill(-1)
    dynamic_state.ct_running.fill(0)

    dynamic_state.last_time = dynamic_state.time
    dynamic_state.time = time

    for i in range(n_compute_tasks):
        dynamic_state.ct_state[i] = task_runtime.get_compute_task_state_at_time(i, time)

    if gather_data_tasks:
        dynamic_state.dt_state.fill(-1)
        for i in range(n_data_tasks):
            dynamic_state.dt_state[i] = task_runtime.get_data_task_state_at_time(i, time)

    dynamic_state.ct_changed.fill(False)
    dynamic_state.ct_changed = (dynamic_state.ct_state != prev_state)
    dynamic_state.ct_running = (dynamic_state.ct_state == fastsim.TaskState.LAUNCHED)

    return dynamic_state


def _update_initial_partition(env, current_time: int, static_state: EnvStaticState, dynamic_state: EnvDynamicState, *, labels: bool = False):
    graph = env.get_graph()
    cell_locations = np.asarray(graph.get_cell_locations(as_dict=False), dtype=np.int32)


    if cell_locations is None or len(cell_locations) == 0:
        print("No initial partition found, using default partition.")
        dynamic_state.partition.fill(-1)
        dynamic_state.last_duration.fill(0.0)
        dynamic_state.last_label = [""] * len(dynamic_state.partition)
        return dynamic_state.partition, dynamic_state.last_duration, dynamic_state.last_label, np.empty((0,), dtype=np.int64)

    if len(cell_locations) != len(dynamic_state.partition):
        raise ValueError(f"Cell locations length {len(cell_locations)} does not match partition length {len(dynamic_state.partition)}")
    
    dynamic_state.partition[:] = cell_locations

    dynamic_state.last_duration.fill(0.0)
    dynamic_state.last_label = [""] * len(dynamic_state.partition)
    dynamic_state.last_cell_update.fill(current_time)
    changed_cells = np.arange(len(dynamic_state.partition), dtype=np.int64)

    if labels:
        for i, cell in enumerate(dynamic_state.partition):
            if 0 <= cell < len(graph.partitions):
                level = graph.task_to_level.get(int(i), -1)
                if hasattr(graph, "task_to_direction"):
                    direction = graph.task_to_direction.get(int(i), -2)
                    dynamic_state.last_label[i] = f"{level}:{direction}"
                else:
                    dynamic_state.last_label[i] = f"{level}"
    else:
        dynamic_state.last_label = [""] * len(dynamic_state.partition)

    dynamic_state.ct_running.fill(False)
    dynamic_state.ct_changed.fill(False)
    dynamic_state.ct_state.fill(-1)
    dynamic_state.dt_state.fill(-1)
    dynamic_state.ct_running = (dynamic_state.ct_state == fastsim.TaskState.LAUNCHED)

    return dynamic_state.partition, dynamic_state.last_duration, dynamic_state.last_label, changed_cells


def _update_dynamic_paritition(env, current_time: int,
                                    static_state: EnvStaticState,
                                    dynamic_state: EnvDynamicState,
                                    *,
                                    labels: bool = False):
    graph = env.get_graph()
    dy = dynamic_state

    changed_idx = np.where(
        dy.ct_changed & (
            (dy.ct_state == fastsim.TaskState.COMPLETED) |
            (dy.ct_state == fastsim.TaskState.LAUNCHED)
        )
    )[0]
    
    if changed_idx.size == 0:
        return dy.ct_running, dy.partition, dy.last_duration, dy.last_label, np.empty((0,), dtype=np.int64)

    launch_time = static_state.ct_launch_time[changed_idx]
    completed_time = static_state.ct_complete_time[changed_idx]
    
    valid_completed_mask = completed_time <= current_time
    valid_launched_mask = launch_time <= current_time

    launch_time[~valid_launched_mask] = -1
    completed_time[~valid_completed_mask] = -1

    changed_cells = static_state.ct_cell[changed_idx]
    changed_devices = static_state.ct_device[changed_idx]
    changed_durations = static_state.ct_duration_us[changed_idx]

    order = np.lexsort((completed_time, launch_time))
    changed_cells = changed_cells[order]
    changed_devices = changed_devices[order]
    changed_tasks = changed_idx[order]
    changed_durations = changed_durations[order]

    for cell, device, task, dur in zip(changed_cells, changed_devices, changed_tasks, changed_durations):
        if 0 <= cell < dy.partition.size:
            dy.partition[cell] = device
            dy.last_duration[cell] = float(dur)
            dy.last_cell_update[cell] = int(current_time)
            if labels:
                level = graph.task_to_level.get(int(task), -1)
                if hasattr(graph, "task_to_direction"):
                    direction = graph.task_to_direction.get(int(task), -2)
                    dy.last_label[cell] = f"{level}:{direction}"
                else:
                    dy.last_label[cell] = f"{level}"

    return dy.ct_running, dy.partition, dy.last_duration, dy.last_label, changed_cells

def _create_axes(_geom, _figsize=(8,8), pad=0.05):
    fig, ax = plt.subplots(figsize=_figsize)
    pts = _geom.cell_points
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def _save_animation(ani, path: str, dpi: Optional[int] = None, bitrate: Optional[int] = None):
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


def animate_mesh_execution(env, path: str, color_cfg: Optional[ColorConfig] = None, plot_cfg: Optional[PlotConfig] = None):
    color_cfg = color_cfg if color_cfg is not None else ColorConfig()
    plot_cfg = plot_cfg if plot_cfg is not None else PlotConfig()
    video_seconds = plot_cfg.video_seconds
    n_frames = plot_cfg.n_frames


    static_state, dynamic_state = _build_state(env)
    n_devices = int(max(0, static_state.ct_device.max() + 1))


    device_palette_rgba = _build_device_palette(n_devices, color_cfg)
    duration_cmap = _get_cmap(color_cfg.duration_cmap)
    dur_norm = PercentileEMANormalizer(
            p=color_cfg.duration_percentile_max,
            tau=color_cfg.ema_tau_frames
    )

    _update_dynamic_state(env, env.simulator.time, static_state, dynamic_state)
    _update_initial_partition(env, 0, static_state, dynamic_state, labels=plot_cfg.use_labels)

    geom = env.get_graph().data.geometry

    fig, ax = _create_axes(geom, _figsize=(8,8), pad=0.05)

    points = geom.cell_points
    cells = geom.cells 
    edges = geom.edges
    polys =  points[cells]

    face_colors = np.zeros((len(polys), 4), dtype=np.float32)
    edge_colors = np.zeros((len(edges), 4), dtype=np.float32)
    line_width = np.zeros((len(edges),), dtype=np.float32)

    part_index = (dynamic_state.partition.astype(np.int64) + 1).clip(0, device_palette_rgba.shape[0]-1)
    base_colors = device_palette_rgba[part_index]                 
    face_colors[:] = base_colors
    edge_colors[:] = mcolors.to_rgba("black")
    line_width[:] = 3.0
    
    interior_polys = PolyCollection(
        polys, facecolors=face_colors, edgecolors=edge_colors, linewidths=line_width, zorder=9, alpha=1, antialiased=False,
    )
    interior_polys.set_animated(True)

    centroids = polys.mean(axis=1)  # (n_cells, 2)
    label_artists = None
    if plot_cfg.use_labels:
        label_artists = np.empty((len(polys),), dtype=object)
        for i, (cx, cy) in enumerate(centroids):
            t = ax.text(cx, cy, "", ha="center", va="center",
                        fontsize=plot_cfg.fontsize, zorder=12, color="black", alpha=0.9)
            t.set_animated(True)
            label_artists[i] = t

    ax.add_collection(interior_polys)

    shade_fc = interior_polys.get_facecolors()
    shade_ec = interior_polys.get_edgecolors()
    shade_lw = interior_polys.get_linewidths()

    fps = max(1, round(n_frames / max(1, video_seconds)))
    

    T = int(env.simulator.time)
    if n_frames <= 1:
        time_list = np.array([T], dtype=np.int64)
    else:
        time_list = np.rint(np.linspace(0, T, n_frames)).astype(np.int64)
        time_list[-1] = T
    time_interval = int(env.simulator.time / n_frames)

    background = None 

    def frame_builder(frame):
        time = time_list[frame]
        #print(f"Frame {frame+1}/{n_frames}, time={time:.2f}/{env.simulator.time:.2f}")
        _update_dynamic_state(env, time, static_state, dynamic_state, gather_data_tasks=False)
        _, _, _, _, changed_cells = _update_dynamic_paritition(env, time, static_state, dynamic_state, labels=plot_cfg.use_labels)

        return dynamic_state.ct_running, dynamic_state.partition, dynamic_state.last_duration, dynamic_state.last_label, changed_cells

    def _cache_bg(_event=None):
        nonlocal background
        fig.canvas.draw()
        if plot_cfg.use_blit:
            background = fig.canvas.copy_from_bbox(ax.bbox)

    cid_resize = fig.canvas.mpl_connect("resize_event", _cache_bg)

    def _init():
        nonlocal background
        fig.canvas.draw()
        if plot_cfg.use_blit:
            background = fig.canvas.copy_from_bbox(ax.bbox)

        interior_polys.set_facecolors(shade_fc)

        if plot_cfg.use_labels and label_artists is not None:
            return tuple([interior_polys, *label_artists.tolist()])
        return (interior_polys,)
    
    max_duration = 100

    def _update(frame):
        nonlocal background 
        nonlocal shade_fc
        nonlocal max_duration

        ct_running, partition, last_duration, last_label, changed_cells = frame_builder(frame)

        part_index = (partition.astype(np.int64) + 1).clip(0, device_palette_rgba.shape[0]-1)
        base_colors = device_palette_rgba[part_index]                 
        shade_fc[:] = base_colors                                      

        if plot_cfg.use_duration_shading:
            vmax = dur_norm.update_and_get(last_duration)
            norm = np.clip((last_duration / vmax), 0.0, 1.0)
            norm = norm[:, None]
            
            norm[last_duration <= 0.0] = 1.0

            if frame<=1:
                shade_fc[:, :3] = base_colors[:, :3]
            else:
                if color_cfg.duration_mode == "overlay":
                    dur_colors = duration_cmap(norm.squeeze()).astype(np.float32)  
                    a = (color_cfg.duration_alpha * norm).astype(np.float32)       
                    shade_fc[:, :3] = (1.0 - a) * shade_fc[:, :3] + a * dur_colors[:, :3]
                elif color_cfg.duration_mode == "to_white":
                    WHITE = np.array([1, 1, 1], dtype=np.float32)
                    a = norm ** float(color_cfg.duration_gamma)
                    shade_fc[:, :3] = (a) * shade_fc[:, :3] + (1.0 - a) * WHITE[None, :]
                elif color_cfg.duration_mode == "duration_only":
                    dur_colors = duration_cmap(norm.squeeze()).astype(np.float32)
                    shade_fc[:, :3] = dur_colors[:, :3]


        running_task_idx = np.where(ct_running)[0]
        if running_task_idx.size > 0:
            running_cells = static_state.ct_cell[running_task_idx]
            running_cells = running_cells[(running_cells >= 0) & (running_cells < shade_fc.shape[0])]
            if running_cells.size > 0:
                shade_fc[running_cells] = mcolors.to_rgba(color_cfg.running_color)

        interior_polys.set_facecolors(shade_fc)

        artists = [interior_polys]
        if plot_cfg.use_labels and label_artists is not None and changed_cells.size > 0:
            text_colors = _auto_text_color(shade_fc[:, :3])
            for c in changed_cells:
                new_txt = dynamic_state.last_label[c]
                if new_txt is None:
                    new_txt = ""
                if label_artists[c].get_text() != new_txt:
                    label_artists[c].set_text(new_txt)
                label_artists[c].set_color(text_colors[c])
                artists.append(label_artists[c])

        return tuple(artists)

    ani = animation.FuncAnimation(
        fig, _update, init_func=_init, 
        frames=n_frames, 
        interval=time_interval, 
        blit=bool(plot_cfg.use_blit),
        repeat=False, 
        cache_frame_data=False,
    )

    ani._fig_ref = fig 
    ani._nframes = n_frames
    ani._fps = fps 
    ani._disconnect_resize = lambda: fig.canvas.mpl_disconnect(cid_resize)

    _save_animation(ani, path=path, dpi=plot_cfg.dpi, bitrate=plot_cfg.bitrate)

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

def animate_mesh_graph(
    env,
    plot_cfg: Optional[PlotConfig] = None,
    color_cfg: Optional[ColorConfig] = None,
    folder: Optional[str] = None,
    filename: str = "mesh_animation.mp4",
):
    if folder is None:
        run_dir = getattr(getattr(wandb, "run", None), "dir", None) if "wandb" in globals() else None
        folder_path = Path(run_dir) if run_dir else Path(".")
    else:
        folder_path = Path(folder)

    folder_path.mkdir(parents=True, exist_ok=True)
    out_path = folder_path / filename

    if plot_cfg is None:
        plot_cfg = PlotConfig()
    if color_cfg is None:
        color_cfg = ColorConfig()

    animate_mesh_execution(
        env=env,
        path=str(out_path),
        color_cfg=color_cfg,
        plot_cfg=plot_cfg,
    )
    return str(out_path)
