"""
bench_mapper_compare.py
Compare DequeueEFTMapper (baseline) vs MemoryAwareEFTMapper on the 16x16
circle Jacobi workload under tight GPU memory to force eviction pressure.
"""

import argparse
import random
import time
import numpy as np
import torch

from task4feedback.graphs.mesh import generate_quad_mesh, build_geometry
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph, DynamicJacobiConfig
from task4feedback.graphs.base import TrajectoryWorkload
from task4feedback.interface.wrappers import (
    uniform_connected_devices,
    SimulatorInput,
    SimulatorDriver,
)
from task4feedback.fastsim2 import (
    DequeueEFTMapper,
    MemoryAwareEFTMapper,
    BatchTransitionConditions,
    TaskNoise,
    ExecutionState,
)

N_DEVICES  = 5
H2D_BW     = 129_000_000_000
D2D_BW     =  54_000_000_000
LATENCY    = 1

SYSTEM_SPECS = dict(
    fastest_flops=67_000_000_000_000,
    slowest_flops=67_000_000_000_000,
    gpu_flop=67_000_000_000_000,
    fastest_gmbw=3_350_000_000_000,
    slowest_gmbw=3_350_000_000_000,
)

GRID_N    = 16
STEPS     = 256
LEVEL_MEM = 120e9

WORKLOAD_ARGS = dict(
    traj_type="circle",
    lower_bound=1,
    upper_bound=5,
    scale=0.3,
    traj_specifics=dict(radius=0.3, phase_length=128),
)

# Tight memory scenarios (per GPU) to probe eviction behaviour
TIGHT_MEMS = [
    ("20 GB", 20_000_000_000),
    ("15 GB", 15_000_000_000),
]

N_REPS       = 3
DEFAULT_SEED = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_system(mem):
    return uniform_connected_devices(
        n_devices=N_DEVICES,
        mem=mem,
        latency=LATENCY,
        h2d_bw=H2D_BW,
        d2d_bw=D2D_BW,
        h2d_links=1,
        d2d_links=1,
        cpu_copyengines=2,
        device_copyengines=2,
        system_specs=SYSTEM_SPECS,
    )


def build_graph(system):
    mesh = generate_quad_mesh(L=1, n=GRID_N)
    geom = build_geometry(mesh)
    config = DynamicJacobiConfig(
        n=GRID_N,
        steps=STEPS,
        level_memory=LEVEL_MEM,
        domain_ratio=1.0,
        arithmetic_intensity=595.5555555,
        arithmetic_complexity=1.0,
        boundary_complexity=1,
        memory_intensity=0,
        boundary_width=0.25,
        morton_priority_enabled=False,
        r_interior=10,
        r_boundary=0.1,
        vcu_usage=1.0,
        task_internal_memory=0,
        workload=TrajectoryWorkload(),
        workload_args=WORKLOAD_ARGS,
    )
    graph = DynamicJacobiGraph(geom, config, system=system)
    graph.randomize_locations(1.0, location_list=list(range(1, N_DEVICES)), step=0)
    return graph


def make_sim_input(graph, system):
    d = graph.get_blocks()
    transition_conditions = BatchTransitionConditions(5, 5, 256)
    task_noise = TaskNoise(graph.static_graph)
    return SimulatorInput(
        graph,
        d,
        system,
        task_noise=task_noise,
        transition_conditions=transition_conditions,
        top_k_candidates=256,
    )


def run_once(sim_input, mapper_cls, mapper_kwargs=None):
    mapper_kwargs = mapper_kwargs or {}
    fresh = SimulatorDriver(
        sim_input,
        internal_mapper=mapper_cls(**mapper_kwargs) if mapper_kwargs else mapper_cls,
    )
    fresh.initialize()
    fresh.initialize_data()
    fresh.disable_external_mapper()

    t0 = time.perf_counter()
    status = fresh.run()
    wall_s = time.perf_counter() - t0

    sim_us   = fresh.time
    max_mem  = fresh.max_mem_usage
    total_mv = sum(fresh.total_data_movement())
    evict_mv = sum(fresh.total_eviction_movement())

    rt = fresh.state.get_task_runtime()
    n_evict_tasks = rt.get_n_eviction_tasks()
    n_evict_moves = sum(
        1 for i in range(n_evict_tasks) if not rt.is_eviction_task_virtual(i)
    )

    return status, wall_s, sim_us, max_mem, total_mv, evict_mv, n_evict_tasks, n_evict_moves


def _hr(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def run_mapper(label, sim_input, mapper_cls, mapper_kwargs, n_reps):
    wall_times, sim_times = [], []
    last = {}
    for rep in range(n_reps):
        status, wall_s, sim_us, max_mem, total_mv, evict_mv, n_et, n_em = run_once(
            sim_input, mapper_cls, mapper_kwargs
        )
        if status != ExecutionState.COMPLETE:
            print(f"      [{label}] rep {rep+1}: FAILED (status={status})")
            return None
        wall_times.append(wall_s)
        sim_times.append(sim_us)
        last = dict(max_mem=max_mem, total_mv=total_mv, evict_mv=evict_mv,
                    n_et=n_et, n_em=n_em)
        print(
            f"      [{label}] rep {rep+1}/{n_reps}: "
            f"wall={wall_s:.3f}s  sim={sim_us/1e6:.3f}s  "
            f"evict_mv={_hr(evict_mv)}  evict_tasks={n_et}(mv={n_em})"
        )

    return dict(
        best_wall=min(wall_times),
        avg_wall=sum(wall_times)/len(wall_times),
        avg_sim=sum(sim_times)/len(sim_times),
        **last,
    )


def compare(mem_label, mem_bytes, graph, n_reps):
    system = build_system(mem_bytes)
    sim_input = make_sim_input(graph, system)

    print(f"\n  --- GPU mem = {mem_label} ---")

    mappers = [
        ("DequeueEFT (baseline)", DequeueEFTMapper, {}),
        ("MemoryAwareEFT α=1.0",  MemoryAwareEFTMapper, {"alpha": 1.0}),
        ("MemoryAwareEFT α=2.0",  MemoryAwareEFTMapper, {"alpha": 2.0}),
    ]

    results = {}
    for label, cls, kwargs in mappers:
        r = run_mapper(label, sim_input, cls, kwargs, n_reps)
        if r is not None:
            results[label] = r

    if not results:
        return

    print(f"\n  Summary ({mem_label}):")
    print(f"  {'Mapper':<30} {'avg_sim':>10} {'best_wall':>10} {'evict_mv':>12} {'evict_tasks':>12} {'evict_moves':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    baseline_sim = results.get("DequeueEFT (baseline)", {}).get("avg_sim", None)
    for label, r in results.items():
        delta = ""
        if baseline_sim and label != "DequeueEFT (baseline)":
            pct = (r["avg_sim"] - baseline_sim) / baseline_sim * 100
            delta = f"  ({pct:+.1f}% sim)"
        print(
            f"  {label:<30} {r['avg_sim']/1e6:>9.3f}s {r['best_wall']:>9.3f}s "
            f" {_hr(r['evict_mv']):>12} {r['n_et']:>12} {r['n_em']:>12}{delta}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reps", type=int, default=N_REPS)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Mapper comparison: {GRID_N}x{GRID_N} grid, {STEPS}-step circle Jacobi")
    print(f"Level data size : {_hr(LEVEL_MEM)}   GPUs: {N_DEVICES-1}   seed: {args.seed}")
    print("=" * 70)

    set_seed(args.seed)
    gen_system = build_system(max(m for _, m in TIGHT_MEMS))
    t0 = time.perf_counter()
    graph = build_graph(gen_system)
    print(f"Graph built in {time.perf_counter()-t0:.1f}s   Tasks: {len(graph)}")

    for mem_label, mem_bytes in TIGHT_MEMS:
        compare(mem_label, mem_bytes, graph, args.reps)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
