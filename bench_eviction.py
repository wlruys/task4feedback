"""
bench_eviction.py
Benchmark: 8x8 grid, 256-step dynamic circle Jacobi stencil
           with limited per-GPU memory (forces eviction pressure).
"""

import argparse
import random
import statistics
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
    BatchTransitionConditions,
    TaskNoise,
    ExecutionState,
)

N_DEVICES   = 9        
GPU_MEM     = 96000000000    
H2D_BW      = 129000000000  
D2D_BW      =  54000000000  
LATENCY     = 1           

SYSTEM_SPECS = dict(
    fastest_flops=67000000000000,
    slowest_flops=67000000000000,
    gpu_flop=67000000000000,
    fastest_gmbw=3350000000000,
    slowest_gmbw=3350000000000,
)

GRID_N     = 16
STEPS      = 256
LEVEL_MEM  = 120e9

WORKLOAD_ARGS = dict(
    traj_type="circle",
    lower_bound=1,
    upper_bound=5,
    scale=0.3,
    traj_specifics=dict(radius=0.3, phase_length=128),
)

N_REPS = 3 
DEFAULT_SEED = 0
DEFAULT_N_SEEDS = 1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_system(mem=GPU_MEM):
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
    graph.randomize_locations(
        1.0,
        location_list=[1, 2, 3, 4, 5, 6, 7, 8],
        step=0,
    )
    return graph


def make_sim_input(graph, system):
    d = graph.get_blocks()
    transition_conditions = BatchTransitionConditions(5, 5, 256)
    task_noise = TaskNoise(graph.static_graph)
    sim_input = SimulatorInput(
        graph,
        d,
        system,
        task_noise=task_noise,
        transition_conditions=transition_conditions,
        top_k_candidates=256,
    )
    return sim_input


def run_once(sim_input: SimulatorInput):
    fresh = SimulatorDriver(sim_input, internal_mapper=DequeueEFTMapper)
    fresh.initialize()
    fresh.initialize_data()
    fresh.disable_external_mapper()

    t0 = time.perf_counter()
    status = fresh.run()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    sim_us  = fresh.time
    max_mem = fresh.max_mem_usage
    total_mv = sum(fresh.total_data_movement())
    evict_mv = sum(fresh.total_eviction_movement())

    return status, wall_s, sim_us, max_mem, total_mv, evict_mv


def _bytes_hr(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def run_scenario(label, graph, system_mem, n_reps=N_REPS):
    tight_system = build_system(mem=system_mem)
    sim_input = make_sim_input(graph, tight_system)

    print(f"\n  [{label}]  GPU mem={_bytes_hr(system_mem)}")
    wall_times = []
    sim_times = []
    last_evict = 0

    for rep in range(n_reps):
        status, wall_s, sim_us, max_mem, total_mv, evict_mv = run_once(sim_input)
        if status != ExecutionState.COMPLETE:
            print(f"    Rep {rep+1}: FAILED (status={status})")
            return
        wall_times.append(wall_s)
        sim_times.append(sim_us)
        last_evict = evict_mv
        print(
            f"    Rep {rep+1}/{n_reps}:  "
            f"wall={wall_s:.3f}s  sim={sim_us/1e6:.3f}s  "
            f"max_mem={_bytes_hr(max_mem)}  "
            f"total_mv={_bytes_hr(total_mv)}  "
            f"evict_mv={_bytes_hr(evict_mv)}"
        )

    best_w = min(wall_times)
    avg_w  = sum(wall_times) / len(wall_times)
    avg_s  = sum(sim_times) / len(sim_times)
    print(
        f"    => best_wall={best_w:.3f}s  avg_wall={avg_w:.3f}s  "
        f"avg_sim={avg_s/1e6:.3f}s  evict={_bytes_hr(last_evict)}"
    )
    return best_w, avg_w, avg_s, last_evict


def main():
    parser = argparse.ArgumentParser(description="Eviction stress benchmark")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base RNG seed")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of consecutive seeds to run (seed, seed+1, ...)",
    )
    parser.add_argument("--reps", type=int, default=N_REPS, help="Repetitions per seed")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Benchmark: {GRID_N}x{GRID_N} grid, {STEPS}-step circle Jacobi")
    print(f"Level data size : {_bytes_hr(LEVEL_MEM)}")
    print(f"GPUs            : {N_DEVICES - 1}")
    print(f"Repetitions     : {args.reps}")
    print(f"Base seed       : {args.seed}")
    print(f"Seed count      : {args.n_seeds}")
    print("=" * 60)

    aggregated = []
    for seed_offset in range(args.n_seeds):
        seed = args.seed + seed_offset
        print(f"\nSeed {seed}")
        set_seed(seed)
        t0 = time.perf_counter()
        gen_system = build_system()
        graph = build_graph(gen_system)
        build_time = time.perf_counter() - t0
        print(f"      Build time: {build_time:.1f}s   Tasks: {len(graph)}")
        result = run_scenario("(H100)", graph, GPU_MEM, n_reps=args.reps)
        if result is not None:
            best_w, avg_w, avg_s, _ = result
            aggregated.append((seed, best_w, avg_w, avg_s))

    if len(aggregated) > 1:
        bests = [x[1] for x in aggregated]
        avgs = [x[2] for x in aggregated]
        sims = [x[3] for x in aggregated]
        print("\nAcross-seed summary")
        print(f"  best_wall median={statistics.median(bests):.3f}s  min={min(bests):.3f}s")
        print(f"  avg_wall  median={statistics.median(avgs):.3f}s  min={min(avgs):.3f}s")
        print(f"  avg_sim   median={statistics.median(sims)/1e6:.3f}s")

    print("=" * 60)


if __name__ == "__main__":
    main()
