"""
bench_copy.py
Benchmark copy-heavy simulator usage patterns.

Focuses on:
1) raw SimulatorDriver.copy() cost
2) IncrementalSchedule-like copy -> drain/run cost
"""

import argparse
import json
import random
import statistics
import time

import numpy as np
import torch

import task4feedback.fastsim2 as fastsim
from task4feedback.fastsim2 import DequeueEFTMapper, ExecutionState, TaskNoise
from task4feedback.graphs.base import TrajectoryWorkload
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiConfig, DynamicJacobiGraph
from task4feedback.graphs.mesh import build_geometry, generate_quad_mesh
from task4feedback.interface.wrappers import SimulatorDriver, SimulatorInput, uniform_connected_devices

N_DEVICES = 5
GPU_MEM = 9600000000000
H2D_BW = 129000000000
D2D_BW = 54000000000
LATENCY = 1

SYSTEM_SPECS = dict(
    fastest_flops=67000000000000,
    slowest_flops=67000000000000,
    gpu_flop=67000000000000,
    fastest_gmbw=3350000000000,
    slowest_gmbw=3350000000000,
)

GRID_N = 16
STEPS = 256
LEVEL_MEM = 120e9

WORKLOAD_ARGS = dict(
    traj_type="circle",
    lower_bound=1,
    upper_bound=5,
    scale=0.3,
    traj_specifics=dict(radius=0.3, phase_length=128),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_system(mem: int = GPU_MEM):
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
        location_list=list(range(1, N_DEVICES)),
        step=0,
    )
    return graph


def make_sim_input(
    graph,
    system,
    expected_inflight_events: int = 0,
    expected_eviction_tasks: int = 0,
    expected_eviction_wave_keys: int = 0,
):
    d = graph.get_blocks()
    task_noise = TaskNoise(graph.static_graph)
    return SimulatorInput(
        graph,
        d,
        system,
        task_noise=task_noise,
        top_k_candidates=256,
        expected_inflight_events=expected_inflight_events,
        expected_eviction_tasks=expected_eviction_tasks,
        expected_eviction_wave_keys=expected_eviction_wave_keys,
    )


def prepare_partial_state(sim_input, prep_steps: int) -> SimulatorDriver:
    sim = SimulatorDriver(sim_input, internal_mapper=DequeueEFTMapper)
    sim.initialize()
    sim.initialize_data()
    sim.disable_external_mapper()
    if prep_steps > 0:
        sim.set_steps(prep_steps)
        status = sim.run()
        if status not in (ExecutionState.BREAKPOINT, ExecutionState.COMPLETE):
            raise RuntimeError(f"Unexpected status while preparing state: {status}")
    return sim


def quantile_sorted(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(q * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def report(name: str, vals_sec):
    vals_sorted = sorted(vals_sec)
    mean_ms = statistics.mean(vals_sec) * 1000.0
    p50_ms = quantile_sorted(vals_sorted, 0.50) * 1000.0
    p95_ms = quantile_sorted(vals_sorted, 0.95) * 1000.0
    min_ms = vals_sorted[0] * 1000.0
    max_ms = vals_sorted[-1] * 1000.0
    print(
        f"{name}: n={len(vals_sec)} mean={mean_ms:.3f}ms p50={p50_ms:.3f}ms "
        f"p95={p95_ms:.3f}ms min={min_ms:.3f}ms max={max_ms:.3f}ms"
    )
    return {
        "name": name,
        "n": int(len(vals_sec)),
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
    }


def bench_raw_copy(base_sim: SimulatorDriver, n: int):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = fastsim.Simulator(base_sim.simulator)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return report("raw_copy", times)


def bench_copy_and_drain(base_sim: SimulatorDriver, n: int, lookahead_steps: int):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sim_c = fastsim.Simulator(base_sim.simulator)
        sim_c.disable_python_mapper()
        if lookahead_steps > 0:
            sim_c.set_steps(int(lookahead_steps))
            _ = sim_c.run()
        sim_c.start_drain()
        _ = sim_c.run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return report("copy_plus_drain", times)

def bench_large_checkpoint_copy(sim_input: SimulatorInput, n: int, checkpoint_steps: int):
    print(f"Preparing large checkpoint state (steps={checkpoint_steps})...")
    base = prepare_partial_state(sim_input, checkpoint_steps)
    print(f"checkpoint_time_us={base.time}")
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = fastsim.Simulator(base.simulator)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return report("raw_copy_large_checkpoint", times)


def main():
    parser = argparse.ArgumentParser(description="Copy-path benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--copy-iters",
        type=int,
        default=200,
        help="Iterations for raw copy benchmark",
    )
    parser.add_argument(
        "--drain-iters",
        type=int,
        default=20,
        help="Iterations for copy+drain benchmark",
    )
    parser.add_argument(
        "--prep-steps",
        type=int,
        default=4096,
        help="Steps to advance base sim before benchmarking copies",
    )
    parser.add_argument(
        "--lookahead-steps",
        type=int,
        default=0,
        help="Optional set_steps on copied sim before drain",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=0,
        help="Optional second copy benchmark from a larger checkpointed state (0 disables)",
    )
    parser.add_argument("--expected-inflight-events", type=int, default=0)
    parser.add_argument("--expected-eviction-tasks", type=int, default=0)
    parser.add_argument("--expected-eviction-wave-keys", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    print("Building workload...")
    sys_for_graph = build_system()
    t0 = time.perf_counter()
    graph = build_graph(sys_for_graph)
    t1 = time.perf_counter()
    print(f"graph_build={t1 - t0:.3f}s tasks={len(graph)}")

    sim_input = make_sim_input(
        graph,
        build_system(),
        expected_inflight_events=args.expected_inflight_events,
        expected_eviction_tasks=args.expected_eviction_tasks,
        expected_eviction_wave_keys=args.expected_eviction_wave_keys,
    )
    print(f"Preparing partial state (prep_steps={args.prep_steps})...")
    base = prepare_partial_state(sim_input, args.prep_steps)
    print(f"prepared_time_us={base.time}")

    raw_copy_stats = bench_raw_copy(base, args.copy_iters)
    copy_plus_drain_stats = bench_copy_and_drain(base, args.drain_iters, args.lookahead_steps)

    large_checkpoint_stats = None
    if args.checkpoint_steps > 0:
        large_checkpoint_stats = bench_large_checkpoint_copy(
            sim_input, args.copy_iters, args.checkpoint_steps
        )

    if args.json_out:
        payload = {
            "benchmark": "bench_copy",
            "seed": int(args.seed),
            "copy_iters": int(args.copy_iters),
            "drain_iters": int(args.drain_iters),
            "prep_steps": int(args.prep_steps),
            "lookahead_steps": int(args.lookahead_steps),
            "checkpoint_steps": int(args.checkpoint_steps),
            "expected_inflight_events": int(args.expected_inflight_events),
            "expected_eviction_tasks": int(args.expected_eviction_tasks),
            "expected_eviction_wave_keys": int(args.expected_eviction_wave_keys),
            "raw_copy": raw_copy_stats,
            "copy_plus_drain": copy_plus_drain_stats,
            "raw_copy_large_checkpoint": large_checkpoint_stats,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
