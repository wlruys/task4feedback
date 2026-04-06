from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_fastsim_extension_from_env() -> None:
    extension_path = os.environ.get("TASK4FEEDBACK_FASTSIM2_PATH")
    if not extension_path:
        return

    spec = importlib.util.spec_from_file_location("task4feedback.fastsim2", extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {extension_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["task4feedback.fastsim2"] = module
    sys.modules["fastsim2"] = module
    spec.loader.exec_module(module)


_load_fastsim_extension_from_env()

import bench_eviction_cholesky as cholesky_bench
import bench_eviction_jacobi as jacobi_bench
from bench_mapper_support import (
    ALL_MAPPER_NAMES,
    ExternalMapperConfig,
    MemoryAwareEFTConfig,
    TransitionConfig,
    is_external_mapper,
    make_external_mapper,
    make_internal_mapper,
    mapper_label,
)
from task4feedback.fastsim2 import EvictionPolicy, ExecutionState
from task4feedback.graphs.mesh.plot import ColorConfig, PlotConfig, animate_mesh_graph
from task4feedback.interface.wrappers import SimulatorDriver


@dataclass
class MeshVizEnv:
    simulator: SimulatorDriver

    def get_graph(self):
        return self.simulator.input.graph


def _parse_csv_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _resolve_eviction_policy(name: str) -> EvictionPolicy:
    policy_name = name.lower()
    if policy_name == "lru":
        return EvictionPolicy.LRU
    if policy_name in {"least_used_mapped", "lum"}:
        return EvictionPolicy.LEAST_USED_MAPPED
    raise ValueError(f"Unsupported eviction policy '{name}'")


def _validate_mapper_names(mapper_names: list[str]) -> list[str]:
    unknown = [name for name in mapper_names if name not in ALL_MAPPER_NAMES]
    if unknown:
        supported = ", ".join(ALL_MAPPER_NAMES)
        raise ValueError(f"Unsupported mapper(s): {unknown}. Supported values: {supported}")
    return mapper_names


def _sanitize_token(value: str) -> str:
    return value.lower().replace(" ", "_").replace(",", "_")


def _default_stem(benchmark: str, mapper_name: str, eviction_policy: str) -> str:
    return f"bench_eviction_{benchmark}_{_sanitize_token(mapper_name)}_{_sanitize_token(eviction_policy)}"


def _resolve_stem(
    *,
    benchmark: str,
    mapper_name: str,
    eviction_policy: str,
    requested_stem: Optional[str],
    multiple_mappers: bool,
) -> str:
    if requested_stem is None:
        return _default_stem(benchmark, mapper_name, eviction_policy)
    if multiple_mappers:
        return f"{requested_stem}_{_sanitize_token(mapper_name)}"
    return requested_stem


def _reset_initial_cell_locations(graph) -> None:
    graph.set_cell_locations([-1 for _ in range(len(graph.data.geometry.cells))], step=0)


def _make_transition_config(args: argparse.Namespace) -> TransitionConfig:
    return TransitionConfig(
        kind=args.transition_kind,
        planned_threshold=args.transition_planned_threshold,
        max_reserved_threshold=args.transition_max_reserved_threshold,
        batch_size=args.transition_batch_size,
        queue_threshold=args.transition_queue_threshold,
        max_in_flight=args.transition_max_in_flight,
        mapped_reserved_gap=args.transition_mapped_reserved_gap,
        reserved_launched_gap=args.transition_reserved_launched_gap,
        total_in_flight=args.transition_total_in_flight,
    )


def _make_external_mapper_config(args: argparse.Namespace) -> ExternalMapperConfig:
    default_block_span = 2 if args.benchmark == "jacobi" else 1
    return ExternalMapperConfig(
        block_rows=default_block_span if args.block_rows is None else args.block_rows,
        block_cols=default_block_span if args.block_cols is None else args.block_cols,
        processor_rows=args.processor_rows,
        processor_cols=args.processor_cols,
    )


def _make_memory_aware_config(args: argparse.Namespace) -> MemoryAwareEFTConfig:
    return MemoryAwareEFTConfig(
        alpha=args.memory_aware_eft_alpha,
        eviction_cost_location_state=args.memory_aware_location_state,
        overflow_state=args.memory_aware_overflow_state,
        overflow_mode=args.memory_aware_overflow_mode,
    )


def _build_driver(
    sim_input,
    *,
    mapper_name: str,
    memory_aware_config: MemoryAwareEFTConfig,
    external_mapper_config: ExternalMapperConfig,
    n_gpu_devices: int,
) -> SimulatorDriver:
    if is_external_mapper(mapper_name):
        driver = SimulatorDriver(sim_input)
        external_mapper = make_external_mapper(
            mapper_name,
            sim_input.graph,
            n_gpu_devices=n_gpu_devices,
            config=external_mapper_config,
        )
        driver.initialize()
        driver.initialize_data()
        driver.enable_external_mapper(external_mapper)
        return driver

    internal_mapper = make_internal_mapper(
        mapper_name,
        memory_aware_eft_alpha=memory_aware_config.alpha,
        memory_aware_eft_config=memory_aware_config,
    )
    driver = SimulatorDriver(sim_input, internal_mapper=internal_mapper)
    driver.initialize()
    driver.initialize_data()
    driver.disable_external_mapper()
    return driver


def _build_jacobi_input(
    args: argparse.Namespace,
    *,
    mapper_name: str,
    eviction_policy: EvictionPolicy,
    transition_config: TransitionConfig,
):
    graph_system = jacobi_bench.build_system(mem=int(args.gpu_mem_gb * 1e9))
    graph = jacobi_bench.build_graph(
        graph_system,
        grid_n=args.grid_n,
        steps=args.steps,
        level_memory=int(args.level_memory_gb * 1e9),
        domain_ratio=args.domain_ratio,
        arithmetic_intensity=args.arithmetic_intensity,
        arithmetic_complexity=args.arithmetic_complexity,
        boundary_complexity=args.boundary_complexity,
        memory_intensity=args.memory_intensity,
        boundary_width=args.boundary_width,
        r_interior=args.r_interior,
        r_boundary=args.r_boundary,
        workload_scale=args.workload_scale,
        workload_lower_bound=args.workload_lower_bound,
        workload_upper_bound=args.workload_upper_bound,
        workload_phase_length=args.workload_phase_length,
        randomize_initial_placement=not args.no_randomize_locations,
    )
    sim_system = jacobi_bench.build_system(mem=int(args.gpu_mem_gb * 1e9))
    sim_input = jacobi_bench.make_sim_input(
        graph,
        sim_system,
        mapper_name=mapper_name,
        eviction_policy=eviction_policy,
        top_k_candidates=args.top_k_candidates,
        transition_config=transition_config,
    )
    return sim_input


def _build_cholesky_input(
    args: argparse.Namespace,
    *,
    mapper_name: str,
    eviction_policy: EvictionPolicy,
    transition_config: TransitionConfig,
):
    graph_system = cholesky_bench.build_system(mem=int(args.gpu_mem_gb * 1e9))
    graph = cholesky_bench.build_graph(
        graph_system,
        n_blocks=args.n_blocks,
        block_bytes=cholesky_bench.align_block_bytes(int(args.block_bytes_gb * 1e9), 4),
        randomize_initial_placement=not args.no_randomize_locations,
        verbose=args.verbose_graph,
    )
    sim_system = cholesky_bench.build_system(mem=int(args.gpu_mem_gb * 1e9))
    sim_input = cholesky_bench.make_sim_input(
        graph,
        sim_system,
        mapper_name,
        eviction_policy,
        transition_config,
    )
    return sim_input


def _build_sim_input(
    args: argparse.Namespace,
    *,
    benchmark: str,
    mapper_name: str,
    eviction_policy: EvictionPolicy,
    transition_config: TransitionConfig,
):
    if benchmark == "jacobi":
        return _build_jacobi_input(
            args,
            mapper_name=mapper_name,
            eviction_policy=eviction_policy,
            transition_config=transition_config,
        )
    if benchmark == "cholesky":
        return _build_cholesky_input(
            args,
            mapper_name=mapper_name,
            eviction_policy=eviction_policy,
            transition_config=transition_config,
        )
    raise ValueError(f"Unsupported benchmark '{benchmark}'")


def convert_mp4_to_gif(mp4_path: Path, gif_path: Path, fps: int, width: int) -> None:
    palette_path = gif_path.with_suffix(".palette.png")
    vf_base = f"fps={fps},scale={width}:-1:flags=lanczos"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            f"{vf_base},palettegen",
            str(palette_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-i",
            str(palette_path),
            "-lavfi",
            f"{vf_base}[x];[x][1:v]paletteuse",
            str(gif_path),
        ],
        check=True,
    )
    palette_path.unlink(missing_ok=True)


def _run_single_animation(
    args: argparse.Namespace,
    *,
    benchmark: str,
    mapper_name: str,
    transition_config: TransitionConfig,
    external_mapper_config: ExternalMapperConfig,
    memory_aware_config: MemoryAwareEFTConfig,
    multiple_mappers: bool,
) -> tuple[Path, Optional[Path], int]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if benchmark == "jacobi":
        jacobi_bench.set_seed(args.seed)
    else:
        cholesky_bench.set_seed(args.seed)
    stem = _resolve_stem(
        benchmark=benchmark,
        mapper_name=mapper_name,
        eviction_policy=args.eviction_policy,
        requested_stem=args.stem,
        multiple_mappers=multiple_mappers,
    )
    mp4_path = args.output_dir / f"{stem}.mp4"
    gif_path = None if args.skip_gif else args.output_dir / f"{stem}.gif"

    eviction_policy = _resolve_eviction_policy(args.eviction_policy)
    sim_input = _build_sim_input(
        args,
        benchmark=benchmark,
        mapper_name=mapper_name,
        eviction_policy=eviction_policy,
        transition_config=transition_config,
    )
    driver = _build_driver(
        sim_input,
        mapper_name=mapper_name,
        memory_aware_config=memory_aware_config,
        external_mapper_config=external_mapper_config,
        n_gpu_devices=jacobi_bench.N_DEVICES - 1,
    )

    status = driver.run()
    if status != ExecutionState.COMPLETE:
        raise RuntimeError(f"Unexpected simulator status for mapper={mapper_name}: {status}")

    if not args.keep_initial_locations:
        _reset_initial_cell_locations(sim_input.graph)

    plot_cfg = PlotConfig(
        use_labels=False,
        use_blit=False,
        use_duration_shading=False,
        dpi=args.dpi,
        bitrate=args.bitrate,
        figsize=(8.0, 8.0),
        video_seconds=args.video_seconds,
        n_frames=args.frames,
    )
    color_cfg = ColorConfig()
    viz_env = MeshVizEnv(driver)

    animate_mesh_graph(
        viz_env,
        plot_cfg=plot_cfg,
        color_cfg=color_cfg,
        folder=str(args.output_dir),
        filename=mp4_path.name,
    )

    if gif_path is not None:
        fps = max(1, round(args.frames / max(1, args.video_seconds)))
        convert_mp4_to_gif(mp4_path, gif_path, fps=fps, width=args.gif_width)

    return mp4_path, gif_path, driver.time


def build_parser(
    *,
    default_benchmark: str = "jacobi",
    default_mappers: str = "kahypar",
    default_stem: Optional[str] = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render bench_eviction animations for one or more mapper methods"
    )
    parser.add_argument(
        "--benchmark",
        choices=("jacobi", "cholesky"),
        default=default_benchmark,
        help="Benchmark family to render",
    )
    parser.add_argument(
        "--mappers",
        default=default_mappers,
        help="Comma-separated mapper list to animate",
    )
    parser.add_argument(
        "--eviction-policy",
        choices=("lru", "least_used_mapped"),
        default="lru",
        help="Eviction policy used for the render",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--video-seconds", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--bitrate", type=int, default=3000)
    parser.add_argument("--gif-width", type=int, default=900)
    parser.add_argument("--skip-gif", action="store_true", help="Skip GIF generation and keep only MP4 output")
    parser.add_argument(
        "--keep-initial-locations",
        action="store_true",
        help="Keep randomized initial data placement instead of starting from unknown locations",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--stem", type=str, default=default_stem)

    parser.add_argument("--gpu-mem-gb", type=float, default=jacobi_bench.GPU_MEM / 1e9)
    parser.add_argument(
        "--transition-kind",
        choices=("auto", "planned", "default", "batch", "range", "hysteresis"),
        default="auto",
    )
    parser.add_argument("--transition-planned-threshold", type=int, default=1)
    parser.add_argument("--transition-max-reserved-threshold", type=int, default=16)
    parser.add_argument("--transition-batch-size", type=int, default=5)
    parser.add_argument("--transition-queue-threshold", type=int, default=5)
    parser.add_argument("--transition-max-in-flight", type=int, default=None)
    parser.add_argument("--transition-mapped-reserved-gap", type=int, default=5)
    parser.add_argument("--transition-reserved-launched-gap", type=int, default=5)
    parser.add_argument("--transition-total-in-flight", type=int, default=None)
    parser.add_argument("--block-rows", type=int, default=None)
    parser.add_argument("--block-cols", type=int, default=None)
    parser.add_argument("--processor-rows", type=int, default=None)
    parser.add_argument("--processor-cols", type=int, default=None)
    parser.add_argument("--memory-aware-eft-alpha", type=float, default=1.0)
    parser.add_argument(
        "--memory-aware-location-state",
        choices=("launched", "reserved", "mapped"),
        default="mapped",
    )
    parser.add_argument(
        "--memory-aware-overflow-state",
        choices=("reserved", "mapped"),
        default="reserved",
    )
    parser.add_argument(
        "--memory-aware-overflow-mode",
        choices=("full_spill", "incoming_only"),
        default="full_spill",
    )
    parser.add_argument(
        "--no-randomize-locations",
        action="store_true",
        help="Keep all initial data on CPU before simulation",
    )

    parser.add_argument("--grid-n", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--level-memory-gb", type=float, default=120.0)
    parser.add_argument("--top-k-candidates", type=int, default=jacobi_bench.TOP_K_CANDIDATES)
    parser.add_argument("--domain-ratio", type=float, default=1.0)
    parser.add_argument("--arithmetic-intensity", type=float, default=595.5555555)
    parser.add_argument("--arithmetic-complexity", type=float, default=1.0)
    parser.add_argument("--boundary-complexity", type=float, default=1.0)
    parser.add_argument("--memory-intensity", type=float, default=0.0)
    parser.add_argument("--boundary-width", type=float, default=0.25)
    parser.add_argument("--r-interior", type=float, default=10.0)
    parser.add_argument("--r-boundary", type=float, default=0.1)
    parser.add_argument("--workload-scale", type=float, default=0.3)
    parser.add_argument("--workload-lower-bound", type=float, default=1.0)
    parser.add_argument("--workload-upper-bound", type=float, default=5.0)
    parser.add_argument("--workload-phase-length", type=int, default=128)

    parser.add_argument("--n-blocks", type=int, default=24)
    parser.add_argument("--block-bytes-gb", type=float, default=1.0)
    parser.add_argument(
        "--verbose-graph",
        action="store_true",
        help="Enable Cholesky graph construction logging",
    )

    return parser


def main(
    *,
    default_benchmark: str = "jacobi",
    default_mappers: str = "kahypar",
    default_stem: Optional[str] = None,
) -> None:
    parser = build_parser(
        default_benchmark=default_benchmark,
        default_mappers=default_mappers,
        default_stem=default_stem,
    )
    args = parser.parse_args()

    mapper_names = _validate_mapper_names(_parse_csv_list(args.mappers))
    if not mapper_names:
        raise ValueError("At least one mapper must be provided via --mappers")

    transition_config = _make_transition_config(args)
    external_mapper_config = _make_external_mapper_config(args)
    memory_aware_config = _make_memory_aware_config(args)

    print(f"Benchmark: {args.benchmark}")
    print(f"Mappers: {mapper_names}")
    print(f"Eviction policy: {args.eviction_policy}")
    if "memory_aware_eft" in mapper_names:
        print(
            "MemoryAwareEFT config: "
            f"{mapper_label('memory_aware_eft', memory_aware_eft_alpha=args.memory_aware_eft_alpha, memory_aware_eft_config=memory_aware_config)}"
        )

    multiple_mappers = len(mapper_names) > 1
    for mapper_name in mapper_names:
        mp4_path, gif_path, sim_time_us = _run_single_animation(
            args,
            benchmark=args.benchmark,
            mapper_name=mapper_name,
            transition_config=transition_config,
            external_mapper_config=external_mapper_config,
            memory_aware_config=memory_aware_config,
            multiple_mappers=multiple_mappers,
        )
        print()
        print(f"Mapper: {mapper_name}")
        print(f"Label: {mapper_label(mapper_name, memory_aware_eft_alpha=args.memory_aware_eft_alpha, memory_aware_eft_config=memory_aware_config)}")
        print(f"MP4: {mp4_path}")
        if gif_path is not None:
            print(f"GIF: {gif_path}")
        print(f"Simulated time: {sim_time_us}")


if __name__ == "__main__":
    main()
