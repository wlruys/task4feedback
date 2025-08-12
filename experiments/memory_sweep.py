from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import git
import hydra
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpi4py import MPI
import wandb  # kept because original imports it

from hydra.experimental.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from task4feedback.interface.wrappers import *           # noqa: F401,F403
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
)
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.partition import * 
from task4feedback.graphs.base import weighted_cell_partition

from helper.graph import make_graph_builder, GraphBuilder
from helper.env import make_env

font_scale = 1.75
mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * font_scale

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ParMETIS = ParMETIS_wrapper()

NUM_SAMPLES = 1

MetricKeys = ("time", "mem_usage", "total_mem_movement", "eviction_movement")

class GitInfo(Callback):
    """Hydra callback to snapshot Git state into the Hydra run directory."""
    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            outdir = Path(config.hydra.runtime.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "git_sha.txt").write_text(repo.head.commit.hexsha)
            (outdir / "git_dirty.txt").write_text(str(repo.is_dirty()))
            diff = repo.git.diff(None)
            (outdir / "git_diff.patch").write_text(diff)

            print(
                "Git SHA:",
                repo.head.commit.hexsha,
                " (dirty)" if repo.is_dirty() else " (clean)",
                flush=True,
            )
        except Exception as e:
            print(f"GitInfo callback failed: {e}")


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def factorize(steps: int, include_one: bool) -> List[int]:
    f: List[int] = []
    start = 1 if include_one else 0
    for i in range(start, steps // 3):
        if steps % (i + 1) == 0:
            f.append(i + 1)
    return f


def assign_colors(keys: Iterable[str]) -> Dict[str, any]:
    """Deterministic color assignment from tab10."""
    colors = {}
    for i, k in enumerate(keys):
        colors[k] = plt.cm.tab10(i)
    return colors


MapperFn = Callable[[DynamicJacobiGraph], LevelPartitionMapper]


def mapper_registry(cfg: DictConfig, d2d_bandwidth: int) -> Dict[str, Optional[MapperFn]]:
    """
    Returns a name->mapper function registry.
    `None` means the baseline EFT or ParMETIS which are handled specially.
    """
    def naive_mapper(graph: DynamicJacobiGraph) -> LevelPartitionMapper:
        graph.mincut_per_levels(
            bandwidth=d2d_bandwidth,
            mode="dynamic",
            offset=1,
            level_chunks=cfg.graph.config.steps // 5,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    def rr_mapper(_: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiRoundRobinMapper(
            n_devices=cfg.system.n_devices - 1,
            offset=1,
        )

    def global_min_cut_mapper(graph: DynamicJacobiGraph) -> LevelPartitionMapper:
        graph.mincut_per_levels(
            bandwidth=d2d_bandwidth,
            mode="metis",
            offset=1,
            level_chunks=1,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    def dynamic_metis_mapper(graph: DynamicJacobiGraph, level_chunks: int) -> LevelPartitionMapper:
        graph.mincut_per_levels(
            bandwidth=d2d_bandwidth,
            mode="metis",
            offset=1,
            level_chunks=level_chunks,
        )
        graph.align_partitions()
        return LevelPartitionMapper(level_cell_mapping=graph.partitions)

    return {
        "EFT": None,                    # baseline (no external mapper)
        "Naïve": naive_mapper,          # dynamic mode with chunks
        "Cyclic": rr_mapper,            # round-robin
        "GlobalMinCut": global_min_cut_mapper,
        "Oracle": None,                 # handled separately (dynamic k sweep using dynamic_metis_mapper)
        "ParMETIS": None,               # handled by distributed loop
        # Expose dynamic_metis factory for the Oracle path:
        "_dynamic_metis_factory": dynamic_metis_mapper,
    }




def init_metrics(names: Iterable[str], keys: Iterable[str] = MetricKeys) -> Dict[str, Dict[str, List[float]]]:
    return {name: {k: [] for k in keys} for name in names}


def append_zero_row(metrics: Dict[str, Dict[str, List[float]]], names: Iterable[str], keys: Iterable[str] = MetricKeys) -> None:
    for n in names:
        for k in keys:
            metrics[n][k].append(0.0)


def add_metric_row(metrics: Dict[str, Dict[str, List[float]]], name: str, sim: "SimulatorDriver", idx: int = -1) -> None:
    if "time" in metrics[name]:
        metrics[name]["time"][idx] += sim.time
    if "mem_usage" in metrics[name]:
        metrics[name]["mem_usage"][idx] += sim.max_mem_usage
    if "total_mem_movement" in metrics[name]:
        metrics[name]["total_mem_movement"][idx] += (sum(list(sim.total_data_movement())[1:]) / 4)
    if "eviction_movement" in metrics[name]:
        metrics[name]["eviction_movement"][idx] += (sum(list(sim.total_eviction_movement())[1:]) / 4)


def average_last(metrics: Dict[str, Dict[str, List[float]]], names: Iterable[str], keys: Iterable[str], num_samples: int) -> None:
    for n in names:
        for k in keys:
            if metrics[n][k]:
                metrics[n][k][-1] /= num_samples


def compute_speedup(metrics: Dict[str, Dict[str, List[float]]], baseline: str, compare: Iterable[str]) -> Dict[str, List[float]]:
    speedup = {k: [] for k in compare}
    base_times = metrics[baseline]["time"]
    for i, b in enumerate(base_times):
        for k in compare:
            speed = (b / metrics[k]["time"][i]) if b > 0 else 0.0
            speedup[k].append(speed)
    return speedup


@dataclass
class ProblemSizing:
    with_retire: int
    without_retire: int
    problem_size: int
    avg_step_data: int
    start_single_gpu: int
    step_size: int


def compute_problem_sizing(cfg: DictConfig, data_stat: Dict[str, int]) -> ProblemSizing:
    """Preserve the exact arithmetic and choices from the original script."""
    n = cfg.graph.config.n
    c = 1  # retire factor (unchanged)

    with_retire = (n**2) * (
        1 * data_stat["interior_average"]
        + 4 * data_stat["boundary_average"]
        + 4 * data_stat["boundary_average"]
        + 1 * data_stat["interior_average"]
        + 4 * data_stat["boundary_average"]
        + 1 * data_stat["interior_average"] * c
        + 4 * data_stat["boundary_average"] * c
    ) - 2 * (n * 4) * data_stat["boundary_average"]

    without_retire = (n**2) * (
        1 * data_stat["interior_average"]
        + 4 * data_stat["boundary_average"]
        + 4 * data_stat["boundary_average"]
        + 1 * data_stat["interior_average"]
        + 4 * data_stat["boundary_average"]
    ) - 2 * (n * 4) * data_stat["boundary_average"]

    avg_step = data_stat["average_step_data"]

    # Original choices (commented alternatives kept as comments).
    problem_size = with_retire
    single_gpu_mem_size = int(avg_step // 2)  # start
    single_gpu_mem_size = int((single_gpu_mem_size // 10000 + 1) * 10000)
    step_size = (problem_size - single_gpu_mem_size) // 30
    step_size = int((step_size // 10000 + 1) * 10000)

    return ProblemSizing(
        with_retire=with_retire,
        without_retire=without_retire,
        problem_size=problem_size,
        avg_step_data=avg_step,
        start_single_gpu=single_gpu_mem_size,
        step_size=step_size,
    )


def build_sweep_list(problem_size: int, start_single_gpu: int, step_size: int) -> Tuple[List[int], List[int]]:
    """
    Builds:
      - sweep_list: per-GPU memory sizes to test
      - total_mem_list: 4 * per-GPU memory (unchanged from original)
    """
    sweep_list: List[int] = []
    total_mem_list: List[int] = []

    single_gpu_mem = start_single_gpu
    print("Sweep List:")
    while 4 * single_gpu_mem / problem_size < 3:
        sweep_list.append(single_gpu_mem)
        print(f"{single_gpu_mem:,}")
        single_gpu_mem += step_size
    sweep_list.append(int(problem_size // 4))
    sweep_list = sorted(sweep_list)

    for per_gpu in sweep_list:
        total_mem_list.append(4 * per_gpu)

    return sweep_list, total_mem_list


def normalize_for_plots(metrics: Dict[str, Dict[str, List[float]]],
                        total_mem_list: List[int],
                        keys: Iterable[str]) -> None:
    """
    In-place normalization of mem_usage and movement metrics for plotting.
    """
    for i, m in enumerate(total_mem_list):
        scale = (m / 4)
        for k in metrics.keys():
            metrics[k]["mem_usage"][i] = metrics[k]["mem_usage"][i] / scale * 80
            metrics[k]["total_mem_movement"][i] = metrics[k]["total_mem_movement"][i] / scale * 80
            metrics[k]["eviction_movement"][i] = metrics[k]["eviction_movement"][i] / scale * 80


def plot_time(mem_x_axis, metrics, colors):
    fig0, ax0 = plt.subplots(figsize=(6, 6))
    for k in metrics.keys():
        ax0.plot(mem_x_axis, metrics[k]["time"], label=k, linestyle="-", color=colors[k])
    ax0.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax0.set_ylabel("Execution Time (s)")
    ax0.set_yscale("log")
    ax0.legend(loc="upper right")
    return fig0, ax0


def plot_speedup(mem_x_axis, speedup, colors, speedup_keys):
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    for k in speedup_keys:
        ax1.plot(mem_x_axis, speedup[k], label=f"{k}", color=colors[k])
    ax1.set_ylabel("Relative Speedup vs EFT")
    ax1.legend(loc="upper right")
    ax1.grid()
    return fig1, ax1


def plot_mem_usage(mem_x_axis, metrics, colors, keys_for_mem):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for k in keys_for_mem:
        ax2.plot(mem_x_axis, metrics[k]["mem_usage"], label=k, color=colors[k])
    ax2.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax2.set_ylabel("GB")
    ax2.legend(loc="upper right")
    ax2.set_title("Max Memory Usage")
    ax2.grid()
    return fig2, ax2


def plot_movement(mem_x_axis, metrics, colors, keys_for_move):
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    for k in keys_for_move:
        ax3.plot(mem_x_axis, metrics[k]["total_mem_movement"], label=k, linestyle="-", color=colors[k])
        ax3.plot(mem_x_axis, metrics[k]["eviction_movement"], linestyle="--", color=colors[k])

    ax3.set_xlabel("Total GPUs Memory Size / Problem Size")
    ax3.set_ylabel("GB")
    ax3.set_title("")
    ax3.set_yscale("log")
    ax3.grid()

    # primary legend (mappers)
    legend1 = ax3.legend(loc="upper right")
    ax3.add_artist(legend1)

    # secondary legend (styles)
    style_handles = [Line2D([0], [0], linestyle="-", color="black"),
                     Line2D([0], [0], linestyle="--", color="black")]
    style_labels = ["Total", "Eviction"]
    ax3.legend(handles=style_handles, labels=style_labels, loc="center right")
    return fig3, ax3


def plot_combined(mem_x_axis, metrics, colors, speedup, speedup_keys, mem_keys):
    # Combined 1×4 panel
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True)

    # (a) time
    for k in metrics.keys():
        axes[0].plot(mem_x_axis, metrics[k]["time"], label=k, linestyle="-", color=colors[k])
    axes[0].set_ylabel("Execution Time (s)")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", color="gray", linestyle="--", linewidth=0.5)
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("(a)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)

    # (b) speedup
    for k in speedup_keys:
        axes[1].plot(mem_x_axis, speedup[k], label=k, color=colors[k])
    axes[1].set_ylabel("Relative Speedup vs EFT")
    axes[1].legend(loc="upper right")
    axes[1].grid()
    axes[1].set_xlabel("(b)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)

    # (c) mem usage
    keys_for_mem = mem_keys
    for k in keys_for_mem:
        axes[2].plot(mem_x_axis, metrics[k]["mem_usage"], label=k, color=colors[k])
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("GB")
    axes[2].grid()
    axes[2].set_xlabel("(c)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)

    # # (d) movement
    # for k in keys_for_mem:
    #     axes[3].plot(mem_x_axis, metrics[k]["total_mem_movement"], linestyle="-", color=colors[k], label=k)
    #     axes[3].plot(mem_x_axis, metrics[k]["eviction_movement"], linestyle="--", color=colors[k])
    # legend1 = axes[3].legend(loc="upper right")
    # axes[3].add_artist(legend1)
    # style_handles = [Line2D([0], [0], linestyle="-", color="black"),
    #                  Line2D([0], [0], linestyle="--", color="black")]
    # style_labels = ["Total", "Eviction"]
    # axes[3].legend(handles=style_handles, labels=style_labels, loc="lower left")
    # axes[3].set_ylabel("GB")
    # axes[3].set_xlabel("(d)", fontsize=mpl.rcParams["font.size"] / font_scale * 2.25)
    # axes[3].set_yscale("log")
    # axes[3].grid()

    fig.supxlabel(
        "Total GPUs Memory Size / Problem Size",
        fontsize=mpl.rcParams["font.size"] / font_scale * 2.25,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])

    return fig, axes


# =====================================================================
# ParMETIS distributed mapping (preserved logic)
# =====================================================================

def run_parmetis_distributed(
    cfg: DictConfig,
    sweep_list: List[int],
    num_samples: int,
    d2d_bandwidth: int,
    metrics: Dict[str, Dict[str, List[float]]],
) -> None:
    """
    Executes the ParMETIS portion with MPI, accumulating results in `metrics["ParMETIS"]`.
    """
    # Only rank 0 needs an env
    if rank == 0:
        env = make_env(graph_builder=make_graph_builder(cfg), cfg=cfg, normalization=False)
    else:
        env = None
    partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for sweep_idx, single_gpu_mem_size in enumerate(sweep_list):
        if rank == 0:
            s = uniform_connected_devices(
                n_devices=cfg.system.n_devices,
                latency=cfg.system.latency,
                h2d_bw=cfg.system.h2d_bw,
                d2d_bw=cfg.system.d2d_bw,
                mem=single_gpu_mem_size,
            )
            env.simulator_factory[0].input.system = s

        for _ in range(NUM_SAMPLES):
            done = False
            if rank == 0:
                env._reset()
                sim: SimulatorDriver = env.simulator_factory[0].create()
                graph = sim.input.graph

                # initial partition
                cell_graph = graph.get_weighted_cell_graph(
                    DeviceType.GPU,
                    bandwidth=d2d_bandwidth,
                    levels=[0, 1],
                )
                edge_cut, partition = weighted_cell_partition(
                    cell_graph, nparts=(cfg.system.n_devices - 1)
                )
                cell_loc = [x + 1 for x in partition]

                partition = [-1 for _ in range(sim.observer.graph_spec.max_candidates)]
                sim.enable_external_mapper()
                done = (sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE)

                candidates = torch.zeros(
                    (sim.observer.graph_spec.max_candidates), dtype=torch.int64
                )
                sim.get_mappable_candidates(candidates)
                actions = []
                for i, id in enumerate(candidates):
                    mapping_priority = sim.get_mapping_priority(id)
                    actions.append(
                        fastsim.Action(
                            i,
                            cell_loc[graph.task_to_cell[id.item()]],
                            mapping_priority,
                            mapping_priority,
                        )
                    )
                sim.simulator.map_tasks(actions)
                done = (sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE)

            while True:
                if rank == 0 and not done:
                    sim.get_mappable_candidates(candidates)
                    # partition candidates according to current cell_loc
                    for i, id in enumerate(candidates):
                        partition[i] = cell_loc[graph.task_to_cell[id.item()]] - 1  # offset by CPU=0

                    (
                        partitioned_tasks,
                        vtxdist,
                        xadj,
                        adjncy,
                        vwgt,
                        adjwgt,
                        vsize,
                    ) = graph.get_distributed_weighted_graph(
                        bandwidth=d2d_bandwidth,
                        task_ids=candidates.tolist(),
                        partition=partition,
                        future_levels=0,
                    )

                done = comm.bcast(done, root=0)
                if done:
                    break

                # distribute graph pieces
                vtxdist = comm.bcast(vtxdist, root=0)
                xadj = comm.bcast(xadj, root=0)
                adjncy = comm.bcast(adjncy, root=0)
                vwgt = comm.bcast(vwgt, root=0)
                adjwgt = comm.bcast(adjwgt, root=0)
                vsize = comm.bcast(vsize, root=0)

                xadj_local = xadj[rank]
                adjncy_local = adjncy[rank]
                vwgt_local = vwgt[rank]
                adjwgt_local = adjwgt[rank]
                vsize_local = vsize[rank]

                wgtflag = 3
                numflag = 0
                ncon = 1
                tpwgts = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
                ubvec = np.array([1.225], dtype=np.float32)
                itr = 10000.0
                part = np.array([-1 for _ in range(64)], dtype=np.int32)

                comm.Barrier()
                ParMETIS.callParMETIS(
                    vtxdist,
                    xadj_local,
                    adjncy_local,
                    vwgt_local,
                    vsize_local,
                    adjwgt_local,
                    wgtflag,
                    numflag,
                    ncon,
                    tpwgts,
                    ubvec,
                    itr,
                    part,
                )
                parts = comm.gather(part, root=0)

                if rank == 0:
                    for i_p, p in enumerate(parts):
                        for j, dev in enumerate(p):
                            if dev == -1:
                                break
                            task_id = partitioned_tasks[i_p][j]
                            cell_loc[graph.task_to_cell[task_id]] = int(dev) + 1  # GPU devices are 1..4

                    actions = []
                    for i, id in enumerate(candidates):
                        mapping_priority = sim.get_mapping_priority(id)
                        actions.append(
                            fastsim.Action(
                                i,
                                cell_loc[graph.task_to_cell[id.item()]],
                                mapping_priority,
                                mapping_priority,
                            )
                        )
                    sim.simulator.map_tasks(actions)
                    done = (sim.run_until_external_mapping() == fastsim.ExecutionState.COMPLETE)

            if rank == 0:
                print(sim.time)
                add_metric_row(metrics, "ParMETIS", sim, sweep_idx)

    if rank == 0:
        # average ParMETIS metrics over samples
        for i in range(len(sweep_list)):
            for key in MetricKeys:
                metrics["ParMETIS"][key][i] /= num_samples


def run_host_experiments_and_plot(cfg: DictConfig):
    d2d_bandwidth = cfg.system.d2d_bw

    if rank == 0:
        seed_everything(0)

        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)

        experiment_mappers = mapper_registry(cfg, d2d_bandwidth)
        
        experiment_names = ["EFT", "Naïve", "Cyclic", "GlobalMinCut", "Oracle", "ParMETIS"]
        speedup_keys = ["Cyclic", "GlobalMinCut", "Oracle", "ParMETIS"]
        mem_keys = ["EFT", "Cyclic", "GlobalMinCut", "Oracle", "ParMETIS"]

        if size < 4:
            print("ParMETIS requires at least 4 ranks. Removing it from experiments.")
            if "ParMETIS" in experiment_names:
                experiment_names.remove("ParMETIS")
            if "ParMETIS" in speedup_keys:
                speedup_keys.remove("ParMETIS")
            if "ParMETIS" in mem_keys:
                mem_keys.remove("ParMETIS")
        
        colors = assign_colors(experiment_names)

        # --- Data stats & sizing
        assert isinstance(env.simulator_factory[0].input.graph, DynamicJacobiGraph)
        data_stat = env.simulator_factory[0].input.graph.data.data_stat

        with_retire, without_retire = None, None
        sizing = compute_problem_sizing(cfg, data_stat)
        with_retire = sizing.with_retire
        without_retire = sizing.without_retire

        print(f"with Retire Memory Size: {with_retire:,}")
        print(f"without Retire Memory Size: {without_retire:,}")
        print(f"Average Step Data Size: {sizing.avg_step_data:,}")

        sweep_list, total_mem_list = build_sweep_list(
            problem_size=sizing.problem_size,
            start_single_gpu=sizing.start_single_gpu,
            step_size=sizing.step_size,
        )

        # factorization for Oracle k
        include_one = "GlobalMinCut" in experiment_names
        f = factorize(cfg.graph.config.steps, include_one=include_one)
        print(f"Factors of {cfg.graph.config.steps}: {f}")

        # --- Metrics
        metrics = init_metrics(experiment_names, MetricKeys)
        dynamic_metis_k_best: List[int] = []

        print(
            f"Memory,{str.join(',', [str(m) for m in experiment_names])}",
            flush=True,
        )

        # ---- Sweep host-side (everything but ParMETIS)
        for single_gpu_mem_size in sweep_list:
            # system config for this memory size
            s = uniform_connected_devices(
                n_devices=cfg.system.n_devices,
                latency=cfg.system.latency,
                h2d_bw=cfg.system.h2d_bw,
                d2d_bw=cfg.system.d2d_bw,
                mem=single_gpu_mem_size,
            )
            env.simulator_factory[0].input.system = s

            # per-k Oracle metrics
            metis_metrics = init_metrics(f, MetricKeys)

            # prepare accumulation bins for this memory point
            append_zero_row(metrics, experiment_names, MetricKeys)
            append_zero_row(metis_metrics, f, MetricKeys)

            for _ in range(NUM_SAMPLES):
                env._reset()

                # baseline graph and simulator
                graph = env.simulator_factory[0].input.graph
                sim_base = env.simulator_factory[0].create()

                # ---- EFT
                sim = sim_base.copy()
                sim.disable_external_mapper()
                sim.run()
                add_metric_row(metrics, "EFT", sim)

                # ---- Naïve / Cyclic / GlobalMinCut
                for name in ["Naïve", "Cyclic", "GlobalMinCut"]:
                    mapper_fn = experiment_mappers[name]
                    assert mapper_fn is not None
                    sim = sim_base.copy()
                    mapper = mapper_fn(graph)
                    sim.external_mapper = mapper
                    sim.enable_external_mapper()
                    sim.run()
                    add_metric_row(metrics, name, sim)

                # ---- Oracle (dynamic METIS over factors)
                dynamic_metis_factory = experiment_mappers["_dynamic_metis_factory"]
                assert dynamic_metis_factory is not None
                for k in f:
                    sim = sim_base.copy()
                    mapper = dynamic_metis_factory(graph, level_chunks=k)
                    sim.external_mapper = mapper
                    sim.enable_external_mapper()
                    sim.run()
                    add_metric_row(metis_metrics, k, sim)

            # --- Average over samples
            average_last(metrics, experiment_names, MetricKeys, NUM_SAMPLES)
            average_last(metis_metrics, f, MetricKeys, NUM_SAMPLES)

            # --- Pick best k for Oracle at this memory
            if 1 not in f:
                min_time = metrics["GlobalMinCut"]["time"][-1]
                best_k = 1
            else:
                min_time = metis_metrics[1]["time"][-1]
                best_k = 1
            for k in f:
                if k > 1 and metis_metrics[k]["time"][-1] < min_time:
                    min_time = metis_metrics[k]["time"][-1]
                    best_k = k

            dynamic_metis_k_best.append(best_k)
            # Copy best to "Oracle"
            if best_k != 1:
                for key in MetricKeys:
                    metrics["Oracle"][key][-1] = metis_metrics[best_k][key][-1]
            else:
                for key in MetricKeys:
                    metrics["Oracle"][key][-1] = metrics["GlobalMinCut"][key][-1]

            print(f"{single_gpu_mem_size:_},", end="")
            for name in experiment_names:
                if name == "Oracle":
                    print(
                        f"{int(metrics[name]['time'][-1])}({dynamic_metis_k_best[-1]})",
                        end=",",
                    )
                else:
                    print(int(metrics[name]["time"][-1]), end=",")
            print("")

    # ---- Sync, broadcast sweep list to all ranks (for ParMETIS phase)
    comm.Barrier()
    sweep_list = comm.bcast(sweep_list if rank == 0 else None, root=0)
    comm.Barrier()
    if "ParMETIS" in experiment_names:
        run_parmetis_distributed(
            cfg=cfg,
            sweep_list=sweep_list,
            num_samples=NUM_SAMPLES,
            d2d_bandwidth=cfg.system.d2d_bw,
            metrics=metrics if rank == 0 else None,
        )

    # ---- Post-processing, plots, and saving (rank 0 only)
    if rank == 0:
        # Compute speedup curves
        speedup = compute_speedup(metrics, baseline="EFT", compare=speedup_keys)

        # Find problem_size pivot for x-axis scaling (preserved behavior)
        # Use the first index where mem usage spread >= 0.1%
        problem_size = sizing.problem_size
        for i in range(len(total_mem_list)):
            min_val = 2**60
            max_val = 1
            for name in metrics:
                if metrics[name]["mem_usage"][i] < min_val:
                    min_val = metrics[name]["mem_usage"][i]
                if metrics[name]["mem_usage"][i] > max_val:
                    max_val = metrics[name]["mem_usage"][i]
            print(
                f"Memory Usage at {total_mem_list[i]}: min={min_val}, max={max_val}, diff={(max_val - min_val)/min_val:.3f}"
            )
            if (max_val - min_val) >= 0.001 * min_val:
                problem_size = total_mem_list[i]
                break

        mem_x_axis = [m / problem_size for m in total_mem_list]

        fig0, ax0 = plot_time(mem_x_axis, metrics, colors)
        fig1, ax1 = plot_speedup(mem_x_axis, speedup, colors, speedup_keys)

        normalize_for_plots(metrics, total_mem_list, MetricKeys)

        # Memory usage
        fig2, ax2 = plot_mem_usage(
            mem_x_axis, metrics, colors, keys_for_mem=mem_keys
        )

        # Movement
        fig3, ax3 = plot_movement(
            mem_x_axis, metrics, colors, keys_for_move=mem_keys
        )

        # Combined panel
        fig, axes = plot_combined(mem_x_axis, metrics, colors, speedup, speedup_keys, mem_keys)

        saved_lines = (
            f"mem_x_axis={mem_x_axis}\n"
            f"metrics={metrics}\n"
            f"colors={colors}\n"
            f"speedup_keys={speedup_keys}\n"
            f"speedup={speedup}\n"
        )
        print(saved_lines)

        # -------- Save Figures & Logs
        file_name = f"{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}"
        try:
            file_name += f"_{cfg.graph.config.workload_args.scale}"
        except AttributeError:
            pass
        file_name += f"_{cfg.graph.config.workload_args.upper_bound}w{cfg.graph.config.workload_args.lower_bound}"
        file_name += f"_{cfg.graph.config.workload_args.traj_type}"
        print(f"File Name: {file_name}")

        # Directory layout
        base = Path("outputs") / file_name
        (base / "time").mkdir(parents=True, exist_ok=True)
        (base / "speed").mkdir(parents=True, exist_ok=True)
        (base / "mem").mkdir(parents=True, exist_ok=True)
        (base / "log").mkdir(parents=True, exist_ok=True)
        (base / "move").mkdir(parents=True, exist_ok=True)

        # Save figures
        fig0.savefig(base / f"time/time_{file_name}.png")
        fig1.savefig(base / f"speed/speed_{file_name}.png")
        fig2.savefig(base / f"mem/mem_{file_name}.png")
        fig3.savefig(base / f"move/move_{file_name}.png")
        fig.savefig(base / f"combined_{file_name}.png")

        # Save results text
        with open(base / f"log/result_{file_name}.txt", "w") as ftxt:
            ftxt.write(OmegaConf.to_yaml(cfg.graph))
            ftxt.write(f"Problem Size: {sizing.problem_size/data_stat['average_step_data']}xAvgStep\n")
            ftxt.write(f"Problem Size: {sizing.problem_size}\n")
            ftxt.write(f"with Retire Memory Size: {sizing.with_retire}\n")
            ftxt.write(f"without Retire Memory Size: {sizing.without_retire}\n")
            ftxt.write(
                "PerGPUMemory,TotalMem/ProblemSize," + ",".join([str(i) for i in metrics.keys()]) + "\n"
            )
            for idx in range(len(metrics["EFT"]["time"])):
                line = f"{int(total_mem_list[idx]/4)},{total_mem_list[idx]/sizing.problem_size:.2f}"
                for k in metrics.keys():
                    # print Oracle time + best k
                    if k == "Oracle":
                        line += f",{int(metrics[k]['time'][idx])}({dynamic_metis_k_best[idx]})"
                    else:
                        line += f",{int(metrics[k]['time'][idx])}"
                line += "\n"
                ftxt.write(line)
            ftxt.write("\n")
            ftxt.write(saved_lines)

@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    run_host_experiments_and_plot(cfg)


if __name__ == "__main__":
    main()