import random
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from datetime import datetime
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
    JacobiQuadrantMapper
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

MetricKeys = ("time", "mem_usage", "total_mem_movement", "eviction_movement", "time_history")
# experiment_names = ["EFT", "ColWise", "ParMETIS", "GlobalMinCut", "Cyclic", "Oracle"]
# experiment_names = ["EFT", "ColWise", "GlobalMinCut", "Cyclic"]
# experiment_names = ["EFT", "ColWise", "RL"]
experiment_names = ["EFT", "Oracle", "GlobalMinCut"]
# experiment_names = ["EFT", "ParMETIS"]
mem_keys = experiment_names.copy()
speedup_keys = experiment_names.copy()
speedup_keys.remove("EFT")

sweep_list = list(range(int(80e9), int(120e9), int(5e9)))

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

    def row_mapper(_: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiRoundRobinMapper(
            n_devices=cfg.system.n_devices - 1,
            offset=0,
        )
    
    def cyclic_mapper(_: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiRoundRobinMapper(
            n_devices=cfg.system.n_devices - 1,
            offset=1,
        )

    def quadrant_mapper(graph: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiQuadrantMapper(
            n_devices=cfg.system.n_devices - 1,
            graph=graph
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
        "ColWise": row_mapper,            # round-robin
        "Cyclic": cyclic_mapper,             # cyclic
        "Quad": quadrant_mapper,        # quadrant
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
            if k in ["time_history"]:
                metrics[n][k].append([])
            else:
                metrics[n][k].append(0.0)


def add_metric_row(metrics: Dict[str, Dict[str, List[float]]], name: str, sim: "SimulatorDriver", idx: int = -1) -> None:
    if "time" in metrics[name]:
        metrics[name]["time"][idx] += sim.time
    if "time_history" in metrics[name]:
        metrics[name]["time_history"][idx].append(sim.time)
    if "mem_usage" in metrics[name]:
        metrics[name]["mem_usage"][idx] += sim.max_mem_usage
    if "total_mem_movement" in metrics[name]:
        metrics[name]["total_mem_movement"][idx] += (sum(list(sim.total_data_movement())[1:]) / 4)
    if "eviction_movement" in metrics[name]:
        metrics[name]["eviction_movement"][idx] += (sum(list(sim.total_eviction_movement())[1:]) / 4)


def average_last(metrics: Dict[str, Dict[str, List[float]]], names: Iterable[str], keys: Iterable[str], num_samples: int) -> None:
    for n in names:
        for k in keys:
            if metrics[n][k] and k != "time_history":
                metrics[n][k][-1] /= num_samples

# =====================================================================
# ParMETIS distributed mapping
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
    partitioned_tasks, vtxdist, xadj, adjncy, vwgt, adjwgt, vsize = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for sweep_idx, sweep_entry in enumerate(sweep_list):
        if rank == 0:
            cfg.graph.config.level_memory = sweep_entry
            graph_builder = make_graph_builder(cfg, verbose=False)
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)

        for _ in range(NUM_SAMPLES):
            done = False
            if rank == 0:
                env._reset()
                sim: SimulatorDriver = env.simulator.copy()
                eft_sim = sim.copy()
                eft_sim.disable_external_mapper()
                eft_sim.run()
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
                itr = 1000.0
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
                add_metric_row(metrics, "ParMETIS", sim, sweep_idx)
                assert metrics["EFT"]["time_history"][sweep_idx][len(metrics["ParMETIS"]["time_history"][sweep_idx])-1] == eft_sim.time, "Mismatch in EFT time history"

    if rank == 0:
        # average ParMETIS metrics over samples
        for i in range(len(sweep_list)):
            for key in MetricKeys:
                if key != "time_history":
                    metrics["ParMETIS"][key][i] /= num_samples


def run_host_experiments_and_plot(cfg: DictConfig):
    d2d_bandwidth = cfg.system.d2d_bw
    global sweep_list
    if size < 4:
        if rank == 0:
            print("ParMETIS requires at least 4 ranks. Removing it from experiments.")
        if "ParMETIS" in experiment_names:
            experiment_names.remove("ParMETIS")
        if "ParMETIS" in speedup_keys:
            speedup_keys.remove("ParMETIS")
        if "ParMETIS" in mem_keys:
            mem_keys.remove("ParMETIS")
    if rank == 0:
        seed_everything(cfg.seed)
        
        if "RL" in experiment_names:
            from task4feedback.ml.models import FeatureDimConfig
            from helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
            graph_builder = make_graph_builder(cfg, verbose=False)
            env = make_env(graph_builder=graph_builder, cfg=cfg, eval=True, normalization=False)
            observer = env.get_observer()
            feature_config = FeatureDimConfig.from_observer(observer)
            model, lstm = create_td_actor_critic_models(cfg, feature_config)
            ckpt_path = Path(__file__).resolve().parent / "model.pt"
            if ckpt_path.exists():
                loaded = load_policy_from_checkpoint(model, ckpt_path)
                if not loaded:
                    print(f"Found {ckpt_path}, but no compatible policy module to load into.")
                    exit()
            else:
                print(f"No model checkpoint found at {ckpt_path}")
                exit()

        experiment_mappers = mapper_registry(cfg, d2d_bandwidth)
        
        colors = assign_colors(experiment_names)

        # factorization for Oracle k
        include_one = "GlobalMinCut" in experiment_names
        if "Oracle" in experiment_names:
            f = factorize(cfg.graph.config.steps, include_one=include_one)
            print(f"Factors of {cfg.graph.config.steps}: {f}")
        
        print(sweep_list)

        # --- Metrics
        metrics = init_metrics(experiment_names, MetricKeys)
        dynamic_metis_k_best: List[int] = []

        print(
            f"Memory,{str.join(',', [str(m) for m in experiment_names])}",
            flush=True,
        )

        # ---- Sweep host-side (everything but ParMETIS)
        for sweep_entry in sweep_list:
            # system config for this memory size
            cfg.graph.config.level_memory = sweep_entry
            graph_builder = make_graph_builder(cfg, verbose=False)
            if "RL" in experiment_names:
                env, norm = make_env(graph_builder=graph_builder, cfg=cfg, normalization=None, eval=True)
                env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm, eval=True)
            else:
                env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True)

            # per-k Oracle metrics
            if "Oracle" in experiment_names:
                metis_metrics = init_metrics(f, MetricKeys)

            # prepare accumulation bins for this memory point
            append_zero_row(metrics, experiment_names, MetricKeys)
            if "Oracle" in experiment_names:
                append_zero_row(metis_metrics, f, MetricKeys)

            for _ in range(NUM_SAMPLES):
                obs = env._reset()

                # baseline graph and simulator
                graph = env.simulator_factory[0].input.graph
                sim_base = env.simulator.copy()

                # ---- EFT
                sim = sim_base.copy()
                sim.disable_external_mapper()
                sim.run()
                add_metric_row(metrics, "EFT", sim)
                eft_time = sim.time

                # ---- Naïve / ColWise / GlobalMinCut
                for name in ["Naïve", "ColWise", "GlobalMinCut", "Quad", "Cyclic"]:
                    if name in experiment_names:
                        mapper_fn = experiment_mappers[name]
                        assert mapper_fn is not None
                        sim = sim_base.copy()
                        mapper = mapper_fn(graph)
                        sim.external_mapper = mapper
                        sim.enable_external_mapper()
                        sim.run()
                        add_metric_row(metrics, name, sim)

                # ---- Oracle (dynamic METIS over factors)
                if "Oracle" in experiment_names:
                    dynamic_metis_factory = experiment_mappers["_dynamic_metis_factory"]
                    assert dynamic_metis_factory is not None
                    for k in f:
                        sim = sim_base.copy()
                        mapper = dynamic_metis_factory(graph, level_chunks=k)
                        sim.external_mapper = mapper
                        sim.enable_external_mapper()
                        sim.run()
                        add_metric_row(metis_metrics, k, sim)
                
                if "RL" in experiment_names:
                    env.rollout(policy=model.actor, max_steps=10000, auto_reset=False, tensordict=obs)
                    add_metric_row(metrics, "RL", env.simulator)
                    assert env.EFT_baseline == eft_time, "Mismatch in EFT baseline time"
                    

            # --- Average over samples
            average_last(metrics, experiment_names, MetricKeys, NUM_SAMPLES)
            if "Oracle" in experiment_names:
                average_last(metis_metrics, f, MetricKeys, NUM_SAMPLES)

            if "Oracle" in experiment_names:
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

            print(f"{sweep_entry:_},", end="")
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
        
        saved_lines = (
            f"# {cfg.graph.config.workload_args.traj_type} Trajectory\n"
            f"# Averaged over {NUM_SAMPLES} runs\n"
            f"all_keys={experiment_names}\n"
            f"speedup_keys={speedup_keys}\n"
            f"mem_keys={mem_keys}\n"
            f"sweep_list={sweep_list}\n"
            f"metrics={metrics}\n"
            f"colors={colors}\n"
        )
        
        print(saved_lines)

        # -------- Save Figures & Logs
        file_name = f"GraphSweep_{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}"
        try:
            file_name += f"_{cfg.graph.config.workload_args.scale}"
        except AttributeError:
            pass
        file_name += f"_{cfg.graph.config.workload_args.upper_bound}w{cfg.graph.config.workload_args.lower_bound}"
        file_name += f"_{cfg.graph.config.workload_args.traj_type}"

        # Directory layout
        base = Path("outputs") / file_name

        print(f"Saving results to {base}")
        base.mkdir(parents=True, exist_ok=True)
        # add time stamp
        ts = datetime.now().strftime("%m%d%H%M")  # mmddhhmm
        log_file = f"result_{file_name}_{ts}.txt"
        fig_file = f"figure_{file_name}_{ts}.png"
        # check if the txt file already exists
        i = 1
        while (base / log_file).exists():
            log_file = f"result_{file_name}_{i}.txt"
            i += 1
        i = 1
        while (base / fig_file).exists():
            fig_file = f"figure_{file_name}_{i}.png"
            i += 1
        # Save results text
        with open(base / log_file, "w") as ftxt:
            ftxt.write(OmegaConf.to_yaml(cfg))
            ftxt.write(
                "Sweep,TotalMem/ProblemSize," + ",".join([str(i) for i in metrics.keys()]) + "\n"
            )
            for idx in range(len(metrics["EFT"]["time"])):
                ftxt.write(f"Problem size: {sweep_list[idx]}\n")
                for k in metrics.keys():
                    if k != "Oracle":
                        line = f"{k},"
                    else:
                        line = f"{k}({dynamic_metis_k_best[idx]}),"
                    for times in metrics[k]["time_history"][idx]:
                        line += f"{times},"
                    ftxt.write(line + "\n")
                ftxt.write("\n\n")
            ftxt.write("\n")
            ftxt.write("\n\n BELOW LINES FOR REPLOTTING \n\n")
            ftxt.write(saved_lines)
        
        offset = 0
        xslice = slice(offset, None)  # Adjust this slice as needed
        fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True)
        sweep_list = [m / 1e9 for m in sweep_list]
        speedup = {}
        for k in speedup_keys:
            speedup[k] = []
            for i in range(len(sweep_list)):
                speedup[k].append(metrics["EFT"]["time"][i]/metrics[k]["time"][i])
        for k in experiment_names:
            axes[0].plot(
                sweep_list[xslice],
                metrics[k]["time"][xslice],
                label=k,
                linestyle="-",
                color=colors[k],
                linewidth=4,
            )
        axes[0].set_title("Execution Time (s)", fontsize=20)
        # axes[0].set_yscale("log")
        axes[0].grid(axis="y", color="gray", linestyle="--", linewidth=0.5)
        axes[0].legend(loc="upper right", fontsize=16)
        axes[0].set_xlabel("(a)", fontsize=20)
        axes[0].tick_params(axis="both", which="major", labelsize=20)
        # 2) Relative Speedup vs EFT
        for k in speedup_keys:
            axes[1].plot(
                sweep_list[xslice], speedup[k][xslice], label=k, color=colors[k], linewidth=4
            )
            # axes[1].plot(
            #     sweep_list[xslice], metrics[k]['vsEFT'][xslice], label=k, color=colors[k], linewidth=4
            # )
        axes[1].set_title("Relative Speedup vs EFT", fontsize=20)
        axes[1].legend(loc="upper right", fontsize=16)
        axes[1].grid()
        axes[1].set_xlabel("(b)", fontsize=20)
        axes[1].tick_params(axis="both", which="major", labelsize=20)
        # 3) Max Memory Usage
        for k in mem_keys:
            # Divide every entry by 1e9
            axes[2].plot(sweep_list[xslice], np.array(metrics[k]["mem_usage"][xslice]) / 1e9, label=k, color=colors[k], linewidth=4)
            # axes[2].plot(sweep_list[xslice], metrics[k]["mem_usage"][xslice], label=k, color=colors[k], linewidth=4)
        axes[2].legend(loc="lower left", fontsize=16)
        axes[2].set_title("Peak Memory Occupation (GB)", fontsize=20)
        axes[2].grid()
        axes[2].set_xlabel("(c)", fontsize=20)
        axes[2].tick_params(axis="both", which="major", labelsize=20)

        # # Shared x-axis label and layout
        fig.supxlabel(
            "Per Iteration Problem Size (GB)", fontsize=20
        )
        fig.tight_layout()  # leave room at the bottom for the xlabel
        
        fig.savefig(base / fig_file)

@hydra.main(config_path="conf", config_name="dynamic_batch", version_base=None)
def main(cfg: DictConfig):
    run_host_experiments_and_plot(cfg)


if __name__ == "__main__":
    main()