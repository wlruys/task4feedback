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
    JacobiQuadrantMapper, 
    BlockCyclicMapper, 
    GraphMETISMapper
)
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.partition import * 
from task4feedback.graphs.base import weighted_cell_partition
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.ml.models import FeatureDimConfig
from helper.model import create_td_actor_critic_models, load_policy_from_checkpoint

from helper.graph import make_graph_builder, GraphBuilder
from helper.env import make_env
from helper.parmetis import run_parmetis
import math
font_scale = 1.75
mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * font_scale

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ParMETIS = ParMETIS_wrapper()

MetricKeys = ("time", "mem_usage", "total_mem_movement", "eviction_movement", "time_history")

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
            setting=1,
            offset=1
        )
    
    def cyclic_mapper(_: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiRoundRobinMapper(
            n_devices=cfg.system.n_devices - 1,
            setting=0,
            offset=1
        )

    def quadrant_mapper(graph: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiQuadrantMapper(
            n_devices=cfg.system.n_devices - 1,
            graph=graph,
            offset=1
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
    

    def block_cyclic_mapper(graph: DynamicJacobiGraph, block_size=1) -> BlockCyclicMapper:
        return BlockCyclicMapper(
            geometry=graph.data.geometry,
            n_devices=cfg.system.n_devices - 1,
            block_size=block_size,
            offset=1
        )
    
    def global_metis_cut(graph: DynamicJacobiGraph) -> GraphMETISMapper:
        return GraphMETISMapper(
            graph=graph,
            n_devices=cfg.system.n_devices - 1,
            offset=1,
            bandwidth= d2d_bandwidth,
        )

    return {
        "EFT": None,                    # baseline (no external mapper)
        "Naive": naive_mapper,          # dynamic mode with chunks
        "ColWise": row_mapper,            # round-robin
        "Cyclic": cyclic_mapper,             # cyclic
        "Quad": quadrant_mapper,        # quadrant
        "GlbAvg": global_min_cut_mapper,
        "BlockCyclic": block_cyclic_mapper,  # BlockCyclicMapper
        "Oracle": None,                 # handled separately (dynamic k sweep using dynamic_metis_mapper)
        "ParMETIS": None,               # handled by distributed loop
        "GraphMETISMapper": global_metis_cut,  # GraphMETISMapper
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


def average_metric(metrics: Dict[str, Dict[str, List[float]]], names: Iterable[str], keys: Iterable[str], num_samples: int, idx: int = -1) -> None:
    for n in names:
        for k in keys:
            if metrics[n][k] and k != "time_history":
                metrics[n][k][idx] /= num_samples

# =====================================================================
# ParMETIS distributed mapping
# =====================================================================

def run_parmetis_distributed(
    cfg: DictConfig,
    sweep_list: List[int],
    num_samples: int,
    metrics: Dict[str, Dict[str, List[float]]],
) -> None:
    """
    Executes the ParMETIS portion with MPI, accumulating results in `metrics["ParMETIS"]`.
    """
    
    for sweep_idx, sweep_entry in enumerate(sweep_list):
        if rank == 0:
            cfg.graph.config.boundary_width = sweep_entry
            graph_builder = make_graph_builder(cfg, verbose=False)
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        
        # Find best ITR using recommended ub = 1.05
        best_cfg = (None, None, float('inf'))  # (itr, ub, time)

        for itr in [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
            if rank == 0:
                temp = env.simulator.copy()
            run_parmetis(sim=temp if rank == 0 else None, cfg=cfg, unbalance=1.05, itr=itr)
            if rank == 0 and temp.time < best_cfg[2]:
                best_cfg = (itr, 1.05, temp.time)
                print(f"New best ITR {itr} with time {temp.time}", flush=True)
        best_cfg = comm.bcast(best_cfg, root=0)
        ub_lo, ub_hi = 1.01, 2.0
        min_step   = 1e-3        # stop when step size shrinks below this
        max_runs   = 50          # hard cap on total ParMETIS runs for this phase

        # Initialize from existing best ub if present; otherwise use midpoint.
        ub_cur = best_cfg[1]
        ub_cur = max(min(ub_cur, ub_hi), ub_lo)

        # Initial step = quarter of the range (conservative)
        step = 0.25 * (ub_hi - ub_lo)
        direction = 1.0  # +1 to go up, -1 to go down

        runs = 0
        while step >= min_step and runs < max_runs:
            # Propose next ub and clamp to bounds
            ub_next = ub_cur + direction * step
            if ub_next < ub_lo or ub_next > ub_hi:
                # Bounce off bound: reverse direction and shrink step
                direction *= -1.0
                step *= 0.5
                continue
            if rank == 0:
                temp = env.simulator.copy()
            run_parmetis(sim=(temp if rank == 0 else None), cfg=cfg, unbalance=ub_next, itr=best_cfg[0])
            runs += 1
            if rank == 0:
                print(f"Tried ub {ub_next:.6f} with time {temp.time}", flush=True)
                if temp.time < best_cfg[2]:
                    # Improvement: accept move, keep direction, keep step
                    ub_cur = ub_next
                    best_cfg = (best_cfg[0], ub_cur, temp.time)
                    print(f"New best ub {ub_cur:.6f} with time {temp.time}", flush=True)
                else:
                    # No improvement: reverse direction and halve step
                    direction *= -1.0
                    step *= 0.5
            ub_cur = comm.bcast(ub_cur, root=0)
            direction = comm.bcast(direction, root=0)
            step = comm.bcast(step, root=0)
        best_cfg = comm.bcast(best_cfg, root=0)
                
        for _ in range(cfg.sweep.n_samples):
            if rank == 0:
                env._reset()
                eft_sim = env.simulator.copy()
                eft_sim.disable_external_mapper()
                eft_sim.run()
                add_metric_row(metrics, "EFT", eft_sim, sweep_idx)

            run_parmetis(sim=env.simulator if rank == 0 else None, cfg=cfg, unbalance=cfg.parmetis.unbalance, itr=cfg.parmetis.itr)
            
            if rank == 0:
                add_metric_row(metrics, "ParMETIS", env.simulator, sweep_idx)
                print(f"ParMETIS run complete: {env.simulator.time} s (EFT: {eft_sim.time} s, speedup: {eft_sim.time/env.simulator.time:.2f}x)", flush=True)
            

def run_host_experiments_and_plot(cfg: DictConfig):
    d2d_bandwidth = cfg.system.d2d_bw
    cfg.system.mem = 2**62
    if rank == 0:
        graph_builder = make_graph_builder(cfg, verbose=False)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        data_stat = env.simulator_factory[0].input.graph.data.data_stat
        print(data_stat)
        cfg.graph.config.arithmetic_intensity = cfg.sweep.interior_ratio * (data_stat["interior_average_comm"] / data_stat["compute_average"]) * cfg.graph.config.arithmetic_intensity
        boundary_scale = data_stat["interior_average_comm"] / data_stat["boundary_average_comm"]
        sweep_list = [int(n * boundary_scale * cfg.graph.config.boundary_width) for n in cfg.sweep.list]
        print(cfg.graph.config.arithmetic_intensity)

    sweep_list = comm.bcast(sweep_list if rank == 0 else None, root=0)
    
    experiment_names = cfg.sweep.exps
    mem_keys = experiment_names.copy()
    speedup_keys = experiment_names.copy()
    speedup_keys.remove("EFT")
    seed_everything(cfg.seed)
    
    if rank == 0:
        print(sweep_list)
        metrics = init_metrics(experiment_names, MetricKeys)
        for sweep_index, sweep_entry in enumerate(sweep_list):
            append_zero_row(metrics, experiment_names, MetricKeys)

    
    
    if size < 4 and "ParMETIS" in experiment_names:
        print("ParMETIS is in experiment lists and it requires at least 4 ranks. Stopping...")
        exit()
    elif "ParMETIS" in experiment_names:
        run_parmetis_distributed(
            cfg=cfg,
            sweep_list=sweep_list,
            num_samples=cfg.sweep.n_samples,
            metrics=metrics if rank == 0 else None,
        )
        
    if rank == 0:
        print(metrics)
        if "RL" in experiment_names:
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
        include_one = "GlbAvg" in experiment_names
        f = factorize(cfg.graph.config.steps, include_one=include_one)
        print(f"Factors of {cfg.graph.config.steps}: {f}")
        
        dynamic_metis_k_best: List[int] = []

        print(
            f"Memory,{str.join(',', [str(m) for m in experiment_names])}",
            flush=True,
        )

        # ---- Sweep host-side (everything but ParMETIS)
        for sweep_index, sweep_entry in enumerate(sweep_list):
            # system config for this memory size
            cfg.graph.config.boundary_width = sweep_entry
            graph_builder = make_graph_builder(cfg, verbose=False)
            if "RL" in experiment_names:
                env, norm = make_env(graph_builder=graph_builder, cfg=cfg, normalization=None, eval=True)
                env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm, eval=True)
            else:
                env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True)
            
            # per-k Oracle metrics
            if "Oracle" in experiment_names:
                metis_metrics = init_metrics(f, MetricKeys)
                append_zero_row(metis_metrics, f, MetricKeys)

            for sample_idx in range(cfg.sweep.n_samples):
                obs = env._reset()

                # baseline graph and simulator
                graph = env.simulator_factory[0].input.graph
                sim_base = env.simulator.copy()

                # ---- EFT
                sim = sim_base.copy()
                sim.disable_external_mapper()
                sim.run()
                if "ParMETIS" in experiment_names:
                    assert metrics["EFT"]["time_history"][sweep_index][sample_idx] == sim.time, "Mismatch in EFT time history"
                else:
                    add_metric_row(metrics, "EFT", sim, sweep_index)

                # ---- Naïve / ColWise / GlbAvg
                for name in ["Naive", "ColWise", "GlbAvg", "Quad", "Cyclic", "BlockCyclic", "GraphMETISMapper"]:
                    if name in experiment_names:
                        mapper_fn = experiment_mappers[name]
                        assert mapper_fn is not None
                        sim = sim_base.copy()
                        mapper = mapper_fn(graph)
                        sim.external_mapper = mapper
                        sim.enable_external_mapper()
                        sim.run()
                        add_metric_row(metrics, name, sim, sweep_index)

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
                    add_metric_row(metrics, "RL", env.simulator, sweep_index)
            
            if "Oracle" in experiment_names:
                # --- Pick best k for Oracle at this memory
                min_time = float("inf")
                best_k = 99999
                for k in f:
                    assert len(metis_metrics[k]["time"]) == 1
                    if k > 1 and metis_metrics[k]["time"][0] < min_time:
                        min_time = metis_metrics[k]["time"][0]
                        best_k = k

                dynamic_metis_k_best.append(best_k)
                # Copy best to "Oracle"
                for key in MetricKeys:
                    metrics["Oracle"][key][sweep_index] = metis_metrics[best_k][key][0]        

            # --- Average over samples
            average_metric(metrics, experiment_names, MetricKeys, cfg.sweep.n_samples, sweep_index)

            print(f"{sweep_entry},", end="")
            for name in experiment_names:
                if name == "Oracle":
                    print(
                        f"{int(metrics[name]['time'][sweep_index])}({dynamic_metis_k_best[-1]})",
                        end=",",
                    )
                else:
                    print(int(metrics[name]["time"][sweep_index]), end=",")
            print("")

    # ---- Post-processing, plots, and saving (rank 0 only)
    if rank == 0:
        print(metrics)
        
        sweep_list = cfg.sweep.list
        
        saved_lines = (
            f"# {cfg.graph.config.workload_args.traj_type} Trajectory\n"
            f"# Averaged over {cfg.sweep.n_samples} runs\n"
            f"experiment_names={experiment_names}\n"
            f"speedup_keys={speedup_keys}\n"
            f"mem_keys={mem_keys}\n"
            f"sweep_list={sweep_list}\n"
            f"metrics={metrics}\n"
            f"colors={colors}\n"
        )
        
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
                ftxt.write(f"Boundary / Interior: {sweep_list[idx]}\n")
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
        fig, axes = plt.subplots(1, 4, figsize=(26, 6), sharex=True)
        # Interior, Boundary vs Compute
        interior_ratio = [cfg.sweep.interior_ratio for _ in sweep_list]
        boundary_ratio = [i * cfg.sweep.interior_ratio for i in cfg.sweep.list]
        
        axes[0].plot(
            sweep_list[xslice], interior_ratio[xslice], label="Interior", color="tab:blue", linewidth=4
        )
        axes[0].plot(
            sweep_list[xslice], boundary_ratio[xslice], label="Boundary", color="tab:orange", linewidth=4
        )
        axes[0].set_title("Communication Time / Compute Time", fontsize=20)
        axes[0].legend(loc="lower right", fontsize=16)
        axes[0].grid()
        # axes[0].set_xlabel("(d)", fontsize=20)
        axes[0].tick_params(axis="both", which="major", labelsize=20)
        axes[0].set_xscale("log", base=2)
        
        speedup = {}
        for k in speedup_keys:
            speedup[k] = []
            for i in range(len(sweep_list)):
                speedup[k].append(metrics["EFT"]["time"][i]/metrics[k]["time"][i])
        for k in experiment_names:
            axes[1].plot(
                sweep_list[xslice],
                metrics[k]["time"][xslice],
                label=k,
                linestyle="-",
                color=colors[k],
                linewidth=4,
            )
        axes[1].set_title("Execution Time (s)", fontsize=20)
        # axes[1].set_yscale("log")
        axes[1].grid(axis="y", color="gray", linestyle="--", linewidth=0.5)
        axes[1].legend(loc="upper right", fontsize=16)
        # axes[1].set_xlabel("(a)", fontsize=20)
        axes[1].tick_params(axis="both", which="major", labelsize=20)
        axes[1].set_xscale("log", base=2)
        # 2) Relative Speedup vs EFT
        for k in speedup_keys:
            axes[2].plot(
                sweep_list[xslice], speedup[k][xslice], label=k, color=colors[k], linewidth=4
            )
        axes[2].set_title("Relative Speedup vs EFT", fontsize=20)
        axes[2].legend(loc="upper left", fontsize=16)
        axes[2].grid()
        # axes[2].set_xlabel("(b)", fontsize=20)
        axes[2].tick_params(axis="both", which="major", labelsize=20)
        axes[2].set_xscale("log", base=2)
        # 3) Max Memory Usage
        for k in mem_keys:
            # Divide every entry by 1e9
            axes[3].plot(sweep_list[xslice], np.array(metrics[k]["mem_usage"][xslice]) / 1e9, label=k, color=colors[k], linewidth=4)
            # axes[3].plot(sweep_list[xslice], metrics[k]["mem_usage"][xslice], label=k, color=colors[k], linewidth=4)
        axes[3].legend(loc="upper left", fontsize=16)
        axes[3].set_title("Peak Memory Occupation (GB)", fontsize=20)
        axes[3].grid()
        # axes[3].set_xlabel("(c)", fontsize=20)
        axes[3].tick_params(axis="both", which="major", labelsize=20)
        axes[3].set_xscale("log", base=2)

        # # Shared x-axis label and layout
        fig.supxlabel(
            "Boundary Memory / Interior Memory", fontsize=20
        )
        fig.tight_layout()  # leave room at the bottom for the xlabel
        
        fig.savefig(base / fig_file)

@hydra.main(config_path="conf", config_name="boundary_sweep", version_base=None)
def main(cfg: DictConfig):
    run_host_experiments_and_plot(cfg)


if __name__ == "__main__":
    main()