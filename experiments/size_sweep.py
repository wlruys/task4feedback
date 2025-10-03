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

from task4feedback.interface.wrappers import *  # noqa: F401,F403
from task4feedback.graphs.jacobi import JacobiGraph, LevelPartitionMapper, JacobiRoundRobinMapper, JacobiQuadrantMapper, BlockCyclicMapper, GraphMETISMapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.partition import *
from task4feedback.graphs.base import weighted_cell_partition
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.ml.models import FeatureDimConfig
from helper.model import create_td_actor_critic_models, load_policy_from_checkpoint

from helper.graph import make_graph_builder, GraphBuilder
from helper.env import make_env, create_system
from helper.parmetis import run_parmetis
import math

font_scale = 1.75
mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * font_scale

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ParMETIS = ParMETIS_wrapper()

MetricKeys = ("time", "mem_usage", "total_mem_movement", "eviction_movement", "time_history", "vsInf")

ITR_UB = []
INF_SYSTEM = None


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
    for i in range(start, steps):
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
        return JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, setting=1, offset=1)

    def cyclic_mapper(_: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, setting=0, offset=1)

    def quadrant_mapper(graph: DynamicJacobiGraph) -> LevelPartitionMapper:
        return JacobiQuadrantMapper(n_devices=cfg.system.n_devices - 1, graph=graph, offset=1)

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
        return BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=block_size, offset=1)

    def global_metis_cut(graph: DynamicJacobiGraph) -> GraphMETISMapper:
        return GraphMETISMapper(
            graph=graph,
            n_devices=cfg.system.n_devices - 1,
            offset=1,
            bandwidth=d2d_bandwidth,
        )

    return {
        "EFT": None,  # baseline (no external mapper)
        "Naive": naive_mapper,  # dynamic mode with chunks
        "ColWise": row_mapper,  # round-robin
        "Cyclic": cyclic_mapper,  # cyclic
        "Quad": quadrant_mapper,  # quadrant
        "GlbAvg": global_min_cut_mapper,
        "BlockCyclic": block_cyclic_mapper,  # BlockCyclicMapper
        "Oracle": None,  # handled separately (dynamic k sweep using dynamic_metis_mapper)
        "ParMETIS": None,  # handled by distributed loop
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


def run_inf_mem_sim(sim: "SimulatorDriver") -> "SimulatorDriver":
    sim.input.system = INF_SYSTEM
    task_runtime = sim.get_state().get_task_runtime()
    sim.external_mapper = ExternalMapper()
    inf_sim = sim.fresh_copy()
    inf_sim.initialize()
    inf_sim.initialize_data()
    inf_sim.enable_external_mapper()
    candidates = torch.zeros((sim.observer.graph_spec.max_candidates), dtype=torch.int64)

    state = inf_sim.run_until_external_mapping()
    while state != fastsim.ExecutionState.COMPLETE:
        inf_sim.get_mappable_candidates(candidates)
        actions = []
        for i, id in enumerate(candidates):
            task_id = id.item()
            mapping_priority = inf_sim.get_mapping_priority(task_id)
            actions.append(fastsim.Action(i, task_runtime.get_compute_task_mapped_device(task_id), mapping_priority, mapping_priority))
        inf_sim.simulator.map_tasks(actions)
        state = inf_sim.run_until_external_mapping()
    return inf_sim


skip_inf = []


def add_metric_row(metrics: Dict[str, Dict[str, List[float]]], name: str, sim: "SimulatorDriver", idx: int = -1) -> None:
    if "time" in metrics[name]:
        metrics[name]["time"][idx] += sim.time
    if "time_history" in metrics[name]:
        metrics[name]["time_history"][idx].append(sim.time)
    if "mem_usage" in metrics[name]:
        metrics[name]["mem_usage"][idx] += sim.max_mem_usage
    if "total_mem_movement" in metrics[name]:
        metrics[name]["total_mem_movement"][idx] += sum(list(sim.total_data_movement())[1:]) / 4
    if "eviction_movement" in metrics[name]:
        metrics[name]["eviction_movement"][idx] += sum(list(sim.total_eviction_movement())[1:]) / 4
    if "vsInf" in metrics[name]:
        if name not in skip_inf:
            inf_sim = run_inf_mem_sim(sim)
            slowdown = sim.time / inf_sim.time
            metrics[name]["vsInf"][idx] += slowdown
            if slowdown > 2:
                skip_inf.append(name)
        else:
            metrics[name]["vsInf"][idx] += 99999.0


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
            cfg.graph.config.level_memory = sweep_entry[0]
            cfg.graph.config.boundary_width = sweep_entry[2]
            graph_builder = make_graph_builder(cfg, verbose=False)
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)

        # Find best ITR using recommended ub = 1.05
        best_cfg = (None, None, float("inf"))  # (itr, ub, time)

        for itr in [0.000111, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]:
            if rank == 0:
                temp = env.simulator.copy()
            run_parmetis(sim=temp if rank == 0 else None, cfg=cfg, unbalance=1.019, itr=itr)
            if rank == 0 and temp.time < best_cfg[2]:
                best_cfg = (itr, 1.019, temp.time)
                print(f"New best ITR {itr} with time {temp.time}", flush=True)
        best_cfg = comm.bcast(best_cfg, root=0)
        ub_lo, ub_hi = 1.00, 1.10
        min_step = 1e-3  # stop when step size shrinks below this
        max_runs = 50  # hard cap on total ParMETIS runs for this phase

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
        ITR_UB.append(best_cfg)

        for _ in range(cfg.sweep.n_samples):
            if rank == 0:
                env._reset()
                eft_sim = env.simulator.copy()
                eft_sim.disable_external_mapper()
                eft_sim.run()
                add_metric_row(metrics, "EFT", eft_sim, sweep_idx)
            run_parmetis(sim=env.simulator if rank == 0 else None, cfg=cfg, unbalance=best_cfg[1], itr=best_cfg[0])

            if rank == 0:
                add_metric_row(metrics, "ParMETIS", env.simulator, sweep_idx)
                print(f"ParMETIS run complete: {env.simulator.time} s (EFT: {eft_sim.time} s, speedup: {eft_sim.time/env.simulator.time:.2f}x)", flush=True)


def run_host_experiments_and_plot(cfg: DictConfig):
    d2d_bandwidth = cfg.system.d2d_bw
    before_inf_sim = cfg.system.mem
    cfg.system.mem = 999999e9
    global INF_SYSTEM
    INF_SYSTEM = create_system(cfg)
    cfg.system.mem = before_inf_sim

    if not cfg.graph.env.change_priority and not cfg.graph.env.change_location and not cfg.graph.env.change_workload and not cfg.graph.env.change_duration:
        cfg.sweep.n_samples = 1
    if rank == 0:
        sweep_list = []

        cfg.graph.config.level_memory = cfg.sweep.start_mem
        graph_builder = make_graph_builder(cfg, verbose=False)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        data_stat = env.simulator_factory[0].input.graph.data.data_stat

        cnt = 0
        while True:
            graph_builder = make_graph_builder(cfg, verbose=False)
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
            graph = env.simulator_factory[0].input.graph
            if isinstance(graph, DynamicJacobiGraph):
                if graph.data.data_stat["average_step_data"] < 100e9:
                    print("Average step data too small (<100GB), skipping...")
                    cfg.graph.config.level_memory += cfg.sweep.level_size_step
                    continue
                elif graph.data.data_stat["average_step_data"] > 120e9:
                    print("Average step data too large (>120GB), stopping...")
                    break
            max_mem = 0
            for i in range(3):
                env.reset()
                env.simulator.disable_external_mapper()
                env.simulator.run()
                max_mem += env.simulator.max_mem_usage
            max_mem /= 3
            print(f"GPU MAX:{max_mem/1e9:.2f}GB")
            data_stat = env.simulator_factory[0].input.graph.data.data_stat
            print(f"Step: {data_stat['average_step_data']/1e9:.2f}GB")
            print(f"Task MAX: {env.simulator_factory[0].input.graph.max_requirement/1e9:.2f}GB")
            print(f"Boundary Width: {cfg.graph.config.boundary_width}")

            sweep_list.append((cfg.graph.config.level_memory, data_stat["average_step_data"], cfg.graph.config.boundary_width, env.simulator.max_mem_usage))

            cfg.graph.config.level_memory += cfg.sweep.level_size_step
            if max_mem > cfg.system.mem * 0.99:
                if cnt == 0 and len(sweep_list) > 5:
                    sweep_list = sweep_list[len(sweep_list) - 5 :]
                cnt += 1
            if env.simulator_factory[0].input.graph.max_requirement / cfg.system.mem > cfg.sweep.task_th or cnt >= 5 or sweep_list[-1][0] > cfg.sweep.end_mem:
                break

    sweep_list = comm.bcast(sweep_list if rank == 0 else None, root=0)

    experiment_names = ["EFT", "GlbAvg", "Oracle", "BlockCyclic", "ParMETIS", "Quad"]
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
            cfg.graph.config.level_memory = sweep_entry[0]
            cfg.graph.config.boundary_width = sweep_entry[2]
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

        saved_lines = (
            f"# {cfg.graph.config.workload_args.traj_type} Trajectory\n"
            f"# Averaged over {cfg.sweep.n_samples} runs\n"
            f"experiment_names={experiment_names}\n"
            f"speedup_keys={speedup_keys}\n"
            f"mem_keys={mem_keys}\n"
            f"sweep_list={[x[1] for x in sweep_list]}\n"
            f"metrics={metrics}\n"
            f"colors={colors}\n"
        )

        # -------- Save Figures & Logs
        file_name = f"SizeSweep_{cfg.graph.config.n}x{cfg.graph.config.n}x{cfg.graph.config.steps}"
        file_name += f"_{cfg.graph.config.workload_args.upper_bound}w{cfg.graph.config.workload_args.lower_bound}"
        file_name += f"_{cfg.graph.config.workload_args.traj_type}"
        try:
            file_name += f"_{cfg.graph.config.workload_args.traj_specifics.transition_frac}tf"
        except AttributeError:
            pass
        try:
            file_name += f"_{cfg.graph.config.workload_args.scale}"
        except AttributeError:
            pass

        # Directory layout
        base = Path("outputs") / file_name

        def closest_ratio_string(value: float) -> str:
            mapping = {10: "10", 1: "1", 0.1: "0.1"}
            closest = min(mapping.keys(), key=lambda x: abs(value - x))
            return mapping[closest]

        interior_ratio = 595.5555555 / (cfg.graph.config.arithmetic_intensity)
        boundary_ratio = interior_ratio * cfg.graph.config.boundary_width * 4

        interior_ratio_str = closest_ratio_string(interior_ratio)
        boundary_ratio_str = closest_ratio_string(boundary_ratio)

        print(f"Saving results to {base}")
        base.mkdir(parents=True, exist_ok=True)
        # add time stamp
        ts = datetime.now().strftime("%m%d%H%M")  # mmddhhmm
        log_file = f"result_{interior_ratio_str}I_{boundary_ratio_str}B.txt"
        fig_file = f"figure_{interior_ratio_str}I_{boundary_ratio_str}B.png"
        # Save results text
        with open(base / log_file, "w") as ftxt:
            ftxt.write("Sweep,TotalMem/ProblemSize," + ",".join([str(i) for i in metrics.keys()]) + "\n")
            for idx in range(len(metrics["EFT"]["time"])):
                ftxt.write(f"level_mem: {sweep_list[idx][0]} step_mem: {sweep_list[idx][1]} boundary_width: {sweep_list[idx][2]}\n")
                for k in metrics.keys():
                    if k == "Oracle":
                        line = f"{k}({dynamic_metis_k_best[idx]}),"
                    elif k == "ParMETIS":
                        line = f"{k}(ub={ITR_UB[idx][1]:.3f},ITR={ITR_UB[idx][0]:.3f}),"
                    else:
                        line = f"{k},"
                    for times in metrics[k]["time_history"][idx]:
                        line += f"{times},"
                    line += f"{metrics[k]['vsInf'][idx]:.3f}"
                    ftxt.write(line + "\n")
                ftxt.write("\n\n")
            ftxt.write("\n")
            ftxt.write("\n\n BELOW LINES FOR REPLOTTING \n\n")
            ftxt.write(saved_lines)
            ftxt.write("\n\n CONFIGURATION \n\n")
            ftxt.write(OmegaConf.to_yaml(cfg))

        offset = 0
        xslice = slice(offset, None)  # Adjust this slice as needed
        fig, axes = plt.subplots(1, 4, figsize=(26, 6), sharex=True)
        # Interior, Boundary vs Compute
        interior_ratio = [interior_ratio for _ in sweep_list]
        boundary_ratio = [boundary_ratio for _ in sweep_list]

        xaxis = [i[0] / 1e9 for i in sweep_list]

        axes[0].plot(xaxis, interior_ratio[xslice], label="Interior", color="tab:blue", linewidth=4)
        axes[0].plot(xaxis, boundary_ratio[xslice], label="Boundary", color="tab:orange", linewidth=4)
        axes[0].set_title("Communication Time / Compute Time", fontsize=20)
        axes[0].legend(loc="lower right", fontsize=16)
        axes[0].grid()
        # axes[0].set_xlabel("(d)", fontsize=20)
        axes[0].tick_params(axis="both", which="major", labelsize=20)
        axes[0].set_yscale("log", base=2)

        speedup = {}
        for k in speedup_keys:
            speedup[k] = []
            for i in range(len(sweep_list)):
                speedup[k].append(metrics["EFT"]["time"][i] / metrics[k]["time"][i])
        for k in experiment_names:
            axes[1].plot(
                xaxis,
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
        # 2) Relative Speedup vs EFT
        for k in speedup_keys:
            axes[2].plot(xaxis, speedup[k][xslice], label=k, color=colors[k], linewidth=4)
        axes[2].set_title("Relative Speedup vs EFT", fontsize=20)
        axes[2].legend(loc="upper left", fontsize=16)
        axes[2].grid()
        # axes[2].set_xlabel("(b)", fontsize=20)
        axes[2].tick_params(axis="both", which="major", labelsize=20)
        # 3) Max Memory Usage
        for k in mem_keys:
            # Divide every entry by 1e9
            axes[3].plot(xaxis, np.array(metrics[k]["mem_usage"][xslice]) / 1e9, label=k, color=colors[k], linewidth=4)
            # axes[3].plot(xaxis, metrics[k]["mem_usage"][xslice], label=k, color=colors[k], linewidth=4)
        axes[3].legend(loc="upper left", fontsize=16)
        axes[3].set_title("Peak Memory Occupation (GB)", fontsize=20)
        axes[3].grid()
        # axes[3].set_xlabel("(c)", fontsize=20)
        axes[3].tick_params(axis="both", which="major", labelsize=20)

        # # Shared x-axis label and layout
        fig.supxlabel("Average Step Memory Requirement (GB)", fontsize=20)
        fig.tight_layout()  # leave room at the bottom for the xlabel

        fig.savefig(base / fig_file)


@hydra.main(config_path="conf", config_name="size_sweep", version_base=None)
def main(cfg: DictConfig):
    run_host_experiments_and_plot(cfg)


if __name__ == "__main__":
    main()
