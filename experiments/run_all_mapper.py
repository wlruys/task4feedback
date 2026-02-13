from email import policy
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import create_td_actor_critic_models
from task4feedback.experiment_helper.algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.graphs.jacobi import (
    JacobiGraph,
    LevelPartitionMapper,
    JacobiRoundRobinMapper,
    JacobiQuadrantMapper,
    BlockCyclicMapper,
    GraphMETISMapper,
)

# from task4feedback.graphs.mesh.plot_fast import *
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from hydra.experimental.callbacks import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig, open_dict
from pathlib import Path
import git
import os
from hydra.core.hydra_config import HydraConfig
from task4feedback.experiment_helper.run_name import make_run_name, cfg_hash
import torch
import numpy
import random
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.ml.util import EvaluationConfig
from task4feedback.experiment_helper.parmetis import run_parmetis, find_best_cfg, find_best_cfg_optuna
from mpi4py import MPI
import socket
import time
import fcntl


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
EVAL_GRAPH_STEPS = 256
PHASE_LENGTH = 128
SYSTEM_MEMORY = 96e9


def write_results_atomic(path, lines):
    """
    Append lines to a file using an exclusive file lock.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def csv_entry_exists(path, key_tuple):
    """
    Check whether a CSV file already contains an entry starting with key_tuple.
    key_tuple corresponds to:
    (traj_type, level_memory, r_interior, r_boundary)
    """
    if not os.path.exists(path):
        return False

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            if tuple(parts[:4]) == tuple(map(str, key_tuple)):
                return True
    return False


def configure_training(cfg: DictConfig):
    if cfg.graph.env.change_duration:
        output_name = f"noise_level_sweep_results_{EVAL_GRAPH_STEPS}.csv"
    else:
        output_name = f"level_sweep_results_{EVAL_GRAPH_STEPS}.csv"
    out_file = os.path.join(f"./results/{cfg.system.n_devices-1}gpus/", output_name)
    graph_name = cfg.graph.config.workload_args.traj_type
    key = (
        graph_name,
        cfg.graph.config.level_memory,
        cfg.graph.config.r_interior,
        cfg.graph.config.r_boundary,
    )

    if rank == 0:
        exists = csv_entry_exists(out_file, key)
        if exists:
            print(
                f"[SKIP] Entry already exists in CSV for " f"{key}, exiting.",
                flush=True,
            )
    else:
        exists = None

    # Broadcast decision to all ranks
    exists = comm.bcast(exists, root=0)

    if exists:
        return  # clean early exit for all ranks
    # start_logger()
    cfg.system.mem = SYSTEM_MEMORY
    cfg.graph.config.steps = EVAL_GRAPH_STEPS
    cfg.graph.config.workload_args.traj_specifics.phase_length = PHASE_LENGTH
    num_runs = 20 if cfg.graph.env.change_duration else 12 if cfg.graph.env.change_workload else 1
    best_cfg = None
    best_inf_cfg = None
    ParMETIS = ParMETIS_wrapper()

    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True)

        graph_builder = make_graph_builder(cfg)
        # backup = cfg.system.h2d_bw
        # cfg.system.h2d_bw = int(99999e9)
        cfg.system.mem *= 1.25
        infenv = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False, eval=True)
        # cfg.system.h2d_bw = backup
        infenv.simulator_factory
        graph = env.get_graph()
        results = {"parmetis": [], "eft": [], "b4": [], "b2": [], "rc": []}
        inf_results = {"parmetis": [], "eft": [], "b4": [], "b2": [], "rc": []}
    else:
        env = None
        copy_sim = None
        copy_inf_sim = None
        infenv = None

    for i in range(num_runs):
        if rank == 0:
            env.reset()
            infenv.reset()
            copy_sim = env.simulator.copy()
            copy_inf_sim = infenv.simulator.copy()

        comm.barrier()
        if best_cfg is None:
            best_cfg = find_best_cfg(cfg, ParMETIS, env=env, skip_search=True)
            best_cfg_optuna = find_best_cfg_optuna(cfg, ParMETIS, env=env, skip_search=best_cfg is not None, mode="normal_optuna")
            # best_cfg_optuna = find_best_cfg_optuna(cfg, ParMETIS, env=env, skip_search=True, mode="normal_optuna")
            if best_cfg is None:
                best_cfg = best_cfg_optuna
            elif best_cfg_optuna is not None and best_cfg[2] > best_cfg_optuna[2]:
                best_cfg = best_cfg_optuna
            best_inf_cfg = best_cfg
        comm.barrier()
        # exit()

        if best_cfg is not None:
            run_parmetis(sim=copy_sim, cfg=cfg, unbalance=best_cfg[1], itr=best_cfg[0], ParMETIS=ParMETIS, n_compute_devices=cfg.system.n_devices - 1)
            run_parmetis(sim=copy_inf_sim, cfg=cfg, unbalance=best_inf_cfg[1], itr=best_inf_cfg[0], ParMETIS=ParMETIS, n_compute_devices=cfg.system.n_devices - 1)
        comm.barrier()

        if rank == 0:
            if best_cfg is not None:
                print(f"ParMETIS: {(copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement()))}")
                results["parmetis"].append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))
                inf_results["parmetis"].append((copy_inf_sim.time, 0, sum(copy_inf_sim.total_data_movement())))

            # copy_sim = env.simulator.copy()
            # copy_sim.disable_external_mapper()
            # copy_sim.run()
            # copy_inf_sim = infenv.simulator.copy()
            # copy_inf_sim.disable_external_mapper()
            # copy_inf_sim.run()
            # print(f"EFT: {(copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement()))}")
            # results["eft"].append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))
            # inf_results["eft"].append((copy_inf_sim.time, 0, sum(copy_inf_sim.total_data_movement())))

            if cfg.system.n_devices - 1 == 4:
                copy_sim = env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=4, offset=1)
                copy_sim.run()
                copy_inf_sim = infenv.simulator.copy()
                copy_inf_sim.enable_external_mapper()
                copy_inf_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=4, offset=1)
                copy_inf_sim.run()
                print(f"BlockCyclic(4x4): {(copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement()))}")
                results["b4"].append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))
                inf_results["b4"].append((copy_inf_sim.time, 0, sum(copy_inf_sim.total_data_movement())))
            elif cfg.system.n_devices - 1 == 8:
                copy_sim = env.simulator.copy()
                copy_sim.enable_external_mapper()
                copy_sim.external_mapper = JacobiQuadrantMapper(graph=graph, n_devices=cfg.system.n_devices - 1, offset=1)
                copy_sim.run()
                copy_inf_sim = infenv.simulator.copy()
                copy_inf_sim.enable_external_mapper()
                copy_inf_sim.external_mapper = JacobiQuadrantMapper(graph=graph, n_devices=cfg.system.n_devices - 1, offset=1)
                copy_inf_sim.run()
                print(f"Quadrant: {(copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement()))}")
                results["b4"].append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))
                inf_results["b4"].append((copy_inf_sim.time, 0, sum(copy_inf_sim.total_data_movement())))

            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=2, offset=1)
            copy_sim.run()
            copy_inf_sim = infenv.simulator.copy()
            copy_inf_sim.enable_external_mapper()
            copy_inf_sim.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=2, offset=1)
            copy_inf_sim.run()
            print(f"BlockCyclic(2x2): {(copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement()))}")
            results["b2"].append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))
            inf_results["b2"].append((copy_inf_sim.time, 0, sum(copy_inf_sim.total_data_movement())))

            copy_sim = env.simulator.copy()
            copy_sim.enable_external_mapper()
            copy_sim.external_mapper = JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, offset=1, setting=1)
            copy_sim.run()
            copy_inf_sim = infenv.simulator.copy()
            copy_inf_sim.enable_external_mapper()
            copy_inf_sim.external_mapper = JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, offset=1, setting=1)
            copy_inf_sim.run()
            print(f"RowCyclic: {(copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement()))}")
            results["rc"].append((copy_sim.time, sum(list(copy_sim.total_eviction_movement())[1:]), sum(copy_sim.total_data_movement())))
            inf_results["rc"].append((copy_inf_sim.time, 0, sum(copy_inf_sim.total_data_movement())))

    if rank == 0:
        # ---- aggregate ----
        averaged = {}
        inf_averaged = {}
        for mapper, times in results.items():
            if len(times) == 0:
                continue
            averaged[mapper] = (sum(t[0] for t in times) / len(times), sum(t[1] for t in times) / len(times), sum(t[2] for t in times) / len(times))
        for mapper, times in inf_results.items():
            if len(times) == 0:
                continue
            inf_averaged[mapper] = (sum(t[0] for t in times) / len(times), sum(t[1] for t in times) / len(times), sum(t[2] for t in times) / len(times))
        # ---- prepare CSV lines ----
        lines = []
        for mapper, avg_time in averaged.items():
            # line = f"{graph_name},{cfg.graph.config.workload_args.traj_specifics.phase_length},{cfg.graph.config.r_interior},{cfg.graph.config.r_boundary},{mapper},{avg_time:.0f}"
            line = f"{graph_name},{cfg.graph.config.level_memory},{cfg.graph.config.r_interior},{cfg.graph.config.r_boundary},{mapper},{avg_time[0]:.0f},{avg_time[1]:.0f},{avg_time[2]:.0f}"
            lines.append(line)

        for mapper, avg_time in inf_averaged.items():
            # line = f"{graph_name},{cfg.graph.config.workload_args.traj_specifics.phase_length},{cfg.graph.config.r_interior},{cfg.graph.config.r_boundary},{mapper}_inf,{avg_time:.0f}"
            line = f"{graph_name},{cfg.graph.config.level_memory},{cfg.graph.config.r_interior},{cfg.graph.config.r_boundary},{mapper}_inf,{avg_time[0]:.0f},{avg_time[1]:.0f},{avg_time[2]:.0f}"
            lines.append(line)

        # ---- write safely ----
        write_results_atomic(out_file, lines)


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
