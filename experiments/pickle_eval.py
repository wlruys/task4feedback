from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models
from helper.algorithm import create_optimizer, create_lr_scheduler
from helper.run_name import make_folder_name

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
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
from helper.run_name import make_run_name, cfg_hash
import torch
import numpy
import random
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.ml.util import EvaluationConfig
from helper.parmetis import run_parmetis
import pickle
import re

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_policy(policy_str: str):
    """
    Parse cfg.sweep.policy string into its components.
    Supports Oracle(k), ParMETIS(ub,itr), BlockCyclic, EFT.
    """
    # Regex to capture function-like calls
    match = re.match(r"([A-Za-z]+)\(([^)]*)\)", policy_str)

    if match:
        name = match.group(1)
        args_str = match.group(2)
        # Split args by comma, convert to int or float
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            if arg.isdigit():
                args.append(int(arg))
            else:
                try:
                    args.append(float(arg))
                except ValueError:
                    args.append(arg)  # fallback as string
        return name, args
    else:
        # Just a string (BlockCyclic, EFT, etc.)
        return policy_str, []


def configure_training(cfg: DictConfig):
    # start_logger()
    extend = 2
    num_samples = cfg.eval.samples

    eval_state = {"cfg": OmegaConf.to_yaml(cfg), "init_locs": [], "workloads": [], "eft_times": [], "policy_times": [], "reset_counter": []}
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        env.set_reset_counter(9999)

    # def closest_ratio_string(value: float) -> str:
    #     mapping = {100: "100", 10: "10", 1: "1", 0.1: "0.1"}
    #     closest = min(mapping.keys(), key=lambda x: abs(value - x))
    #     return mapping[closest]

    # interior_ratio = 595.5555555 / (cfg.graph.config.arithmetic_intensity)
    # boundary_ratio = interior_ratio * cfg.graph.config.boundary_width * 4

    # interior_str = closest_ratio_string(interior_ratio)
    # boundary_str = closest_ratio_string(boundary_ratio)

    folder_name, graph_name, interior_str, boundary_str = make_folder_name(cfg)

    saved_policy, meta_data = parse_policy(cfg.sweep.policy)

    cfg.graph.config.steps *= extend
    if cfg.graph.config.workload_args.traj_type == "circle":
        cfg.graph.config.workload_args.traj_specifics.max_angle *= extend

    for i in range(num_samples):
        if rank == 0:
            eval_state["reset_counter"].append(env.resets)
            env.reset()
            eval_state["init_locs"].append(env.get_graph().get_cell_locations(as_dict=False))
            graph = env.get_graph()
            if isinstance(graph, DynamicJacobiGraph):
                eval_state["workloads"].append(dict(graph.get_workload().level_workload))
            else:
                eval_state["workloads"].append(None)
            eval_state["eft_times"].append(env._get_baseline("EFT"))
        if saved_policy == "EFT" and rank == 0:
            eval_state["policy_times"].append(env._get_baseline("EFT"))
        elif saved_policy == "Quad" and rank == 0:
            eval_state["policy_times"].append(env._get_baseline("Quad"))
        elif saved_policy == "RowCyclic" and rank == 0:
            eval_state["policy_times"].append(env._get_baseline("Cyclic"))
        elif saved_policy == "BlockCyclic" and rank == 0:
            env.simulator.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=4, block_size=int(meta_data[0][0]), offset=1)
            env.simulator.run()
            eval_state["policy_times"].append(env.simulator.time)
        elif saved_policy == "Oracle" and rank == 0:
            graph.mincut_per_levels(
                bandwidth=cfg.system.d2d_bw,
                mode="metis",
                offset=1,
                level_chunks=meta_data[0],
            )
            graph.align_partitions()
            env.simulator.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
            env.simulator.run()
            eval_state["policy_times"].append(env.simulator.time)
        elif saved_policy == "ParMETIS":
            run_parmetis(sim=env.simulator if rank == 0 else None, cfg=cfg, unbalance=meta_data[1], itr=meta_data[0])
            if rank == 0:
                eval_state["policy_times"].append(env.simulator.time)

        if rank == 0:
            print(f"{i}: EFT {eval_state['eft_times'][-1]:.4f}, {saved_policy} {eval_state['policy_times'][-1]:.4f} ({eval_state['eft_times'][-1]/eval_state['policy_times'][-1]:.2f}x)")
    # print(eval_state)
    # pickle.dump(eval_state, open("4x4x16_static_1:1:1.pkl", "wb"))
    if rank == 0:
        file_name = f"./pickled_evaluation/{folder_name}"
        pickle.dump(eval_state, open(f"{file_name}.pkl", "wb"))
        # print(eval_state)

        env.set_reset_counter(0)
        env._reset()

        # eval_state = pickle.load(open("dynamic_bump_eval.pkl", "rb"))
        for i in range(num_samples):
            saved_loc = eval_state["init_locs"][i]
            workload = eval_state["workloads"][i]
            env.set_reset_counter(eval_state["reset_counter"][i])
            env.reset()
            # env.reset_to_state(saved_loc, workload)
            print(f"Eval {i}:")
            sim_time = env._get_baseline("EFT")
            if eval_state["eft_times"][i] != sim_time:
                print(f"  Warning: EFT time changed! {eval_state['eft_times'][i]} -> {sim_time}")
                raise ValueError("EFT time mismatch")
            else:
                print("EFT time matches.")
            # print("EFT:", eval_state["eft_times"][i])


@hydra.main(config_path="conf", config_name="static_batch.yaml", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
