from email import policy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models
from helper.algorithm import create_optimizer, create_lr_scheduler

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
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def configure_training(cfg: DictConfig):
    # start_logger()

    option = "Oracle"
    if rank == 0:
        graph_builder = make_graph_builder(cfg)
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
        env._reset()
        graph = env.get_graph()
        if isinstance(graph, DynamicJacobiGraph):
            workload = graph.get_workload()
            # workload.animate_workload(show=False, title="outputs/workload_animation.mp4")

    if option == "EFT" and rank == 0:
        env.simulator.disable_external_mapper()
    elif option == "Oracle" and rank == 0:
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="metis",
            offset=1,
            level_chunks=16,
        )
        graph.align_partitions()
        env.simulator.enable_external_mapper()
        env.simulator.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
    elif option == "BlockCyclic":
        env.simulator.enable_external_mapper()
        env.simulator.external_mapper = BlockCyclicMapper(geometry=graph.data.geometry, n_devices=cfg.system.n_devices - 1, block_size=2, offset=1)
    elif option == "GraphMETISMapper":
        env.simulator.enable_external_mapper()
        env.simulator.external_mapper = GraphMETISMapper(graph=graph, n_devices=cfg.system.n_devices - 1, offset=1)
    elif option == "Quad":
        env.simulator.enable_external_mapper()
        env.simulator.external_mapper = JacobiQuadrantMapper(n_devices=cfg.system.n_devices - 1, graph=graph, offset=1)
    elif option == "Cyclic":
        env.simulator.enable_external_mapper()
        env.simulator.external_mapper = JacobiRoundRobinMapper(n_devices=cfg.system.n_devices - 1, offset=1, setting=0)
    elif option == "ParMETIS":
        run_parmetis(sim=env.simulator if rank == 0 else None, cfg=cfg)
    else:
        raise ValueError(f"Unknown option: {option}")

    # Added to check priority of each task
    sim: SimulatorDriver = env.simulator
    # for i in range(16 * 4):
    #     print(f"Task ID: {i} Mapping Priority: {sim.get_mapping_priority(i)}")

    if rank == 0:
        config = instantiate(cfg.eval)
        # start_logger()
        env.simulator.run()
        env.simulator.external_mapper = ExternalMapper()
        eft = env._get_baseline("EFT")
        print(env.simulator.time, env._get_baseline("EFT"), f"{eft/env.simulator.time:.2f}x")
        print("Interval: ", int(env.simulator.time / config.max_frames))
        start_t = time.perf_counter()
        # animate_mesh_graph(env=env, folder=Path("outputs/"))
        end_t = time.perf_counter()
        print("Plotting time:", end_t - start_t)

        # animate_mesh_graph(env=env)


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
