import os
import random
import time
from email import policy
from pathlib import Path

import git
import hydra
import numpy
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn

# from task4feedback.graphs.mesh.plot_fast import *
# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")
from hydra.experimental.callbacks import Callback
from hydra.utils import instantiate
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict

from task4feedback.experiment_helper.algorithm import (
    create_lr_scheduler,
    create_optimizer,
)
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.model import create_td_actor_critic_models
from task4feedback.experiment_helper.parmetis import find_best_cfg_optuna, run_parmetis
from task4feedback.experiment_helper.run_name import cfg_hash, make_run_name
from task4feedback.fastsim2 import ParMETIS_wrapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
from task4feedback.graphs.jacobi import (
    BlockCyclicMapper,
    GraphMETISMapper,
    JacobiGraph,
    JacobiQuadrantMapper,
    JacobiRoundRobinMapper,
    LevelPartitionMapper,
)
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.interface.wrappers import *
from task4feedback.ml.algorithms.ppo import run_ppo
from task4feedback.ml.models import *
from task4feedback.ml.util import EvaluationConfig

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def configure_training(cfg: DictConfig):
    # start_logger()
    option = "ParMETIS"
    cfg.graph.config.steps = 256
    for i in range(1):
        if rank == 0:
            graph_builder = make_graph_builder(cfg)
            env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
            graph = env.get_graph()
            # env.set_reset_counter(324)
            if isinstance(graph, DynamicJacobiGraph):
                workload = graph.get_workload()
                workload.animate_workload(
                    show=False, title="outputs/workload_animation.mp4"
                )
            exit()
            # print("Running option: EFT")
            # start = time.time()
            # # env.rollout(max_steps=999)
            # env.simulator.disable_external_mapper()
            # env.simulator.run()
            # end = time.time()
            # print(f"EFT time: {end - start:.2f} seconds")
            # exit()
        else:
            env = None

        if option == "EFT" and rank == 0:
            env.simulator.disable_external_mapper()
            env.simulator.run()
        elif option == "Oracle" and rank == 0:
            graph.mincut_per_levels(
                bandwidth=cfg.system.d2d_bw,
                mode="metis",
                offset=1,
                level_chunks=1,
            )
            graph.align_partitions()
            env.simulator.enable_external_mapper()
            env.simulator.external_mapper = LevelPartitionMapper(
                level_cell_mapping=graph.partitions
            )
        elif option == "BlockCyclic":
            env.simulator.enable_external_mapper()
            env.simulator.external_mapper = BlockCyclicMapper(
                geometry=graph.data.geometry,
                n_devices=cfg.system.n_devices - 1,
                block_size=2,
                offset=1,
                verbose=True,
            )
            env.simulator.run()
        elif option == "GraphMETISMapper":
            env.simulator.enable_external_mapper()
            env.simulator.external_mapper = GraphMETISMapper(
                graph=graph, n_devices=cfg.system.n_devices - 1, offset=1
            )
            env.simulator.run()
        elif option == "Quad":
            env.simulator.enable_external_mapper()
            env.simulator.external_mapper = JacobiQuadrantMapper(
                n_devices=cfg.system.n_devices - 1, graph=graph, offset=1
            )
            env.simulator.run()
        elif option == "Cyclic":
            env.simulator.enable_external_mapper()
            env.simulator.external_mapper = JacobiRoundRobinMapper(
                n_devices=cfg.system.n_devices - 1, offset=1, setting=1
            )
            env.simulator.run()
        elif option == "ParMETIS":
            ParMETIS = ParMETIS_wrapper()

            best_cfg = find_best_cfg_optuna(
                cfg, ParMETIS, env=env, skip_search=True, mode="normal_optuna"
            )
            if best_cfg is None:
                print("No valid configuration found for ParMETIS.")
                return
            run_parmetis(
                sim=env.simulator if rank == 0 else None,
                cfg=cfg,
                itr=best_cfg[0],
                unbalance=best_cfg[1],
                n_compute_devices=cfg.system.n_devices - 1,
                ParMETIS=ParMETIS,
                skip_error=True,
            )
        else:
            raise ValueError(f"Unknown option: {option}")

        if rank == 0:
            print(f"{option}: {env.simulator.time}")
            animate_mesh_graph(env=env, folder="./", filename=f"{option}.mp4")


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
