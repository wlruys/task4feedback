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
import pickle


def configure_training(cfg: DictConfig):
    # start_logger()
    eval_state = {"cfg": OmegaConf.to_yaml(cfg), "init_locs": [], "workloads": [], "eft_times": [], "quad_times": []}
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
    env.set_reset_counter(9999)

    for i in range(20):
        env.reset()
        eval_state["init_locs"].append(env.get_graph().get_cell_locations(as_dict=False))
        graph = env.get_graph()
        if isinstance(graph, DynamicJacobiGraph):
            eval_state["workloads"].append(dict(graph.get_workload().level_workload))
        else:
            eval_state["workloads"].append(None)
        eval_state["eft_times"].append(env._get_baseline("EFT"))
        # eval_state["quad_times"].append(env._get_baseline("Quad"))
        graph.mincut_per_levels(
            bandwidth=cfg.system.d2d_bw,
            mode="metis",
            offset=1,
            level_chunks=32,
        )
        graph.align_partitions()
        env.simulator.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)
        env.simulator.run()
        eval_state["quad_times"].append(env.simulator.time)
    # print(eval_state)
    pickle.dump(eval_state, open("8x8x128_diag_1:1:1.pkl", "wb"))
    print(eval_state["cfg"])
    exit()

    # env._reset()

    # eval_state = pickle.load(open("dynamic_bump_eval.pkl", "rb"))
    # for i in range(20):
    #     saved_loc = eval_state["init_locs"][i]
    #     workload = eval_state["workloads"][i]
    #     env.reset_to_state(saved_loc, workload)
    #     print(f"Eval {i}:")
    #     print("EFT:", eval_state["eft_times"][i])
    #     print("Sim EFT:", end=" ")
    #     print(env._get_baseline("EFT"))


@hydra.main(config_path="conf", config_name="static_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
