import os
import pickle
import random
from pathlib import Path

import git
import hydra
import numpy
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn

# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")
from hydra.experimental.callbacks import Callback
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from task4feedback.experiment_helper.algorithm import (
    create_lr_scheduler,
    create_optimizer,
)
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.model import (
    create_td_actor_critic_models,
    load_policy_from_checkpoint,
)
from task4feedback.experiment_helper.run_name import (
    cfg_hash,
    make_folder_name,
    make_run_name,
)
from task4feedback.graphs.mesh.plot import animate_mesh_graph
from task4feedback.interface.wrappers import *
from task4feedback.ml.algorithms.ppo import run_ppo
from task4feedback.ml.models import *


def load_normalization(folder_name: str, observer_version: str):
    norm_path = f"./norms/{folder_name}/{observer_version}_norm.pkl"
    with open(norm_path, "rb") as f:
        return pickle.load(f)


def configure_training(cfg: DictConfig):
    # start_logger()
    graph_builder = make_graph_builder(cfg)
    folder_name, _, _, _ = make_folder_name(cfg, change_name=False)
    norm = load_normalization(
        folder_name,
        cfg.feature.observer.version,
    )
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm, eval=True)

    feature_config = FeatureDimConfig.from_observer(env.get_observer())
    model, _, _ = create_td_actor_critic_models(cfg, feature_config)
    model_path = "/home/cc/task4feedback_torchrl/experiments/models/8gpus/16w_256lvl_8gpu_corners_10-0.1_1000/rl.pt"
    if not load_policy_from_checkpoint(model, model_path):
        raise RuntimeError(f"Failed to load model from {model_path}")

    env.rollout(max_steps=1000, policy=model.actor)
    # graph = env.get_graph()
    # if hasattr(graph, "workload"):
    #     workload = graph.get_workload()
    #     workload.animate_workload(show=False)
    animate_mesh_graph(env=env, folder="./", filename="rl.mp4")


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
