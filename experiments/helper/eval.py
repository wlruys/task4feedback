import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from .graph import make_graph_builder
from .env import make_env
from .model import create_td_actor_critic_models
from .algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

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

def create_reference_eval(cfg: DictConfig):
    eval_state = {"cfg": OmegaConf.to_yaml(cfg), "init_locs": [], "workloads": [], "eft_times": [], "policy_times": []}
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
    env.set_reset_counter(9999)

    graph_config = cfg.graph
    system_config = cfg.system

    graph_type = graph_config.type if "type" in graph_config else "Unknown"
    graph_config_hash = cfg_hash(graph_config)
    system_config_hash = cfg_hash(system_config)

    os.makedirs(f"eval/{}")

