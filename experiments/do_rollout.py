import hydra
from omegaconf import DictConfig

from helper.graph import make_graph_builder
from helper.env import make_env

from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

from omegaconf import DictConfig
import torch
import numpy
import random


def configure_training(cfg: DictConfig):
    # start_logger()

    option = "Oracle"
    graph_builder = make_graph_builder(cfg)
    env = make_env(
        graph_builder=graph_builder,
        cfg=cfg,
        normalization=False,
    )
    env._reset()
    env.rollout(max_steps=99999, auto_reset=False)


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
