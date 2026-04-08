import random
import time

import hydra
import numpy
import torch
from omegaconf import DictConfig

from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.model import create_td_actor_critic_models
from task4feedback.ml.models import FeatureDimConfig


def configure_rollout(cfg: DictConfig):
    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=False)
    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, _, _ = create_td_actor_critic_models(cfg, feature_config)

    with torch.no_grad():
        start_t = time.time()
        env.rollout(max_steps=1000, policy=model.actor)
        end_t = time.time()
        print(f"Rollout completed in {end_t - start_t:.2f} seconds")


@hydra.main(
    config_path="conf",
    config_name="8x8x128_dynamic_circle_gnn.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_rollout(cfg)


if __name__ == "__main__":
    main()
