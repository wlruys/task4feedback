# src/train.py
import hydra
from omegaconf import DictConfig, OmegaConf
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.ml.env import *
from task4feedback.ml.algorithms import *
from task4feedback.interface import TaskTuple
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import *
from functools import partial

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm

# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from rich import print as rprint
from torchrl.envs.utils import check_env_specs
from torchrl.envs import ParallelEnv


def configure_training(cfg: DictConfig):
    graph_builder = make_graph_builder(cfg)
    env, normalization = make_env(graph_builder=graph_builder, cfg=cfg)

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    print(f"Feature config: {feature_config}")

    a, lstm = create_td_actor_critic_models(cfg, feature_config)

    def env_fn():
        return make_env(
            graph_builder=graph_builder, cfg=cfg, lstm=lstm, normalization=normalization
        )

    alg_config = instantiate(cfg.algorithm)

    if cfg.wandb.enabled:
        logging_config = instantiate(cfg.logging)
    else:
        logging_config = None

    if lstm is not None:
        run_ppo_lstm(
            actor_critic_module=a,
            env_constructors=[env_fn],
            logging_config=logging_config,
            ppo_config=alg_config,
        )
    else:
        run_ppo(
            actor_critic_module=a,
            env_constructors=[env_fn],
            logging_config=logging_config,
            ppo_config=alg_config,
        )
    


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.wandb.name,
            dir=cfg.wandb.dir,
        )

    configure_training(cfg)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
