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


class GitInfo(Callback):
    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            outdir = Path(config.hydra.runtime.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "git_sha.txt").write_text(repo.head.commit.hexsha)
            (outdir / "git_dirty.txt").write_text(str(repo.is_dirty()))
            diff = repo.git.diff(None)
            (outdir / "git_diff.patch").write_text(diff)

            print(
                "Git SHA:",
                repo.head.commit.hexsha,
                " (dirty)" if repo.is_dirty() else " (clean)",
                flush=True,
            )

        except Exception as e:
            print(f"GitInfo callback failed: {e}")


def configure_training(cfg: DictConfig):
    #start_logger()
    graph_builder = make_graph_builder(cfg)
    env, normalization = make_env(graph_builder=graph_builder, cfg=cfg)

    def env_fn(eval: bool = False):
        return make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            lstm=lstm,
            normalization=normalization,
            eval=eval,
        )

    graph = env.get_graph()
    if hasattr(graph, 'workload'):
        workload = graph.get_workload()
        workload.animate_workload(show=True)


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=make_run_name(cfg),
            # name=f"{cfg.wandb.name}",
            dir=cfg.wandb.dir,
            tags=cfg.wandb.tags,
        )

        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)

        with open_dict(cfg):
            for fname in ["git_sha.txt", "git_diff.patch", "git_dirty.txt"]:
                git_file = hydra_output_dir / fname
                if git_file.exists():
                    wandb.save(str(git_file))

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
