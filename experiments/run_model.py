import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from helper.graph import make_graph_builder
from helper.env import make_env
from helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
from helper.algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *
from task4feedback.ml.util import *
from task4feedback.graphs.jacobi import JacobiRoundRobinMapper, LevelPartitionMapper
from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph
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

from torchrl.envs import set_exploration_type, ExplorationType

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
    # start_logger()
    graph_builder = make_graph_builder(cfg)
    env, norm = make_env(graph_builder=graph_builder, cfg=cfg, eval=True)
    
    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, lstm = create_td_actor_critic_models(cfg, feature_config)

    # Attempt to load policy weights from a local checkpoint next to this file
    ckpt_path = Path(__file__).resolve().parent / "model.pt"
    if ckpt_path.exists():
        loaded = load_policy_from_checkpoint(model, ckpt_path)
        if not loaded:
            print(f"Found {ckpt_path}, but no compatible policy module to load into.")
    else:
        print(f"No model checkpoint found at {ckpt_path}; proceeding with randomly initialized policy.")

    eval_env= make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            normalization=norm,
            eval=True,
        )
    def rr_mapper() -> LevelPartitionMapper:
        return JacobiRoundRobinMapper(
            n_devices=4,
            setting=0,
        )
    model.eval()
    config = EvaluationConfig
    initloc: list[dict] = []
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for i in range(20):
            obs = eval_env.reset()
            cyclic=eval_env._get_baseline("Cyclic")
            
            td = eval_env.rollout(policy=model.actor, max_steps=10000, auto_reset=False, tensordict=obs)
            
            print(eval_env.EFT_baseline, cyclic, eval_env.EFT_baseline/td['observation','aux','time'][-1].item(), cyclic/td['observation','aux','time'][-1].item())

    


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
