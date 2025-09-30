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
    # Attempt to load policy weights from a local checkpoint next to this file
    ckpt_path = Path(__file__).resolve().parent / "saved_models" / "8x8x128_corners_D.pt"
    # if 8x8x128_corners_D_norm.pkl exists, load it
    # else, create it and save it
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")
    graph_builder = make_graph_builder(cfg)
    if ckpt_path.with_name(ckpt_path.stem + "_norm.pkl").exists():
        print(f"Loading normalization from {ckpt_path.with_name(ckpt_path.stem + '_norm.pkl')}")
        norm = pickle.load(open(ckpt_path.with_name(ckpt_path.stem + "_norm.pkl"), "rb"))
        env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=norm, eval=True)
    else:
        env, norm = make_env(graph_builder=graph_builder, cfg=cfg, eval=True)
        pickle.dump(norm, open(ckpt_path.with_name(ckpt_path.stem + "_norm.pkl"), "wb"))

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, _, _ = create_td_actor_critic_models(cfg, feature_config)

    loaded = load_policy_from_checkpoint(model, ckpt_path)
    if not loaded:
        print(f"Found {ckpt_path}, but no compatible policy module to load into.")

    eval_env = make_env(
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
            eft_time = eval_env._get_baseline("EFT")

            policy_sim = eval_env.simulator.copy()
            graph = eval_env.get_graph()
            graph.mincut_per_levels(
                bandwidth=cfg.system.d2d_bw,
                mode="metis",
                offset=1,
                level_chunks=64,
            )
            graph.align_partitions()
            policy_sim.enable_external_mapper()
            policy_sim.external_mapper = LevelPartitionMapper(level_cell_mapping=graph.partitions)

            policy_sim.run()

            td = eval_env.rollout(policy=model.actor, max_steps=10000, auto_reset=False, tensordict=obs)

            print(eft_time, td["observation", "aux", "time"][-1].item(), eft_time / td["observation", "aux", "time"][-1].item())
            print(
                f"EFT time: {eft_time}, Best policy time: {policy_sim.time} ML: {td['observation', 'aux', 'time'][-1].item()}, vsEFT: {eft_time / td['observation', 'aux', 'time'][-1].item():.2f}x vsBest: {policy_sim.time / td['observation', 'aux', 'time'][-1].item():.2f}x"
            )


@hydra.main(config_path="conf", config_name="dynamic_batch.yaml", version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg)


if __name__ == "__main__":
    main()
