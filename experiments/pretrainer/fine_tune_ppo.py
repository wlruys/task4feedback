import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
from task4feedback.experiment_helper.algorithm import create_optimizer, create_lr_scheduler

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
from task4feedback.experiment_helper.run_name import make_run_name, cfg_hash

import torch
import numpy
import random
import pickle


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


def load_phase1_dataset(save_dir: Path, max_eps=1024):
    temp_dir = save_dir / "temp"
    assert temp_dir.exists(), f"Phase 1 dataset not found at {temp_dir}"

    episodes = []
    for epfile in sorted(temp_dir.glob("episode_*.pkl")):
        with open(epfile, "rb") as f:
            td = pickle.load(f)
        episodes.append(td)
        if len(episodes) >= max_eps:
            break

    dataset = torch.cat(episodes, dim=0)
    print(f"[DAgger] Loaded Phase 1 expert dataset: {dataset.batch_size}")
    return dataset


def configure_training(cfg: DictConfig):
    # start_logger()
    save_dir = Path("dataset/phase1_expert_data")
    norm_path = save_dir / "normalization.pkl"

    assert norm_path.exists(), f"Normalization file not found at {norm_path}"
    with open(norm_path, "rb") as f:
        normalization = pickle.load(f)

    graph_builder = make_graph_builder(cfg)
    env = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization)

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, reference, lstm = create_td_actor_critic_models(cfg, feature_config)

    # actor_path = save_dir / "actor_dagger.pt"
    # model.actor.load_state_dict(torch.load(actor_path))

    # critic_path = save_dir / "critic_pretrained.pt"
    # model.critic.load_state_dict(torch.load(critic_path))

    actor_path = save_dir / "pretrained_ppo.pt"
    load_policy_from_checkpoint(model, actor_path)

    # dataset = load_phase1_dataset(save_dir, max_eps=1024)
    dataset = None

    def env_fn(eval: bool = False):
        return make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            lstm=lstm,
            normalization=normalization,
            eval=eval,
        )

    alg_config = instantiate(cfg.algorithm)

    optimizer = create_optimizer(cfg)
    lr_scheduler = create_lr_scheduler(cfg)
    logging_config = instantiate(cfg.logging)

    eval_config = instantiate(cfg.eval)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if cfg.wandb.enabled:
        wandb.run.summary["model/parameters_total"] = int(total_params)
        wandb.run.summary["model/parameters_trainable"] = int(trainable_params)
        try:
            arch_file = Path(HydraConfig.get().runtime.output_dir) / "model_arch.txt"
            arch_file.write_text(str(model))
            wandb.save(str(arch_file))
        except Exception as e:
            print(f"Failed to save model architecture: {e}")
        try:
            wandb.watch(model, log="all")
        except Exception as e:
            print(f"wandb.watch failed: {e}")

    run_ppo(
        actor_critic_module=model,
        env_constructors=[env_fn],
        logging_config=logging_config,
        ppo_config=alg_config,
        eval_config=eval_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        seed=cfg.seed,
        expert_demonstration=dataset,
    )


@hydra.main(config_path="conf", config_name="8x8x1024_dynamic_lcorners_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    # cfg.graph.config.workload_args.traj_type exist
    if cfg.graph.mesh._target_ == "task4feedback.graphs.mesh.generate_quad_mesh":
        run_name = cfg.wandb.name
        cfg.wandb.name = cfg.wandb.name + f"_{cfg.seed}"
        checkpoint_path = Path(cfg.wandb.dir)
        checkpoint_path = checkpoint_path.parent / "model_checkpoints" / f"{run_name}"
        cfg.eval.pickle_path = None
        # Make a dir if not exists
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        cfg.logging.best_policy_dir = str(checkpoint_path)
        print(f"Best Policy dir: {cfg.logging.best_policy_dir}")
        cfg.logging.best_policy_name = f"{cfg.feature.observer.version}"
        print(f"Best Policy name: {cfg.logging.best_policy_name}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.wandb.name,
            group=cfg.wandb.group,
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
