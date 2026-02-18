import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from task4feedback.exp_utils.graph import make_graph_builder
from task4feedback.exp_utils.env import make_env
from task4feedback.exp_utils.model import create_td_models
from task4feedback.ml.rl_utils import warmup_lazy_modules
from task4feedback.exp_utils.algorithm import create_optimizer, create_lr_scheduler

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from omegaconf import DictConfig, open_dict
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from task4feedback.exp_utils.run_name import make_run_name, cfg_hash

import torch
import numpy
import random


def configure_training(cfg: DictConfig):
    # start_logger()
    graph_builder = make_graph_builder(cfg)
    env, normalization = make_env(graph_builder=graph_builder, cfg=cfg)

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, reference, lstm = create_td_models(cfg, feature_config)

    network = reference

    # Warm up to materialize Lazy* parameters before logging or cloning.
    try:
        warmup_lazy_modules(model, env, warmup_steps=2)
    except Exception as exc:
        print(f"Model warmup for lazy init failed: {exc}")

    def env_fn(eval: bool = False):
        return make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            lstm=lstm,
            normalization=normalization,
            eval=eval,
        )

    alg_config = instantiate(cfg.algorithm)

    optimizer, lr_scheduler = create_optimizer(cfg)
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

    if lstm is not None:
        run_ppo_lstm(
            actor_critic_module=model,
            env_constructors=[env_fn],
            logging_config=logging_config,
            ppo_config=alg_config,
            eval_config=eval_config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=cfg.seed,
        )
    else:
        run_ppo(
            actor_critic_module=model,
            env_constructors=[env_fn],
            logging_config=logging_config,
            ppo_config=alg_config,
            eval_config=eval_config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            seed=cfg.seed,
        )


@hydra.main(config_path="conf", config_name="8x8x128_dynamic_diag_cnn.yaml", version_base=None)
def main(cfg: DictConfig):
    if cfg.wandb.enabled:
        wandb.init(
            project="8x8x128_sweep",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=make_run_name(cfg),
            # name=f"{cfg.wandb.name}",
            dir=cfg.wandb.dir,
            tags=cfg.wandb.tags,
        )

        with open_dict(cfg):
            for k, v in wandb.config.items():
                OmegaConf.update(cfg, k, v, merge=True)

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
