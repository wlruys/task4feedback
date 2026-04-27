from __future__ import annotations

import wandb
from omegaconf import DictConfig, OmegaConf

from helper.run_name import make_model_artifact_name, make_run_name


def model_artifact_settings(cfg: DictConfig) -> dict:
    enabled = bool(OmegaConf.select(cfg, "wandb.model_artifact.enabled", default=False))
    if not (cfg.wandb.enabled and enabled):
        return {
            "wandb_model_artifact_name": None,
            "log_eval_checkpoints_to_wandb": False,
            "log_final_checkpoint_to_wandb": False,
        }

    if wandb.run is not None and wandb.run.name is not None:
        run_name = wandb.run.name
    elif cfg.wandb.name not in (None, "default"):
        run_name = cfg.wandb.name
    else:
        run_name = make_run_name(cfg)

    return {
        "wandb_model_artifact_name": make_model_artifact_name(run_name),
        "log_eval_checkpoints_to_wandb": bool(
            OmegaConf.select(cfg, "wandb.model_artifact.log_eval_checkpoints", default=True)
        ),
        "log_final_checkpoint_to_wandb": bool(
            OmegaConf.select(cfg, "wandb.model_artifact.log_final_checkpoint", default=True)
        ),
    }
