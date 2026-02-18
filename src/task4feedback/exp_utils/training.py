from __future__ import annotations
from .definitions import *
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from .algorithm import create_optimizer
from .artifacts import (
    load_normalization_state,
    save_normalization_state,
    model_cache_context,
    normalization_cache_context,
)
from .env import NormalizationDetails, make_env
from .eval import ensure_eval_location
from .graph import GraphBuilder, make_graph_builder
from .model import create_td_models
from .run_name import make_run_name
from task4feedback.logging import training
from task4feedback.ml.models import FeatureDimConfig
from task4feedback.ml.rl_utils import compute_model_fingerprint, warmup_lazy_modules


@dataclass
class ModelBundle:
    model: torch.nn.Module
    reference: Any
    lstm: Optional[Any]
    feature_config: FeatureDimConfig


def set_global_seeds(seed: int, deterministic_torch: bool) -> None:
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def prepare_logging_config(cfg: DictConfig):
    logging_config = instantiate(cfg.logging)
    model_ctx = model_cache_context(cfg)
    if logging_config.best_policy_dir is None:
        logging_config.best_policy_dir = str(model_ctx.dir)
    if logging_config.best_policy_name is None:
        logging_config.best_policy_name = make_run_name(cfg)

    best_policy_path = Path(logging_config.best_policy_dir)
    best_policy_path.mkdir(parents=True, exist_ok=True)
    return logging_config, best_policy_path, model_ctx


def load_normalization(cfg: DictConfig) -> tuple[Optional[NormalizationDetails], Any]:
    norm_ctx = normalization_cache_context(cfg)
    normalization = load_normalization_state(cfg, ctx=norm_ctx)
    return normalization, norm_ctx


def initialize_env_and_model(
    cfg: DictConfig, graph_builder: GraphBuilder, normalization: Optional[NormalizationDetails]
) -> Tuple[Any, Optional[NormalizationDetails], Callable[[bool], Any], ModelBundle]:
    env_result = make_env(graph_builder=graph_builder, cfg=cfg, normalization=normalization)
    if isinstance(env_result, tuple):
        env, normalization = env_result
    else:
        env = env_result

    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, reference, lstm = create_td_models(cfg, feature_config)

    # Warm up to materialize Lazy* parameters before logging or cloning.
    try:
        warmup_lazy_modules(model, env, warmup_steps=2)
    except Exception as exc:
        training.warning("Model warmup for lazy init failed: %s", exc)

    def env_fn(eval: bool = False):
        return make_env(
            graph_builder=graph_builder,
            cfg=cfg,
            lstm=lstm,
            normalization=normalization,
            eval=eval,
        )

    bundle = ModelBundle(
        model=model,
        reference=reference,
        lstm=lstm,
        feature_config=feature_config,
    )
    return env, normalization, env_fn, bundle


def prepare_training_components(cfg: DictConfig):
    alg_config = instantiate(cfg.algorithm)
    optimizer, lr_scheduler = create_optimizer(cfg)
    eval_config = instantiate(cfg.eval)
    eval_location = ensure_eval_location(cfg)
    return alg_config, optimizer, lr_scheduler, eval_config, eval_location


def persist_config_and_normalization(
    cfg: DictConfig,
    normalization: Optional[NormalizationDetails],
):
    return save_normalization_state(cfg, normalization)


def parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    from torch.nn.parameter import UninitializedParameter

    params = list(model.parameters())
    uninit = [p for p in params if isinstance(p, UninitializedParameter)]
    if uninit:
        training.warning(
            "Skipping %d uninitialized Lazy parameters in count.",
            len(uninit),
        )
    total_params = sum(
        p.numel() for p in params if not isinstance(p, UninitializedParameter)
    )
    trainable_params = sum(
        p.numel()
        for p in params
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )
    return int(total_params), int(trainable_params)


def maybe_log_model_to_wandb(cfg: DictConfig, logging_config, model: torch.nn.Module, total_params: int, trainable_params: int) -> None:
    if not cfg.wandb.enabled or wandb.run is None:
        return

    model_fingerprint = compute_model_fingerprint(model)
    wandb.run.summary["model/fingerprint"] = model_fingerprint
    wandb.run.summary["model/parameters_total"] = total_params
    wandb.run.summary["model/parameters_trainable"] = trainable_params

    try:
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
        arch_file = hydra_output_dir / "model_arch.txt"
        arch_file.write_text(str(model))
        wandb.save(str(arch_file))
    except Exception as e:
        training.warning("Failed to save model architecture: %s", e)

    if logging_config.watch_model:
        training.info(
            "Enabling wandb.watch with mode=%s, freq=%s. This may slow training.",
            logging_config.watch_log_mode,
            logging_config.watch_log_freq,
        )
        try:
            wandb.watch(
                model,
                log=logging_config.watch_log_mode,
                log_freq=logging_config.watch_log_freq,
            )
        except Exception as e:
            training.warning("wandb.watch failed: %s", e)
    else:
        training.info("wandb.watch is disabled (set logging.watch_model=true to enable)")


def build_graph_builder(cfg: DictConfig) -> GraphBuilder:
    return make_graph_builder(cfg)


def build_signature(cfg: DictConfig) -> ArtifactSignature:
    from .artifacts import ArtifactSignature

    return ArtifactSignature.from_cfg(cfg)
