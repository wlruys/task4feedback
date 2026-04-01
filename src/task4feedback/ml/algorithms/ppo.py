import glob
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl._utils import compile_with_warmup
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import GAE, VTrace

from task4feedback.logging import training
from task4feedback.ml.util import (
    EvaluationConfig,
    load_checkpoint,
    log_parameter_and_gradient_norms,
    make_eval_envs,
    redistribute_rewards_uniform,
    run_evaluation,
    save_checkpoint,
)

from ..base import ActorCriticModule
from .base import AlgorithmConfig, LoggingConfig


@dataclass
class PPOConfig(AlgorithmConfig):
    implementation: str = "torchrl"
    graphs_per_collection: int = 10
    states_per_collection: int = 1920
    minibatch_size: int = 250
    epochs_per_collection: int = 4
    num_collections: int = 1000
    workers: int = 1
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    threads_per_worker: int = 1
    collect_device: str = "cpu"
    update_device: str = "cpu"
    storing_device: str = "cpu"
    gamma: float = 1
    lmbda: float = 0.99
    normalize_advantage: bool = False
    value_norm: str = "l2"
    compile_policy: bool = False
    compile_update: bool = False
    compile_advantage: bool = False
    collector: str = "multi_sync"  # "sync" or "multi_sync"
    sample_slices: bool = True
    slice_len: int = 16
    rollout_steps: int = 250
    advantage_type: str = "gae"  # "gae" or "vtrace"
    bagged_policy: str = "uniform"
    timeout: int = 60 * 60 * 24 * 2  # 2 day


def should_log(n_updates: int, logging_config: LoggingConfig | None) -> bool:
    if logging_config is None:
        return False
    return (
        logging_config.stats_interval > 0
        and n_updates % logging_config.stats_interval == 0
    )


def should_eval(n_updates: int, eval_config: EvaluationConfig | None) -> bool:
    if eval_config is None:
        return False
    return eval_config.eval_interval > 0 and n_updates % eval_config.eval_interval == 0


def should_checkpoint(n_updates: int, logging_config: LoggingConfig | None) -> bool:
    if logging_config is None:
        return False
    return (
        logging_config.checkpoint_interval > 0
        and n_updates % logging_config.checkpoint_interval == 0
    )


def _checkpoint_prefix(logging_config: LoggingConfig) -> str:
    return (
        logging_config.best_policy_name
        if logging_config.best_policy_name
        else "checkpoint"
    )


def _remove_old_checkpoints(
    checkpoint_dir: str, prefix: str, seed: int, keep_path: str
) -> None:
    pattern = os.path.join(checkpoint_dir, f"*_{prefix}_{seed}.pt")
    for old_file in glob.glob(pattern):
        if os.path.abspath(old_file) == os.path.abspath(keep_path):
            continue
        try:
            os.remove(old_file)
            training.info(f"Removed old checkpoint for seed {seed}: {old_file}")
        except OSError as e:
            training.warning(f"Failed to remove {old_file}: {e}")


def _save_best_checkpoint_if_dir_set(
    *,
    logging_config: LoggingConfig,
    seed: int,
    score: float,
    policy_module,
    value_module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    n_collections: int,
) -> str | None:
    if logging_config.best_policy_dir is None:
        return None

    prefix = _checkpoint_prefix(logging_config)
    filename = f"{score:.3f}_{prefix}_{seed}.pt"
    checkpoint_path = os.path.join(logging_config.best_policy_dir, filename)

    _remove_old_checkpoints(
        logging_config.best_policy_dir, prefix, seed, checkpoint_path
    )

    save_checkpoint(
        n_collections,
        policy_module=policy_module,
        value_module=value_module,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        filename=filename,
        checkpoint_dir=logging_config.best_policy_dir,
    )
    return checkpoint_path


def log_training_metrics(
    flattened_data: TensorDict,
    tensordict_data: TensorDict,
    loss: dict[str, torch.Tensor],
    loss_module: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    n_updates: int,
    n_collections: int,
    n_samples: int,
) -> dict[str, Any]:
    """Log training metrics to wandb and return the payload."""
    with torch.no_grad():
        rewards = flattened_data.get(("next", "reward"))
        improvements = flattened_data.get(("next", "observation", "aux", "improvement"))
        vs_policy = flattened_data.get(("next", "observation", "aux", "vs_policy"))

        # Defaults that won't crash wandb logging
        avg_reward = float("nan")
        std_reward = None

        if rewards is not None and rewards.numel() > 0:
            avg_reward = rewards.mean().item()
            std_reward = rewards.std().item() if rewards.numel() > 1 else None

        valid_improvements = None
        valid_quad = None
        if improvements is not None and vs_policy is not None:
            valid_mask = torch.isfinite(improvements) & (improvements > -100)
            if valid_mask.any():
                valid_improvements = improvements[valid_mask]
                valid_quad = vs_policy[valid_mask]

        # Advantage & targets
        advantage_mean = tensordict_data["advantage"].mean().item()
        advantage_std = tensordict_data["advantage"].std().item()
        value_target_mean = tensordict_data["value_target"].mean().item()
        value_target_std = tensordict_data["value_target"].std().item()

        explained_variance = None
        if (
            "state_value" in flattened_data.keys()
            and "value_target" in flattened_data.keys()
        ):
            values = flattened_data["state_value"].squeeze(-1)
            targets = flattened_data["value_target"].squeeze(-1)
            if targets.numel() > 1:
                var_targets = targets.var(unbiased=False)
                if torch.isfinite(var_targets) and var_targets > 1e-8:
                    residual_var = (targets - values).var(unbiased=False)
                    explained_variance = (1.0 - (residual_var / var_targets)).item()

        post_clip_norms = log_parameter_and_gradient_norms(loss_module)

        log_payload: dict[str, Any] = {
            **post_clip_norms,
            "batch/n_updates": n_updates,
            "batch/n_collections": n_collections,
            "batch/n_samples": n_samples,
            "batch/avg_reward": avg_reward,
            "batch/policy_loss": loss["loss_objective"].item(),
            "batch/critic_loss": loss["loss_critic"].item(),
            "batch/entropy_loss": loss["loss_entropy"].item(),
            "batch/entropy": loss["entropy"].item(),
            "batch/kl_approx": loss["kl_approx"].item(),
            "batch/clip_fraction": loss["clip_fraction"].item(),
            "batch/ESS": loss["ESS"].item(),
            "batch/advantage_mean": advantage_mean,
            "batch/advantage_std": advantage_std,
            "batch/mean_value_target": value_target_mean,
            "batch/std_value_target": value_target_std,
            "batch/lr": optimizer.param_groups[0]["lr"],
        }

        # Extra loss keys
        for k, v in loss.items():
            if k in {
                "loss_objective",
                "loss_critic",
                "loss_entropy",
                "entropy",
                "kl_approx",
                "clip_fraction",
                "ESS",
            }:
                continue
            try:
                log_payload[f"batch/{k}"] = v.item() if hasattr(v, "item") else float(v)
            except Exception:
                pass

        if std_reward is not None:
            log_payload["batch/std_reward"] = std_reward
        if explained_variance is not None:
            log_payload["batch/explained_variance"] = explained_variance

        # Improvement metrics
        if valid_improvements is not None and valid_improvements.numel() > 0:
            log_payload.update(
                {
                    "batch/mean_improvement": valid_improvements.mean().item(),
                    "batch/max_improvement": valid_improvements.max().item(),
                    "batch/min_improvement": valid_improvements.min().item(),
                    "batch/mean_vs_policy": valid_quad.mean().item(),
                    "batch/max_vs_policy": valid_quad.max().item(),
                    "batch/min_vs_policy": valid_quad.min().item(),
                }
            )
            if valid_improvements.numel() > 1:
                log_payload["batch/std_improvement"] = valid_improvements.std().item()

            training.info(
                f"Average training improvement: {log_payload['batch/mean_improvement']}"
            )

        msg_parts = []
        for key, value in loss.items():
            try:
                scalar = value.item() if hasattr(value, "item") else float(value)
                msg_parts.append(f"{key}={scalar:.4f}")
            except Exception:
                continue
        training.info(f"[LOSS] {' | '.join(msg_parts)}")

        wandb.log(log_payload)
        return log_payload


def _build_advantage_module(
    actor_critic_module: ActorCriticModule, ppo_config: PPOConfig
):
    if ppo_config.advantage_type == "gae":
        training.info("Using GAE for advantage estimation")
        module = GAE(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            average_gae=False,
            device=ppo_config.update_device,
            vectorized=(False if ppo_config.compile_advantage else True),
            deactivate_vmap=True,
        )
    elif ppo_config.advantage_type == "vtrace":
        training.info("Using VTrace for advantage estimation")
        module = VTrace(
            gamma=ppo_config.gamma,
            value_network=actor_critic_module.critic,
            actor_network=actor_critic_module.actor,
            device=ppo_config.update_device,
        )
    else:
        raise ValueError(f"Unknown advantage_type: {ppo_config.advantage_type}")
    return module.to(ppo_config.update_device)


def _build_loss_module(
    actor_critic_module: ActorCriticModule, ppo_config: PPOConfig
) -> ClipPPOLoss:
    loss_module = ClipPPOLoss(
        actor_network=actor_critic_module.actor,
        critic_network=actor_critic_module.critic,
        clip_epsilon=ppo_config.clip_eps,
        entropy_bonus=True,
        entropy_coeff=ppo_config.ent_coef,
        critic_coeff=ppo_config.val_coef,
        loss_critic_type=ppo_config.value_norm,
        clip_value=ppo_config.clip_vloss,
        normalize_advantage=ppo_config.normalize_advantage,
    )

    if ppo_config.advantage_type == "gae":
        loss_module.make_value_estimator(ValueEstimators.GAE)
    elif ppo_config.advantage_type == "vtrace":
        loss_module.make_value_estimator(ValueEstimators.VTrace)

    return loss_module.to(ppo_config.update_device)


def _build_replay_buffer(
    ppo_config: PPOConfig, max_states_per_collection: int
) -> TensorDictReplayBuffer:
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=max_states_per_collection, device=ppo_config.update_device
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=ppo_config.minibatch_size,
    )


def _build_collector(
    actor_critic_module: ActorCriticModule,
    env_constructors: list[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    max_states_per_collection: int,
):
    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.graphs_per_collection)
        ]

    reset_each_iter = False if ppo_config.rollout_steps > 0 else True

    if ppo_config.collector == "multi_sync":
        return MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            cat_results="stack",
            reset_at_each_iter=reset_each_iter,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            num_threads=ppo_config.workers,
            compile_policy=(
                {"mode": "reduce-overhead"} if ppo_config.compile_policy else None
            ),
        )

    if ppo_config.collector == "sync":
        return SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=reset_each_iter,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy=(
                {"mode": "reduce-overhead"} if ppo_config.compile_policy else None
            ),
        )

    raise ValueError(
        f"Unknown collector type: {ppo_config.collector}. Use 'sync' or 'multi_sync'."
    )


def run_ppo(
    actor_critic_module: ActorCriticModule,
    env_constructors: list[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: LoggingConfig | None,
    eval_config: EvaluationConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    seed: int = 0,
    resume_from: str | Path | None = None,
):
    # Global threading control
    if ppo_config.threads_per_worker and ppo_config.threads_per_worker > 0:
        torch.set_num_threads(ppo_config.threads_per_worker)

    if logging_config is not None and logging_config.stats_interval:
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    eval_envs = make_eval_envs(env_constructors, eval_config)

    # Determine max rollout length
    max_tasks = max([env.size() for env in eval_envs]) if eval_envs else 1
    if ppo_config.rollout_steps > 0:
        max_tasks = ppo_config.rollout_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks

    advantage_module = _build_advantage_module(actor_critic_module, ppo_config)
    loss_module = _build_loss_module(actor_critic_module, ppo_config)
    replay_buffer = _build_replay_buffer(ppo_config, max_states_per_collection)
    collector = _build_collector(
        actor_critic_module, env_constructors, ppo_config, max_states_per_collection
    )
    collector.set_seed(seed)

    if optimizer is None:
        optimizer = torch.optim.Adam(loss_module.parameters())
    else:
        optimizer = optimizer(loss_module.parameters())
    training.info(f"Using optimizer: {optimizer}")

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)
        training.info(f"Using learning rate scheduler: {lr_scheduler}")

    if resume_from is not None:
        state = load_checkpoint(
            resume_from,
            policy_module=actor_critic_module.actor,
            value_module=actor_critic_module.critic,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        start_step = state["step"]

    def update_policy(batch: TensorDict) -> dict[str, torch.Tensor]:
        loss_vals = loss_module(batch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=ppo_config.max_grad_norm
        )
        optimizer.step()
        return loss_vals

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(
            advantage_module, mode="reduce-overhead", warmup=8
        )

    if ppo_config.compile_update:
        update_policy = compile_with_warmup(
            update_policy, mode="reduce-overhead", warmup=8
        )

    states_per_collection = min(
        ppo_config.states_per_collection, max_states_per_collection
    )
    n_batch = max(1, states_per_collection // ppo_config.minibatch_size)

    if ppo_config.minibatch_size > states_per_collection:
        training.warning(
            f"Minibatch size <{ppo_config.minibatch_size}> is larger than states per collection <{states_per_collection}>."
        )

    training.info(
        f"Running PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers."
    )

    training.info("Starting PPO training loop")

    n_collections = start_step if resume_from is not None else 0
    n_updates = n_collections * ppo_config.epochs_per_collection * n_batch
    n_samples = n_collections * max_states_per_collection
    eval_max_performance = 0.0
    batch_threshold = 0.8
    safe_to_eval = False

    start_t = time.perf_counter()

    for collection_idx, tensordict_data in enumerate(collector):
        n_collections += 1
        replay_buffer.empty()

        if n_collections >= ppo_config.num_collections:
            break

        # perf logging
        elapsed_time = time.perf_counter() - start_t
        collections_per_second = (
            (collection_idx + 1) / elapsed_time if elapsed_time > 0 else 0.0
        )
        seconds_per_collection = (
            elapsed_time / (collection_idx + 1) if (collection_idx + 1) > 0 else 0.0
        )
        training.info(
            f"Collection {n_collections}/{ppo_config.num_collections}, "
            f"Collections/s: {collections_per_second:.2f}, "
            f"ms/Collection: {seconds_per_collection * 1000:.2f}"
        )

        tensordict_data = tensordict_data.to(
            ppo_config.update_device, non_blocking=True
        )

        # Advantages
        adv_start = time.perf_counter()
        with torch.no_grad():
            if ppo_config.bagged_policy == "uniform":
                redistribute_rewards_uniform(tensordict_data)
            advantage_module(tensordict_data)
        training.info(
            f"Computed advantages {n_collections} in {time.perf_counter() - adv_start:.2f} seconds"
        )

        flattened_data = tensordict_data.reshape(-1)
        samples_in_collection = int(flattened_data.shape[0])
        n_samples += samples_in_collection
        replay_buffer.extend(flattened_data)

        # Updates
        update_start = time.perf_counter()
        loss_module.actor_network.train()
        loss_module.critic_network.train()

        for _epoch in range(ppo_config.epochs_per_collection):
            for _ in range(n_batch):
                n_updates += 1
                batch = replay_buffer.sample(ppo_config.minibatch_size)
                batch = batch.to(ppo_config.update_device, non_blocking=True)

                loss = update_policy(batch)

                if should_log(n_updates, logging_config):
                    wandb_log = log_training_metrics(
                        flattened_data=flattened_data,
                        tensordict_data=tensordict_data,
                        loss=loss,
                        loss_module=loss_module,
                        optimizer=optimizer,
                        n_updates=n_updates,
                        n_collections=n_collections,
                        n_samples=n_samples,
                    )
                    mean_impr = wandb_log.get("batch/mean_improvement", float("-inf"))
                    if round(mean_impr, 2) > round(batch_threshold, 2):
                        safe_to_eval = True

        # Push updated weights back to collector policy (collect-device)
        collector.update_policy_weights_(
            TensorDict.from_module(loss_module.actor_network).to(
                ppo_config.collect_device
            )
        )

        training.info(
            f"Updated policy {n_collections} in {time.perf_counter() - update_start:.2f} seconds"
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Periodic evaluation
        if should_eval(n_collections, eval_config=eval_config) and safe_to_eval:
            collector.policy.eval()
            metrics = run_evaluation(
                collector.policy,
                eval_envs,
                eval_config,
                n_collections,
                n_updates,
                n_samples,
            )

            if eval_config is not None and eval_config.pickle_path is not None:
                eval_score = metrics["eval/DETERMINISTIC"]["mean_vsPolicy"]
                if eval_score > eval_max_performance:
                    eval_max_performance = float(eval_score)
                    training.info(
                        f"New max performance: {eval_max_performance:.4f}. Saving checkpoint."
                    )
                    if logging_config is not None:
                        _save_best_checkpoint_if_dir_set(
                            logging_config=logging_config,
                            seed=seed,
                            score=eval_max_performance,
                            policy_module=collector.policy,
                            value_module=loss_module.critic_network,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            n_collections=n_collections,
                        )

        # Periodic checkpoint
        if should_checkpoint(n_collections, logging_config):
            training.info(f"Checkpointing at collection {n_collections}")
            save_checkpoint(
                n_collections,
                policy_module=collector.policy,
                value_module=loss_module.critic_network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                checkpoint_dir=logging_config.checkpoint_dir,
            )

        # Timeout
        elapsed_time = time.perf_counter() - start_t
        if elapsed_time > ppo_config.timeout:
            training.warning(
                f"Timeout reached after {elapsed_time:.2f} seconds. Stopping training."
            )
            break

    # Final evaluation
    if eval_config is not None and eval_config.eval_interval > 0:
        training.info("Running final evaluation after training")
        run_evaluation(
            collector.policy,
            eval_envs,
            eval_config,
            n_collections,
            n_updates,
            n_samples,
        )

    # Final checkpoint
    save_checkpoint(
        n_collections,
        policy_module=collector.policy,
        value_module=loss_module.critic_network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint_dir=logging_config.checkpoint_dir,
    )

    collector.shutdown()
