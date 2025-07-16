from pathlib import Path
from ..models import *
from ..util import *
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Any, Tuple
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import (
    ReplayBuffer,
    SliceSampler,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE, VTrace
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule, GRUModule
from torchrl.objectives.utils import ValueEstimators
from torchrl.envs import TransformedEnv
import copy
from tensordict import TensorDict
import wandb
import os
import torch
from torchrl._utils import compile_with_warmup
from .base import AlgorithmConfig, LoggingConfig
from ..base import ActorCriticModule
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from task4feedback.logging import training
import time
from torchrl.envs import ParallelEnv
from torchrl.collectors.utils import split_trajectories
from task4feedback.ml.util import log_parameter_and_gradient_norms


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
    seed: int = 0
    collector: str = "multi_sync"  # "sync" or "multi_sync"
    sample_slices: bool = (
        True  # if using lstm, whether slices are used instead of episodes
    )
    slice_len: int = 16  # length of slices for LSTM, only used if sample_slices is True
    rollout_steps: int = 250
    advantage_type: str = "gae"  # "gae" or "vtrace"
    bagged_policy: str = "uniform"

def should_log(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should log based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.stats_interval == 0

def log_training_metrics(
    flattened_data: TensorDict,
    tensordict_data: TensorDict,
    loss: Dict[str, torch.Tensor],
    loss_module: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    n_updates: int,
    i: int,
    n_samples: int,
) -> None:
    """Log training metrics to wandb."""
    with torch.no_grad():
        rewards = flattened_data["next", "reward"]
        improvements = flattened_data["next", "observation", "aux", "improvement"]
        valid_improvement_mask = torch.isfinite(improvements) & (improvements > -100)
        valid_improvements = improvements[valid_improvement_mask]

        # Calculate improvement metrics
        if valid_improvements.numel() > 0:
            avg_improvement = valid_improvements.mean().item()
            max_improvement = valid_improvements.max().item()
            min_improvement = valid_improvements.min().item()
            std_improvement = valid_improvements.std().item()

        # Calculate reward metrics
        if rewards.numel() > 0:
            avg_reward = rewards.mean().item()
            std_reward = rewards.std().item()

        # Calculate advantage and value target metrics
        advantage_mean = tensordict_data["advantage"].mean().item()
        advantage_std = tensordict_data["advantage"].std().item()
        value_target_mean = tensordict_data["value_target"].mean().item()
        value_target_std = tensordict_data["value_target"].std().item()

        # Get gradient and parameter norms
        post_clip_norms = log_parameter_and_gradient_norms(loss_module)

        # Base log payload
        log_payload = {
            **post_clip_norms,
            "batch/n_updates": n_updates,
            "batch/n_collections": i + 1,
            "batch/avg_reward": avg_reward,
            "batch/std_reward": std_reward,
            "batch/n_samples": n_samples,
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

        # Add improvement metrics if available
        if valid_improvements.numel() > 0:
            log_payload.update(
                {
                    "batch/mean_improvement": avg_improvement,
                    "batch/std_improvement": std_improvement,
                    "batch/max_improvement": max_improvement,
                    "batch/min_improvement": min_improvement,
                }
            )

        wandb.log(log_payload)


def checkpoint(
    step, model, optimizer, lr_scheduler=None, extras: Optional[Dict[str, Any]] = None
):
    try:
        state = dict(
            step=step,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            extras=extras or {},
        )
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()

        if wandb is not None and wandb.run is not None and wandb.run.dir is not None:
            checkpoint_dir = Path(wandb.run.dir)
        else:
            checkpoint_dir = Path(os.environ.get("HYDRA_RUNTIME_OUTPUT_DIR", "."))

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save(state, checkpoint_file)
        training.info(f"Checkpoint saved to {checkpoint_file}")

        return checkpoint_file

    except Exception as e:
        training.error(f"Failed to save checkpoint at step {step}: {e}")
        raise

def run_ppo(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
):
    if logging_config is not None and (
        logging_frequency := logging_config.stats_interval
    ):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="eval/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    max_tasks = max([make_env().size() for make_env in env_constructors])

    if ppo_config.rollout_steps > 0:
        max_tasks = ppo_config.rollout_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks
    
    
    if ppo_config.advantage_type == "gae":
        training.info("Using GAE for advantage estimation")
        advantage_module = GAE(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            average_gae=False,
            device=ppo_config.update_device,
            vectorized=(False if ppo_config.compile_advantage else True),
        )
    elif ppo_config.advantage_type == "vtrace":
        training.info("Using VTrace for advantage estimation")
        advantage_module = VTrace(
            gamma=ppo_config.gamma,
            value_network=actor_critic_module.critic,
            actor_network=actor_critic_module.actor,
            device=ppo_config.update_device,
        )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=max_states_per_collection,
            device=ppo_config.update_device,
        ),
        sampler=SamplerWithoutReplacement(),
        prefetch=4,
        batch_size=ppo_config.minibatch_size,
    )

    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.workers)
        ]

    if ppo_config.collector == "multi_sync":
        collector = MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            cat_results="stack",
            reset_at_each_iter=False if ppo_config.rollout_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy={"mode": "reduce-overhead"}
            if ppo_config.compile_policy
            else None,
        )
    elif ppo_config.collector == "sync":
        collector = SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=True if ppo_config.rollout_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy={"mode": "reduce-overhead"}
            if ppo_config.compile_policy
            else None,
        )
    else:
        raise ValueError(
            f"Unknown collector type: {ppo_config.collector}. "
            "Use 'sync' or 'multi_sync'."
        )

    collector.set_seed(ppo_config.seed)

    loss_module = ClipPPOLoss(
        actor_network=actor_critic_module.actor,
        critic_network=actor_critic_module.critic,
        clip_epsilon=ppo_config.clip_eps,
        entropy_bonus=True,
        entropy_coef=ppo_config.ent_coef,
        critic_coef=ppo_config.val_coef,
        loss_critic_type=ppo_config.value_norm,
        clip_value=ppo_config.clip_vloss,
        normalize_advantage=ppo_config.normalize_advantage,
    )

    if ppo_config.advantage_type == "gae":
        loss_module.make_value_estimator(ValueEstimators.GAE)
    elif ppo_config.advantage_type == "vtrace":
        loss_module.make_value_estimator(ValueEstimators.VTrace)

    if optimizer is None:
        optimizer = torch.optim.Adam(loss_module.parameters())
    else:
        optimizer = optimizer(loss_module.parameters())
    training.info(f"Using optimizer: {optimizer}")

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)
        training.info(f"Using learning rate scheduler: {lr_scheduler}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    def update(batch, loss_module, optimizer,ppo_config):
        loss_vals = loss_module(batch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        optimizer.zero_grad()
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
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(
        ppo_config.states_per_collection, max_states_per_collection
    )
    n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
    if ppo_config.minibatch_size > states_per_collection:
        training.warning(
            f"Minibatch size <{ppo_config.minibatch_size}> is larger than states per collection <{states_per_collection}>. "
        )

    training.info(
        f"Starting PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers."
    )

    start_t = time.perf_counter()

    n_updates = 0
    n_samples = 0

    for i, tensordict_data in enumerate(collector):
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
        seconds_per_update = elapsed_time / (i + 1) if (i + 1) > 0 else 0

        training.info(
            f"Collection {i + 1}/{ppo_config.num_collections}, "
            f"Collections/s: {updates_per_second:.2f}, "
            f"ms/Update: {seconds_per_update * 1000:.2f}"
        )

        tensordict_data = tensordict_data.to(
            ppo_config.update_device, non_blocking=True
        )

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            # Redistribute Rewards
            if ppo_config.bagged_policy == "uniform":
                redistribute_rewards_uniform(tensordict_data)
            # Compute advantages
            advantage_module(tensordict_data)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        flattened_data = tensordict_data.reshape(-1)
        samples_in_collection = flattened_data.shape[0]
        n_samples += samples_in_collection

        replay_buffer.extend(flattened_data)

        update_start_t = time.perf_counter()
        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                batch = replay_buffer.sample(ppo_config.minibatch_size)
                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, loss_module, optimizer, ppo_config)

                if should_log(n_updates, logging_config):
                    log_training_metrics(
                        flattened_data,
                        tensordict_data,
                        loss,
                        loss_module,
                        optimizer,
                        n_updates,
                        i,
                        n_samples,
                    )
                    
                n_updates += 1

            collector.update_policy_weights_(
                TensorDict.from_module(loss_module.actor_network).to(
                    ppo_config.collect_device
                )
            )
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()


def run_ppo_lstm(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    opt_cfg: Optional[DictConfig] = None,
    lr_cfg: Optional[DictConfig] = None,
):
    if logging_config is not None and (
        logging_frequency := logging_config.stats_interval
    ):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="eval/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    max_tasks = max([make_env().size() for make_env in env_constructors])
    print(f"Max tasks in env constructors: {max_tasks}")

    if ppo_config.rollout_steps > 0:
        max_tasks = ppo_config.rollout_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks
    
    if ppo_config.advantage_type == "gae":
        training.info("Using GAE for advantage estimation")
        advantage_module = GAE(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            average_gae=False,
            device=ppo_config.update_device,
            deactivate_vmap=True,
        )

    elif ppo_config.advantage_type == "vtrace":
        training.info("Using VTrace for advantage estimation")
        advantage_module = VTrace(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            actor_network=actor_critic_module.actor,
        device=ppo_config.update_device,
        deactivate_vmap=True,
    )

    if ppo_config.sample_slices:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max_states_per_collection,
                device=ppo_config.update_device,
            ),
            sampler=SliceSampler(
                strict_length=True,
                slice_len=ppo_config.slice_len,
                traj_key=("collector", "traj_ids"),
            ),
            batch_size=ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size // ppo_config.slice_len
    else:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max_states_per_collection,
                device=ppo_config.update_device,
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size

    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.workers)
        ]

    print(f"Creating collector with {ppo_config.workers} workers")

    if ppo_config.collector == "multi_sync":
        collector = MultiSyncDataCollector(
            env_workers(),
            actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            cat_results="stack",
            reset_at_each_iter=False if ppo_config.rollout_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy={"mode": "reduce-overhead"}
            if ppo_config.compile_policy
            else None,
        )
    elif ppo_config.collector == "sync":
        collector = SyncDataCollector(
            env_workers()[0],
            policy=actor_critic_module.actor,
            frames_per_batch=max_states_per_collection,
            reset_at_each_iter=False,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
        )
    else:
        raise ValueError(
            f"Unknown collector type: {ppo_config.collector}. "
            "Use 'sync' or 'multi_sync'."
        )

    collector.set_seed(ppo_config.seed)

    loss_module = ClipPPOLoss(
        actor_network=actor_critic_module.actor,
        critic_network=actor_critic_module.critic,
        clip_epsilon=ppo_config.clip_eps,
        entropy_bonus=True,
        entropy_coef=ppo_config.ent_coef,
        critic_coef=ppo_config.val_coef,
        loss_critic_type=ppo_config.value_norm,
        clip_value=ppo_config.clip_vloss,
        normalize_advantage=ppo_config.normalize_advantage,
    )
    
    if ppo_config.advantage_type == "gae":
        loss_module.make_value_estimator(ValueEstimators.GAE)
    elif ppo_config.advantage_type == "vtrace":
        loss_module.make_value_estimator(ValueEstimators.VTrace)

    # optimizer = instantiate(opt_cfg, params=loss_module.parameters())
    # optimizer = torch.optim.AdamW(
    #     loss_module.parameters(),
    #     lr=opt_cfg.lr if opt_cfg is not None else 3e-4,
    #     eps=opt_cfg.eps if opt_cfg is not None else 1e-5,
    # )
    
    optimizer = optimizer(loss_module.parameters())
    
    print(f"Using optimizer: {optimizer}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    if lr_cfg is not None:
        lr_scheduler = instantiate(lr_cfg, optimizer=optimizer)
    else:
        lr_scheduler = None

    def update(batch, i, j, k):
        if ppo_config.sample_slices:
            batch = batch.reshape(num_slices, -1)
            # print(batch.shape)

        loss_vals = loss_module(batch)
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        optimizer.zero_grad()
        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), max_norm=ppo_config.max_grad_norm
        )

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        return loss_vals

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(
            advantage_module, mode="reduce-overhead", warmup=8
        )

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(
        ppo_config.states_per_collection, max_states_per_collection
    )

    if ppo_config.sample_slices:
        n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
    else:
        n_batch = max(1, ppo_config.graphs_per_collection // ppo_config.minibatch_size)

    training.info(
        f"Starting PPO-LSTM training with {ppo_config.num_collections} collections, "
        f"sample_slices={ppo_config.sample_slices}, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers.",
    )

    start_t = time.perf_counter()

    n_updates = 0
    n_samples = 0
    for i, tensordict_data in enumerate(collector):
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0

        training.info(
            f"Collection {i + 1}/{ppo_config.num_collections}, "
            f"Collections/s: {updates_per_second:.2f}",
        )

        tensordict_data = tensordict_data.to(
            ppo_config.update_device, non_blocking=True
        )

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            # Compute advantages
            advantage_module(tensordict_data)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        flattened_data = tensordict_data.reshape(-1)

        if ppo_config.sample_slices:
            replay_buffer.extend(flattened_data)
        else:
            replay_buffer.extend(tensordict_data)

        n_samples += flattened_data.shape[0]

        update_start_t = time.perf_counter()
        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                batch, info = replay_buffer.sample(
                    ppo_config.minibatch_size, return_info=True
                )

                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, i, j, k)

                if (
                    n_updates % logging_config.stats_interval == 0
                    and logging_config is not None
                ):
                    with torch.no_grad():
                        non_zero_rewards = flattened_data["next", "reward"]
                        improvements = flattened_data[
                            "next", "observation", "aux", "improvement"
                        ]
                        mask = improvements > -100
                        filtered_improvements = improvements[mask]

                        if filtered_improvements.numel() > 0:
                            avg_improvement = filtered_improvements.mean()
                        else:
                            avg_improvement = torch.tensor(0.0, dtype=torch.float32)

                        print(f"Average improvement: {avg_improvement.item()}")

                        if len(non_zero_rewards) > 0:
                            avg_non_zero_reward = non_zero_rewards.mean().item()
                            std_rewards = non_zero_rewards.std().item()
                        else:
                            avg_non_zero_reward = 0.0
                            std_rewards = 0.0

                        post_clip_norms = log_parameter_and_gradient_norms(loss_module)
                        wandb.log(
                            {
                                **post_clip_norms,
                                "batch/n_updates": n_updates,
                                "batch/n_collections": i + 1,
                                "batch/mean_reward": avg_non_zero_reward,
                                "batch/std_reward": std_rewards,
                                "batch/improvement": avg_improvement.item(),
                                "batch/n_samples": n_samples,
                                "batch/policy_loss": loss["loss_objective"].item(),
                                "batch/critic_loss": loss["loss_critic"].item(),
                                "batch/entropy_loss": loss["loss_entropy"].item(),
                                "batch/entropy": loss["entropy"].item(),
                                "batch/kl_approx": loss["kl_approx"].item(),
                                "batch/clip_fraction": loss["clip_fraction"].item(),
                                "batch/ESS": loss["ESS"].item(),
                                "batch/advantage_mean": tensordict_data["advantage"]
                                .mean()
                                .item(),
                                "batch/advantage_std": tensordict_data["advantage"]
                                .std()
                                .item(),
                                "batch/mean_return": tensordict_data["value_target"]
                                .mean()
                                .item(),
                                "batch/std_return": tensordict_data["value_target"]
                                .std()
                                .item(),
                                "batch/mean_improvement": filtered_improvements.mean().item(),
                                "batch/std_improvement": filtered_improvements.std().item(),
                            },
                        )

                n_updates += 1

            collector.update_policy_weights_(
                TensorDict.from_module(loss_module.actor_network).to(
                    ppo_config.collect_device
                )
            )
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")


# def run_ppo_torchrl(
#     actor_model: ProbabilisticActor,
#     critic_model: ValueOperator,
#     make_env: Callable[[], EnvBase],
# ):
#     wandb.define_metric("batch_loss/step")
#     wandb.define_metric("collect_loss/step")
#     wandb.define_metric("batch_loss/*", step_metric="batch_loss/step")
#     wandb.define_metric("grad_norm/*", step_metric="batch_loss/step")
#     wandb.define_metric("param_norm/*", step_metric="batch_loss/step")
#     wandb.define_metric("collect_loss/*", step_metric="collect_loss/step")
#     wandb.define_metric("eval/*", step_metric="eval/step")

#     # _actor_td = HeteroDataWrapper(actor_critic_base.actor)
#     # _critic_td = HeteroDataWrapper(actor_critic_base.critic)

#     _actor_td = actor_critic_base.actor
#     _critic_td = actor_critic_base.critic

#     module_action = TensorDictModule(
#         _actor_td,
#         in_keys=["observation"],
#         out_keys=["logits"],
#     )

#     td_module_action = ProbabilisticActor(
#         module_action,
#         in_keys=["logits"],
#         out_keys=["action"],
#         distribution_class=torch.distributions.Categorical,
#         cache_dist=False,
#         return_log_prob=True,
#     )

#     td_critic_module = ValueOperator(
#         module=_critic_td,
#         in_keys=["observation"],
#     )

#     td_module_action = td_module_action.to(config.collect_device)
#     td_critic_module = td_critic_module.to(config.collect_device)
#     train_actor_network = copy.deepcopy(td_module_action).to(config.update_device)
#     train_critic_network = copy.deepcopy(td_critic_module).to(config.update_device)
#     model = torch.nn.ModuleList([train_actor_network, train_critic_network])

#     # Create evaluation environment if not provided
#     if eval_env_fn is None:
#         eval_env_fn = make_env

#     if model_path:
#         model.load_state_dict(torch.load(model_path))
#         print("Loaded model from path:", model_path)

#     if do_rollout:

#         def rollout_env():
#             env = make_env()
#             env.set_policy(model[0])
#             return env

#         _make_env = rollout_env
#     else:
#         _make_env = make_env

#     collector = MultiSyncDataCollector(
#         [_make_env for _ in range(config.workers)],
#         model[0],
#         frames_per_batch=config.states_per_collection,
#         reset_at_each_iter=True,
#         cat_results=0,
#         device=config.collect_device,
#         env_device="cpu",
#     )
#     out_seed = collector.set_seed(config.seed)

#     replay_buffer = ReplayBuffer(
#         storage=LazyTensorStorage(
#             max_size=config.states_per_collection, device=config.update_device
#         ),
#         sampler=SamplerWithoutReplacement(),
#         pin_memory=torch.cuda.is_available(),
#         prefetch=4,
#         batch_size=config.minibatch_size,
#     )

#     advantage_module = GAE(
#         gamma=config.gae_gamma,
#         lmbda=config.gae_lmbda,
#         value_network=model[1],
#         average_gae=False,
#         device=config.update_device,
#         vectorized=False,
#     )

#     loss_module = ClipPPOLoss(
#         actor_network=model[0],
#         critic_network=model[1],
#         clip_epsilon=config.clip_eps,
#         entropy_bonus=True,
#         entropy_coef=config.ent_coef,
#         critic_coef=config.val_coef,
#         loss_critic_type=config.value_norm,
#         clip_value=config.clip_vloss,
#         normalize_advantage=config.normalize_advantage,
#     )

#     optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.lr)

#     def update(subdata, i, j, k):
#         loss_vals = loss_module(subdata)
#         loss_value = (
#             loss_vals["loss_objective"]
#             + loss_vals["loss_critic"]
#             + loss_vals["loss_entropy"]
#         )

#         optimizer.zero_grad()
#         loss_value.backward()

#         # Log pre-clipping gradient norms
#         pre_clip_norms = log_parameter_and_gradient_norms(loss_module)

#         grad_norm = torch.nn.utils.clip_grad_norm_(
#             loss_module.parameters(), max_norm=config.max_grad_norm
#         )

#         # Log post-clipping gradient norms
#         post_clip_norms = log_parameter_and_gradient_norms(loss_module)
#         post_clip_norms = {
#             f"post_clip_{k}": v for k, v in post_clip_norms.items() if "grad_norm" in k
#         }

#         optimizer.step()

#         # Log norms for this batch
#         step = i * config.num_epochs_per_collection * n_batches + j * n_batches + k
#         wandb.log(
#             {
#                 **pre_clip_norms,
#                 **post_clip_norms,
#                 "batch_loss/step": step,
#                 "batch_loss/objective": loss_vals["loss_objective"].item(),
#                 "batch_loss/critic": loss_vals["loss_critic"].item(),
#                 "batch_loss/entropy": loss_vals["entropy"].item(),
#                 "batch_loss/total": loss_value.item(),
#                 "batch_loss/kl_approx": loss_vals["kl_approx"].item(),
#                 "batch_loss/clip_fraction": loss_vals["clip_fraction"].item(),
#                 # "batch_loss/value_clip_fraction": loss_vals[
#                 #    "value_clip_fraction"
#                 # ].item(),
#                 "batch_loss/ESS": loss_vals["ESS"].item(),
#             },
#         )

#     # advantage_module = torch.compile(advantage_module)
#     # update = torch.compile(update, mode="reduce-overhead")

#     # Run initial evaluation
#     if config.eval_interval > 0:
#         eval_metrics = evaluate_policy(
#             policy=model[0],
#             eval_env_fn=eval_env_fn,
#             num_episodes=config.eval_episodes,
#             step=0,
#         )
#         eval_metrics["eval/step"] = 0
#         wandb.log(eval_metrics)

#     for i, tensordict_data in enumerate(collector):
#         replay_buffer.empty()

#         if (i + 1) % 20 == 0:
#             if wandb.run.dir is None:
#                 path = "."
#             else:
#                 path = wandb.run.dir
#             torch.save(
#                 model.state_dict(),
#                 os.path.join(path, model_name + f"_{i + 1}.pth"),
#             )

#         # Run evaluation at specified intervals
#         if config.eval_interval > 0 and (i + 1) % config.eval_interval == 0:
#             eval_metrics = evaluate_policy(
#                 policy=model[0],
#                 eval_env_fn=eval_env_fn,
#                 num_episodes=config.eval_episodes,
#                 step=i + 1,
#             )
#             eval_metrics["eval/step"] = i + 1
#             wandb.log(eval_metrics)

#         if i >= config.num_collections:
#             break

#         print(f"Collection: {i}")
#         tensordict_data = tensordict_data.to(config.update_device, non_blocking=True)

#         with torch.no_grad():
#             # print(f"Computing advantages for collection {i}")
#             # print("tensordict_data keys:", tensordict_data.keys(True))

#             advantage_module(tensordict_data)

#             non_zero_rewards = tensordict_data["next", "reward"]
#             improvements = tensordict_data["next", "observation", "aux", "improvement"]
#             mask = improvements > -1.5
#             filtered_improvements = improvements[mask]
#             if filtered_improvements.numel() > 0:
#                 avg_improvement = filtered_improvements.mean()

#             if len(non_zero_rewards) > 0:
#                 avg_non_zero_reward = non_zero_rewards.mean().item()
#                 std_rewards = non_zero_rewards.std().item()
#                 print(f"Average reward: {avg_non_zero_reward}")
#                 print(f"Average improvement: {avg_improvement}")

#         replay_buffer.extend(tensordict_data.reshape(-1))

#         # Training loop
#         for j in range(config.num_epochs_per_collection):
#             # n_batches = config.states_per_collection // config.minibatch_size
#             n_batches = (80 * 8) // config.minibatch_size

#             for k in range(n_batches):
#                 subdata = replay_buffer.sample(config.minibatch_size)
#                 subdata.to(config.update_device, non_blocking=True)

#                 update(subdata, i, j, k)

#         # Update the policy
#         collector.policy.load_state_dict(loss_module.actor_network.state_dict())
#         collector.update_policy_weights_(TensorDict.from_module(collector.policy))
#         wandb.log(
#             {
#                 "collect_loss/step": i,
#                 "collect_loss/mean_nonzero_reward": avg_non_zero_reward,
#                 "collect_loss/std_nonzero_reward": std_rewards,
#                 "collect_loss/average_improvement": avg_improvement.item(),
#                 "collect_loss/std_improvement": filtered_improvements.std().item(),
#                 "collect_loss/std_return": tensordict_data["value_target"].std().item(),
#                 "collect_loss/mean_return": tensordict_data["value_target"]
#                 .mean()
#                 .item(),
#                 "collect_loss/advantage_mean": tensordict_data["advantage"]
#                 .mean()
#                 .item(),
#                 "collect_loss/advantage_std": tensordict_data["advantage"].std().item(),
#             },
#         )

#     # Final evaluation
#     if config.eval_interval > 0:
#         eval_metrics = evaluate_policy(
#             policy=td_module_action,
#             eval_env_fn=eval_env_fn,
#             num_episodes=config.eval_episodes,
#             step=config.num_collections,
#         )
#         eval_metrics["eval/step"] = config.num_collections
#         wandb.log(eval_metrics)

#     # save final network
#     if wandb.run.dir is None:
#         path = "."
#     else:
#         path = wandb.run.dir
#     torch.save(
#         model.state_dict(),
#         os.path.join(path, model_name + f"_{config.num_collections}.pth"),
#     )
