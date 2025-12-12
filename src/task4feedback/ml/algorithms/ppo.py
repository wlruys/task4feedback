import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
import wandb
from omegaconf import OmegaConf
from tensordict import TensorDict
import torch.nn.functional as F
from torchrl._utils import compile_with_warmup
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from torchrl.data.replay_buffers import SliceSampler, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage, TensorStorage
from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import GAE, VTrace

from task4feedback.logging import training
from task4feedback.ml.base import ActorCriticModule
from task4feedback.ml.eval import EvaluationConfig, make_eval_envs, run_evaluation
from task4feedback.ml.models import *
from task4feedback.ml.rl_utils import (
    log_parameter_and_gradient_norms,
    redistribute_rewards_uniform,
    save_checkpoint,
)
from task4feedback.utils.atomic import atomic_write_text
from .base import AlgorithmConfig, LoggingConfig


def _best_metric_path(best_dir: Path) -> Path:
    return Path(best_dir) / "best_metric.json"


def load_best_performance(best_dir: Optional[str]) -> float:
    if best_dir is None:
        return 0.0
    path = _best_metric_path(Path(best_dir))
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text())
        return float(data.get("best_mean_vs_EFT", 0.0))
    except Exception as exc:
        training.warning(f"Failed to load existing best metric from {path}: {exc}")
        return 0.0


def save_best_performance(best_dir: Optional[str], value: float, checkpoint_name: str) -> None:
    if best_dir is None:
        return
    path = _best_metric_path(Path(best_dir))
    payload = {
        "best_mean_vs_EFT": value,
        "checkpoint": checkpoint_name,
        "updated_at": time.time(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, indent=2))
    except Exception as exc:
        training.warning(f"Failed to persist best metric to {path}: {exc}")


def joint_stats(td, ppo):
    if not training.isEnabledFor(logging.DEBUG):
        return

    with torch.no_grad():
        prev_lp = td["sample_log_prob"].squeeze(-1)  # [N]
        cur_lp, _, _ = ppo._get_cur_log_prob(td)
        cur_lp = cur_lp.squeeze(-1)  # [N]
        act = td["action"]  # [N, 64]
        logits = td["logits"]  # [N, 64, 4]

        # Recompute joint log-prob explicitly via per-head log_softmax (+ gather)
        logp_heads = F.log_softmax(logits, dim=-1)  # [N, 64, 4]
        gathered = logp_heads.gather(-1, act.unsqueeze(-1)).squeeze(-1)  # [N, 64]
        joint_lp_explicit = gathered.sum(-1)  # [N]

        def _log_scalar(name, tensor):
            values = tensor.detach()
            training.debug(
                "%s mean=%8.3f std=%8.3f min=%8.3f max=%8.3f",
                name,
                values.mean().item(),
                values.std().item(),
                values.min().item(),
                values.max().item(),
            )

        training.debug("=== LOG-PROB STATS ===")
        for name, values in [
            ("prev_lp (stored)", prev_lp),
            ("cur_lp (dist)", cur_lp),
            ("cur_lp (explicit)", joint_lp_explicit),
        ]:
            _log_scalar(name, values)

        obs = td["observation", "nodes", "tasks", "attr"].detach()
        training.debug("=== OBSERVATION STATS ===")
        training.debug(
            "obs shape=%s mean=%8.3f std=%8.3f min=%8.3f max=%8.3f numel=%d nan=%d inf=%d",
            obs.shape,
            obs.mean().item(),
            obs.std().item(),
            obs.min().item(),
            obs.max().item(),
            obs.numel(),
            torch.isnan(obs).sum().item(),
            torch.isinf(obs).sum().item(),
        )
        training.debug("obs unique count=%d", torch.unique(obs).numel())

        max_val = obs.max()
        max_pos = (obs == max_val).nonzero(as_tuple=False)
        training.debug("obs max position count=%d", max_pos.shape[0])
        if 0 < max_pos.shape[0] < 100:
            for pos in max_pos:
                coords = tuple(pos.tolist())
                training.debug("obs%s = %s", coords, obs[coords])

        kl_approx = prev_lp - cur_lp  # [N]
        kl_approx_explicit = prev_lp - joint_lp_explicit
        training.debug("=== KL APPROX (sample-wise) ===")
        for name, values in [
            ("kl_approx", kl_approx),
            ("kl_approx_explicit", kl_approx_explicit),
        ]:
            _log_scalar(name, values)

        l = logits.detach()
        training.debug("logits shape=%s", tuple(l.shape))
        lmax = l.abs().amax().item()
        per_head_span = l.max(dim=-1).values - l.min(dim=-1).values  # [N, 64]
        training.debug(
            "=== LOGIT SCALE === |logits|_max=%.1f span per head mean=%.2f max=%.2f",
            lmax,
            per_head_span.mean().item(),
            per_head_span.max().item(),
        )


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
    sample_slices: bool = True  # if using lstm, whether slices are used instead of episodes
    slice_len: int = 16  # length of slices for LSTM, only used if sample_slices is True
    rollout_steps: int = 250
    advantage_type: str = "gae"  # "gae" or "vtrace"
    bagged_policy: str = "uniform"
    timeout: int = 60 * 60 * 24  # 1 day


def should_log(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should log based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.stats_interval == 0


def should_eval(
    n_updates: int,
    eval_config: Optional[EvaluationConfig],
) -> bool:
    """Check if we should evaluate based on the current update count and logging configuration."""
    if eval_config is None:
        return False
    return eval_config.eval_interval > 0 and n_updates % eval_config.eval_interval == 0


def should_checkpoint(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should checkpoint based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.checkpoint_interval == 0


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
        valid_times = flattened_data["next", "observation", "aux", "time"][valid_improvement_mask]
        valid_times = valid_times.to(torch.float32)

        # Calculate improvement metrics
        if valid_improvements.numel() > 0:
            avg_improvement = valid_improvements.mean().item()
            max_improvement = valid_improvements.max().item()
            min_improvement = valid_improvements.min().item()
            avg_time = valid_times.mean().item()
            min_time = valid_times.min().item()
            max_time = valid_times.max().item()

            if valid_improvements.numel() > 1:
                std_improvement = valid_improvements.std().item()
            else:
                std_improvement = None

        # Calculate reward metrics
        if rewards.numel() > 0:
            avg_reward = rewards.mean().item()

            if rewards.numel() > 1:
                std_reward = rewards.std().item()
            else:
                std_reward = None

        # Calculate advantage and value target metrics
        advantage_mean = tensordict_data["advantage"].mean().item()
        advantage_std = tensordict_data["advantage"].std().item()
        value_target_mean = tensordict_data["value_target"].mean().item()
        value_target_std = tensordict_data["value_target"].std().item()

        explained_variance = None
        if "state_value" in flattened_data.keys() and "value_target" in flattened_data.keys():
            values = flattened_data["state_value"].squeeze(-1)
            targets = flattened_data["value_target"].squeeze(-1)
            if targets.numel() > 1:
                var_targets = targets.var(unbiased=False)
                if torch.isfinite(var_targets) and var_targets > 1e-8:
                    residual_var = (targets - values).var(unbiased=False)
                    explained_variance = (1.0 - (residual_var / var_targets)).item()

        # Get gradient and parameter norms
        post_clip_norms = log_parameter_and_gradient_norms(loss_module)

        # Base log payload
        log_payload = {
            **post_clip_norms,
            "batch/n_updates": n_updates,
            "batch/n_collections": i + 1,
            "batch/avg_reward": avg_reward,
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

        if std_reward is not None:
            log_payload["batch/std_reward"] = std_reward

        if explained_variance is not None:
            log_payload["batch/explained_variance"] = explained_variance

        # Add improvement metrics if available
        if valid_improvements.numel() > 0:
            log_payload.update(
                {
                    "batch/mean_improvement": avg_improvement,
                    "batch/max_improvement": max_improvement,
                    "batch/min_improvement": min_improvement,
                    "batch/mean_time": avg_time,
                    "batch/max_time": max_time,
                    "batch/min_time": min_time,
                }
            )
            if std_improvement is not None:
                log_payload["batch/std_improvement"] = std_improvement

            training.info(f"Average training improvement: {avg_improvement}")

        training.info(f"Average entropy {loss['entropy'].item()}")
        wandb.log(log_payload)


def _should_eval_with_logging(
    n_collections: int,
    eval_config: Optional[EvaluationConfig],
) -> bool:
    if eval_config is None or eval_config.eval_interval <= 0:
        return False
    decision = should_eval(n_collections, eval_config)
    if decision:
        training.info(
            "Evaluation triggered at collection %d (interval=%d)",
            n_collections,
            eval_config.eval_interval,
        )
    else:
        training.debug(
            "Evaluation skipped at collection %d (interval=%d)",
            n_collections,
            eval_config.eval_interval,
        )
    return decision


def _log_collection_timings(
    n_updates: int,
    advantage_seconds: float,
    update_seconds: float,
    batch_size: int,
    logging_config: Optional[LoggingConfig],
) -> None:
    if logging_config is None:
        return
    payload = {
        "timing/advantage_seconds": advantage_seconds,
        "timing/update_seconds": update_seconds,
        "timing/collection_seconds": advantage_seconds + update_seconds,
    }
    if batch_size > 0:
        payload["timing/effective_batch_size"] = batch_size
    wandb.log(payload, step=n_updates)


def _build_advantage_module(
    actor_critic_module: ActorCriticModule,
    ppo_config: PPOConfig,
    *,
    vectorized: bool = True,
    include_vtrace_lmbda: bool = False,
) -> torch.nn.Module:
    if ppo_config.advantage_type == "gae":
        training.info("Using GAE for advantage estimation")
        return GAE(
            gamma=ppo_config.gamma,
            lmbda=ppo_config.lmbda,
            value_network=actor_critic_module.critic,
            average_gae=False,
            device=ppo_config.update_device,
            vectorized=vectorized,
            deactivate_vmap=True,
        )
    if ppo_config.advantage_type == "vtrace":
        training.info("Using VTrace for advantage estimation")
        kwargs = {
            "gamma": ppo_config.gamma,
            "value_network": actor_critic_module.critic,
            "actor_network": actor_critic_module.actor,
            "device": ppo_config.update_device,
            "deactivate_vmap": True,
        }
        if include_vtrace_lmbda:
            kwargs["lmbda"] = ppo_config.lmbda
        return VTrace(**kwargs)
    raise ValueError(f"Unsupported advantage type: {ppo_config.advantage_type}")


def _build_loss_module(
    actor_critic_module: ActorCriticModule,
    ppo_config: PPOConfig,
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
    return loss_module


def _configure_optimizer(
    loss_module: ClipPPOLoss,
    optimizer_ctor: Optional[Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]],
    lr_scheduler_ctor: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LambdaLR]],
    default_optimizer: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer],
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
    if optimizer_ctor is None:
        optimizer = default_optimizer(loss_module.parameters())
    else:
        optimizer = optimizer_ctor(loss_module.parameters())
    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def _build_replay_buffer(
    storage_size: int,
    device: str,
    minibatch_size: int,
    *,
    sample_slices: bool = False,
    slice_len: int = 0,
    traj_key: tuple[str, ...] | None = None,
) -> TensorDictReplayBuffer:
    storage = LazyTensorStorage(max_size=max(1, storage_size), device=device)
    if sample_slices:
        sampler = SliceSampler(
            strict_length=True,
            slice_len=slice_len,
            traj_key=traj_key,
        )
    else:
        sampler = SamplerWithoutReplacement()
    return TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        batch_size=minibatch_size,
    )


def _build_collector(
    env_workers_fn: Callable[[], List[Callable[[], EnvBase]]],
    actor_network: torch.nn.Module,
    ppo_config: PPOConfig,
    frames_per_batch: int,
) -> SyncDataCollector | MultiSyncDataCollector:
    env_workers = env_workers_fn()
    if not env_workers:
        raise ValueError("No environment constructors were provided to the collector factory.")
    compile_policy = {"mode": "reduce-overhead"} if ppo_config.compile_policy else None
    frames_per_batch = max(1, frames_per_batch)
    if ppo_config.collector == "multi_sync":
        training.info(
            "Creating MultiSyncDataCollector with %d workers and %d frames-per-batch",
            len(env_workers),
            frames_per_batch,
        )
        return MultiSyncDataCollector(
            env_workers,
            actor_network,
            frames_per_batch=frames_per_batch,
            cat_results="stack",
            reset_at_each_iter=False if ppo_config.rollout_steps > 0 else True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            num_threads=ppo_config.workers,
            compile_policy=compile_policy,
        )
    if ppo_config.collector == "sync":
        training.info(
            "Creating SyncDataCollector with %d workers and %d frames-per-batch",
            len(env_workers),
            frames_per_batch,
        )
        return SyncDataCollector(
            env_workers[0],
            policy=actor_network,
            frames_per_batch=frames_per_batch,
            reset_at_each_iter=True,
            policy_device=ppo_config.collect_device,
            storing_device=ppo_config.storing_device,
            env_device="cpu",
            use_buffers=True,
            compile_policy=compile_policy,
        )
    raise ValueError(f"Unknown collector type: {ppo_config.collector}. Use 'sync' or 'multi_sync'.")


def _ppo_update_step(
    batch: TensorDict,
    loss_module: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    preprocess_batch: Optional[Callable[[TensorDict], TensorDict]] = None,
) -> Dict[str, torch.Tensor]:
    if preprocess_batch is not None:
        batch = preprocess_batch(batch)

    loss_vals = loss_module(batch)
    loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
    if loss_vals["kl_approx"] > 0.8:
        training.warning(f"High KL divergence detected: {loss_vals['kl_approx'].item()}")
        training.warning("Skipping gradient update to maintain training stability.")
        optimizer.zero_grad()
        return loss_vals
    optimizer.zero_grad()
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=ppo_config.max_grad_norm)
    optimizer.step()
    return loss_vals


def run_ppo(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    eval_config: Optional[EvaluationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    seed: int = 0,
    eval_location = None,
):
    if logging_config is not None and (logging_frequency := logging_config.stats_interval):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")

    training.info("Using PPO with config:\n%s", OmegaConf.to_yaml(ppo_config))

    eval_envs = make_eval_envs(env_constructors)
    max_tasks = max([env.size() for env in eval_envs])
    max_graph_size = max_tasks 
    max_candidates = max([env.simulator_factory[0].graph_spec.max_candidates for env in eval_envs])

    if ppo_config.rollout_steps > 0:
        max_tasks = ppo_config.rollout_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks
    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)
    frames_per_batch = max(1, states_per_collection)

    advantage_module = _build_advantage_module(
        actor_critic_module,
        ppo_config,
        vectorized=(False if ppo_config.compile_advantage else True),
    )
    loss_module = _build_loss_module(actor_critic_module, ppo_config)
    optimizer, lr_scheduler = _configure_optimizer(
        loss_module,
        optimizer,
        lr_scheduler,
        default_optimizer=lambda params: torch.optim.Adam(params),
    )
    training.info(f"Using optimizer: {optimizer}")
    if lr_scheduler is not None:
        training.info(f"Using learning rate scheduler: {lr_scheduler}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    replay_buffer = _build_replay_buffer(
        frames_per_batch,
        ppo_config.update_device,
        ppo_config.minibatch_size,
    )

    def env_workers():
        return [
            env_constructors[i % len(env_constructors)]
            for i in range(ppo_config.graphs_per_collection)
        ]

    collector = _build_collector(env_workers, actor_critic_module.actor, ppo_config, frames_per_batch)
    collector.set_seed(seed)

    def update(batch):
        return _ppo_update_step(batch, loss_module, optimizer, ppo_config)

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(advantage_module, mode="reduce-overhead", warmup=8)

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    configured_n_batch = max(1, frames_per_batch // ppo_config.minibatch_size)
    if ppo_config.minibatch_size > states_per_collection:
        training.warning(f"Minibatch size <{ppo_config.minibatch_size}> is larger than states per collection <{states_per_collection}>. ")

    training.info(
        f"Running PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{configured_n_batch} batches per epoch (config-derived), "
        f"{ppo_config.workers} workers."
    )

    training.info(f"Max tasks per graph: {max_graph_size}, max candidates per task: {max_candidates}")

    max_performance = load_best_performance(logging_config.best_policy_dir if logging_config else None)
    if _should_eval_with_logging(0, eval_config):
        training.info("Running initial evaluation before training")
        metrics = run_evaluation(collector.policy, eval_envs, eval_config, 0, eval_location=eval_location)

    training.info("Starting PPO training loop")

    start_t = time.perf_counter()
    n_updates = 0
    n_samples = 0
    n_collections = 0



    #Main training loop
    for i, tensordict_data in enumerate(collector):
        n_collections += 1
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        collector.policy.eval()

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
        seconds_per_update = elapsed_time / (i + 1) if (i + 1) > 0 else 0

        training.info(f"Collection {i + 1}/{ppo_config.num_collections}, " f"Collections/s: {updates_per_second:.2f}, " f"ms/Update: {seconds_per_update * 1000:.2f}")

        tensordict_data = tensordict_data.to(ppo_config.update_device, non_blocking=True)

        adv_start_t = time.perf_counter()
        with torch.inference_mode():
            advantage_module(tensordict_data)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        flattened_data = tensordict_data.reshape(-1)
        samples_in_collection = flattened_data.shape[0]
        n_samples += samples_in_collection

        replay_buffer.extend(flattened_data)

        buffer_len = len(replay_buffer)
        effective_batch_size = min(ppo_config.minibatch_size, buffer_len) if buffer_len > 0 else 0
        if effective_batch_size == 0:
            training.warning("Replay buffer is empty after collection; skipping update.")
            continue

        n_batch = max(1, buffer_len // effective_batch_size)

        update_start_t = time.perf_counter()
        loss_module.actor_network.train()
        loss_module.critic_network.train()

        #Inner epoch / minibatch loop
        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                n_updates += 1
                batch = replay_buffer.sample(effective_batch_size)
                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch)

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

        collector.update_policy_weights_()
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        _log_collection_timings(
            n_updates,
            adv_elapsed_time,
            update_elapsed_time,
            effective_batch_size,
            logging_config,
        )
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if _should_eval_with_logging(n_collections, eval_config):
            collector.policy.eval()
            metrics = run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)
            det_metrics = metrics.get("eval/DETERMINISTIC", {})
            mean_vs_eft = det_metrics.get("mean_vs_EFT")
            if (
                mean_vs_eft is not None
                and logging_config is not None
                and logging_config.best_policy_dir is not None
            ):
                if mean_vs_eft > max_performance:
                    max_performance = mean_vs_eft
                    training.info(f"New max performance: {max_performance:.4f}. Saving checkpoint.")
                    if logging_config.best_policy_dir is not None:
                        checkpoint_name = (
                            f"{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_"
                            f"perf{int(max_performance * 1e6):09d}_"
                            f"seed{seed:03d}_"
                            f"step{n_collections:06d}.pt"
                        )
                        save_checkpoint(
                            n_collections,
                            policy_module=collector.policy,
                            value_module=loss_module.critic_network,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            filename=checkpoint_name,
                            checkpoint_dir=logging_config.best_policy_dir,
                            performance_metrics={"mean_vs_EFT": max_performance},
                        )
                        save_best_performance(logging_config.best_policy_dir, max_performance, checkpoint_name)

        if should_checkpoint(n_collections, logging_config):
            training.info(f"Checkpointing at collection {n_collections}")
            save_checkpoint(
                n_collections,
                policy_module=collector.policy,
                value_module=loss_module.critic_network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        if elapsed_time > ppo_config.timeout:
            training.warning(f"Timeout reached after {elapsed_time:.2f} seconds. Stopping training.")
            break

    if eval_config is not None and eval_config.eval_interval > 0:
        training.info("Running final evaluation after training")
        metrics = run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)
        det_metrics = metrics.get("eval/DETERMINISTIC", {})
        mean_vs_eft = det_metrics.get("mean_vs_EFT")
        if (
            mean_vs_eft is not None
            and logging_config is not None
            and logging_config.best_policy_dir is not None
            and mean_vs_eft > max_performance
        ):
            max_performance = mean_vs_eft
            training.info(f"New max performance at final eval: {max_performance:.4f}. Saving checkpoint.")
            checkpoint_name = (
                f"{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_"
                f"perf{int(max_performance * 1e6):09d}_"
                f"seed{seed:03d}_"
                f"step{n_collections:06d}.pt"
            )
            save_checkpoint(
                n_collections,
                policy_module=collector.policy,
                value_module=loss_module.critic_network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                filename=checkpoint_name,
                checkpoint_dir=logging_config.best_policy_dir,
                performance_metrics={"mean_vs_EFT": max_performance},
            )
            save_best_performance(logging_config.best_policy_dir, max_performance, checkpoint_name)

    save_checkpoint(n_collections, policy_module=collector.policy, value_module=loss_module.critic_network, optimizer=optimizer, lr_scheduler=lr_scheduler)

    collector.shutdown()


def run_ppo_lstm(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    eval_config: Optional[EvaluationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    seed: int = 0,
    eval_location = None,
):
    if logging_config is not None and (logging_frequency := logging_config.stats_interval):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")

    training.info("Using PPO with config:\n%s", OmegaConf.to_yaml(ppo_config))

    eval_envs = make_eval_envs(env_constructors)
    max_tasks = max([env.size() for env in eval_envs])
    max_candidates = max([env.simulator_factory[0].graph_spec.max_candidates for env in eval_envs])

    training.info("Max tasks in env constructors: %s", max_tasks)

    if ppo_config.rollout_steps > 0:
        max_tasks = ppo_config.rollout_steps

    max_states_per_collection = ppo_config.graphs_per_collection * max_tasks
    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)
    frames_per_batch = max(1, states_per_collection)

    advantage_module = _build_advantage_module(
        actor_critic_module,
        ppo_config,
        vectorized=True,
        include_vtrace_lmbda=True,
    )
    loss_module = _build_loss_module(actor_critic_module, ppo_config)
    optimizer, lr_scheduler = _configure_optimizer(
        loss_module,
        optimizer,
        lr_scheduler,
        default_optimizer=lambda params: torch.optim.AdamW(
            params,
            lr=3e-4,
            eps=1e-5,
        ),
    )
    training.info("Using optimizer: %s", optimizer)

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    if lr_scheduler is not None:
        training.info("Using learning rate scheduler: %s", lr_scheduler)

    if ppo_config.sample_slices:
        replay_buffer = _build_replay_buffer(
            frames_per_batch,
            ppo_config.update_device,
            ppo_config.minibatch_size,
            sample_slices=True,
            slice_len=ppo_config.slice_len,
            traj_key=("collector", "traj_ids"),
        )
        num_slices = max(1, ppo_config.minibatch_size // ppo_config.slice_len)
        def _reshape_batch(batch: TensorDict) -> TensorDict:
            return batch.reshape(num_slices, -1)

        preprocess_batch = _reshape_batch
    else:
        replay_buffer = _build_replay_buffer(
            frames_per_batch,
            ppo_config.update_device,
            ppo_config.minibatch_size,
        )
        num_slices = ppo_config.minibatch_size
        preprocess_batch = None

    def env_workers():
        return [env_constructors[i % len(env_constructors)] for i in range(ppo_config.workers)]

    collector = _build_collector(env_workers, actor_critic_module.actor, ppo_config, frames_per_batch)
    collector.set_seed(seed)

    def update(batch, i, j, k):
        return _ppo_update_step(batch, loss_module, optimizer, ppo_config, preprocess_batch)

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(advantage_module, mode="reduce-overhead", warmup=8)

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)

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

    # Initial evaluation
    max_performance = load_best_performance(logging_config.best_policy_dir if logging_config else None)
    if _should_eval_with_logging(0, eval_config):
        training.info("Running initial evaluation before training")
        metrics = run_evaluation(collector.policy, eval_envs, eval_config, 0, 0, 0, eval_location=eval_location)
        det_metrics = metrics.get("eval/DETERMINISTIC", {})
        mean_vs_eft = det_metrics.get("mean_vs_EFT")
        if (
            mean_vs_eft is not None
            and logging_config is not None
            and logging_config.best_policy_dir is not None
            and mean_vs_eft > max_performance
        ):
            max_performance = mean_vs_eft
            # Use integer-based filename for sortability (microseconds precision)
            checkpoint_name = (
                f"{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_"
                f"perf{int(max_performance * 1e6):09d}_"
                f"seed{seed:03d}_"
                f"step{0:06d}.pt"
            )
            save_checkpoint(
                0,
                policy_module=collector.policy,
                value_module=loss_module.critic_network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                filename=checkpoint_name,
                checkpoint_dir=logging_config.best_policy_dir,
                performance_metrics={"mean_vs_EFT": max_performance},
            )
            save_best_performance(logging_config.best_policy_dir, max_performance, checkpoint_name)

    start_t = time.perf_counter()

    n_updates = 0
    n_samples = 0
    n_collections = 0
    for i, tensordict_data in enumerate(collector):
        n_collections += 1
        replay_buffer.empty()

        if i >= ppo_config.num_collections:
            break

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        updates_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0

        training.info(
            f"Collection {i + 1}/{ppo_config.num_collections}, " f"Collections/s: {updates_per_second:.2f}",
        )

        tensordict_data = tensordict_data.to(ppo_config.update_device, non_blocking=True)

        adv_start_t = time.perf_counter()
        with torch.no_grad():
            advantage_module(tensordict_data)
        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        flattened_data = tensordict_data.reshape(-1)

        if ppo_config.sample_slices:
            if max_candidates > 1:
                flattened_data["advantage"] = flattened_data["advantage"].expand(-1, max_candidates)
                flattened_data["advantage"] = flattened_data["advantage"].unsqueeze(-1)

            replay_buffer.extend(flattened_data)
        else:
            if max_candidates > 1:
                tensordict_data["advantage"] = tensordict_data["advantage"].expand(-1, max_candidates)
                tensordict_data["advantage"] = tensordict_data["advantage"].unsqueeze(-1)
            replay_buffer.extend(tensordict_data)

        n_samples += flattened_data.shape[0]

        update_start_t = time.perf_counter()
        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                n_updates += 1
                batch, info = replay_buffer.sample(ppo_config.minibatch_size, return_info=True)

                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update(batch, i, j, k)

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

        collector.update_policy_weights_()
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        _log_collection_timings(
            n_updates,
            adv_elapsed_time,
            update_elapsed_time,
            ppo_config.minibatch_size,
            logging_config,
        )
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if _should_eval_with_logging(n_collections, eval_config):
            metrics = run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)
            det_metrics = metrics.get("eval/DETERMINISTIC", {})
            mean_vs_eft = det_metrics.get("mean_vs_EFT")
            if (
                mean_vs_eft is not None
                and logging_config is not None
                and logging_config.best_policy_dir is not None
            ):
                if mean_vs_eft > max_performance:
                    max_performance = mean_vs_eft
                    training.info(f"New max performance: {max_performance:.4f}. Saving checkpoint.")
                    # Use integer-based filename for sortability (microseconds precision)
                    checkpoint_name = (
                        f"{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_"
                        f"perf{int(max_performance * 1e6):09d}_"
                        f"seed{seed:03d}_"
                        f"step{n_collections:06d}.pt"
                    )
                    save_checkpoint(
                        n_collections,
                        policy_module=collector.policy,
                        value_module=loss_module.critic_network,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        filename=checkpoint_name,
                        checkpoint_dir=logging_config.best_policy_dir,
                        performance_metrics={"mean_vs_EFT": max_performance},
                    )
                    save_best_performance(logging_config.best_policy_dir, max_performance, checkpoint_name)

        if should_checkpoint(n_collections, logging_config):
            training.info(f"Checkpointing at update: {n_updates}")
            save_checkpoint(n_updates, policy_module=collector.policy, value_module=loss_module.critic_network, optimizer=optimizer, lr_scheduler=lr_scheduler)

        current_t = time.perf_counter()
        elapsed_time = current_t - start_t
        if elapsed_time > ppo_config.timeout:
            training.warning(f"Timeout reached after {elapsed_time:.2f} seconds. Stopping training.")
            break

    if eval_config is not None and eval_config.eval_interval > 0:
        training.info("Running final evaluation after training")
        run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples, eval_location=eval_location)

    save_checkpoint(
        n_collections,
        policy_module=collector.policy,
        value_module=loss_module.critic_network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    collector.shutdown()
