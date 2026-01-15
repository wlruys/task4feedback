import glob
from pathlib import Path
from ..models import *
from ..util import *
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Literal
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import (
    SliceSampler,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage, TensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE, VTrace
from torchrl.objectives.utils import ValueEstimators
import torch.nn.functional as F
from tensordict import TensorDict
import wandb
import torch
from torchrl._utils import compile_with_warmup
from .base import AlgorithmConfig, LoggingConfig
from ..base import ActorCriticModule
from omegaconf import OmegaConf
from task4feedback.logging import training
import time
from torchrl.collectors.utils import split_trajectories
from task4feedback.ml.util import log_parameter_and_gradient_norms


def joint_stats(td, ppo):
    with torch.no_grad():
        prev_lp = td["sample_log_prob"].squeeze(-1)  # [N]
        cur_lp, dist, _ = ppo._get_cur_log_prob(td)
        cur_lp = cur_lp.squeeze(-1)  # [N]
        act = td["action"]  # [N, 64]
        logits = td["logits"]  # [N, 64, 4]

        # Recompute joint log-prob explicitly via per-head log_softmax (+ gather)
        logp_heads = F.log_softmax(logits, dim=-1)  # [N, 64, 4]
        gathered = logp_heads.gather(-1, act.unsqueeze(-1)).squeeze(-1)  # [N, 64]
        joint_lp_explicit = gathered.sum(-1)  # [N]

        print("\n=== LOG-PROB STATS ===")
        for name, x in [
            ("prev_lp (stored)", prev_lp),
            ("cur_lp (dist)", cur_lp),
            ("cur_lp (explicit)", joint_lp_explicit),
        ]:
            x = x.detach()
            print(f"{name:20s} mean={x.mean():8.3f} std={x.std():8.3f} " f"min={x.min():8.3f} max={x.max():8.3f}")

        x = td["observation", "nodes", "tasks", "attr"]  # [B, 600] ideally; if it's [600], fix your batch shaping first
        x = x.detach()
        print("\n=== OBSERVATION STATS ===")
        print(f"obs: {x}")
        print(f"obs shape: {x.shape}")
        print(f"obs mean={x.mean():8.3f} std={x.std():8.3f}")
        print(f"obs min={x.min():8.3f} max={x.max():8.3f}")
        print(f"obs numel={x.numel()} nan={torch.isnan(x).sum()} inf={torch.isinf(x).sum()}")
        print(f"obs unique={torch.unique(x)}")

        # Show me the row where the max value is
        max_val = x.max()
        max_pos = (x == max_val).nonzero(as_tuple=False)
        print(f"obs max position: {max_pos}")
        if max_pos.shape[0] < 100:
            for pos in max_pos:
                b, i, z = pos
                print(f"obs[{b}, {i}, :] = {x[b, i, :]}")

        # KL approximations
        kl_approx = prev_lp - cur_lp  # [N]
        kl_approx_explicit = prev_lp - joint_lp_explicit
        print("\n=== KL APPROX (sample-wise) ===")
        for name, x in [
            ("kl_approx", kl_approx),
            ("kl_approx_explicit", kl_approx_explicit),
        ]:
            x = x.detach()
            print(f"{name:20s} mean={x.mean():8.3f} std={x.std():8.3f} " f"min={x.min():8.3f} max={x.max():8.3f}")

        # Logit scale diagnostics
        l = logits.detach()
        print(f"\nlogits shape: {l.shape}")
        lmax = l.abs().amax().item()
        per_head_span = l.max(dim=-1).values - l.min(dim=-1).values  # [N, 64]
        print("\n=== LOGIT SCALE ===")
        print(f"|logits|_max = {lmax:.1f}; span per head: mean={per_head_span.mean():.2f} " f"max={per_head_span.max():.2f}")


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

    # --- POfD / discriminator reward shaping ---
    lambda1: float = 0.1  # shaping weight: r_hat = r_env - lambda1 * log D
    disc_lr: float = 3e-4
    disc_updates_per_collection: int = 4  # number of SGD steps per collection
    disc_minibatch_size: int = 128
    disc_eval_batch_size: int = 128  # for metrics eval
    disc_max_grad_norm: float = 0.5
    disc_eps: float = 1e-6  # clamp epsilon for log / logits
    intrinsic_clip: Optional[float] = None  # optional clip for intrinsic bonus magnitude


def should_log(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should log based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return logging_config.stats_interval > 0 and n_updates % logging_config.stats_interval == 0


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
    return logging_config.checkpoint_interval > 0 and n_updates % logging_config.checkpoint_interval == 0


def log_training_metrics(
    flattened_data: TensorDict,
    tensordict_data: TensorDict,
    loss: Dict[str, torch.Tensor],
    loss_module: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    n_updates: int,
    i: int,
    n_samples: int,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Log training metrics to wandb."""
    with torch.no_grad():
        rewards = flattened_data["next", "reward"]
        improvements = flattened_data["next", "observation", "aux", "improvement"]
        vs_policy = flattened_data["next", "observation", "aux", "vs_policy"]
        valid_improvement_mask = torch.isfinite(improvements) & (improvements > -100)
        valid_improvements = improvements[valid_improvement_mask]
        valid_quad = vs_policy[valid_improvement_mask]

        # Calculate improvement metrics
        if valid_improvements.numel() > 0:
            avg_improvement = valid_improvements.mean().item()
            max_improvement = valid_improvements.max().item()
            min_improvement = valid_improvements.min().item()

            avg_vs_policy = valid_quad.mean().item()
            max_vs_policy = valid_quad.max().item()
            min_vs_policy = valid_quad.min().item()

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

        # log shaped reward components if present
        try:
            r_env = flattened_data["next", "reward_env"]
            log_payload["shaping/reward_env_mean"] = r_env.mean().item()
        except KeyError:
            pass

        try:
            intrinsic = flattened_data["next", "intrinsic_bonus"]
            log_payload["shaping/intrinsic_mean"] = intrinsic.mean().item()
            log_payload["shaping/intrinsic_std"] = intrinsic.std().item() if intrinsic.numel() > 1 else 0.0
            log_payload["shaping/intrinsic_min"] = intrinsic.min().item()
            log_payload["shaping/intrinsic_max"] = intrinsic.max().item()
        except KeyError:
            pass

        try:
            dprob = flattened_data["disc_prob"]
            # disc_prob could be [N,1] or [N]; make it scalar stats
            dprob_flat = dprob.reshape(-1)
            log_payload["shaping/D_policy_mean"] = dprob_flat.mean().item()
            log_payload["shaping/D_policy_std"] = dprob_flat.std().item() if dprob_flat.numel() > 1 else 0.0
        except KeyError:
            pass

        # merge external metrics (disc training metrics etc.)
        if extra_metrics:
            # ensure python floats
            for k, v in extra_metrics.items():
                try:
                    log_payload[k] = float(v)
                except Exception:
                    pass

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
                    "batch/mean_vs_policy": avg_vs_policy,
                    "batch/max_vs_policy": max_vs_policy,
                    "batch/min_vs_policy": min_vs_policy,
                }
            )
            if std_improvement is not None:
                log_payload["batch/std_improvement"] = std_improvement

            training.info(f"Average training improvement: {avg_improvement}")

        wandb.log(log_payload)
        return log_payload


def _disc_prob_and_logits(discriminator, td_in: TensorDict, eps: float = 1e-6, logit_key: str = "disc_logits"):
    td_out = discriminator(td_in)
    logits = td_out[logit_key].squeeze(-1)  # (B,)
    prob = torch.sigmoid(logits).clamp(eps, 1 - eps)
    return prob, logits


def train_discriminator(
    discriminator,
    disc_optimizer: torch.optim.Optimizer,
    policy_flat: TensorDict,
    expert_demo: TensorDict,
    *,
    device: str,
    n_steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_grad_norm: float,
    eps: float,
) -> Dict[str, float]:
    """
    Train D_w to separate policy vs expert on (s,a).
    Convention here (matches reward shaping r_hat = r_env - lambda1 log D):
      - label policy = 1
      - label expert = 0
      - optimize BCE to minimize: -[E_policy log D + E_expert log(1-D)]
    """
    discriminator.train()

    # sizes
    n_pol = policy_flat.batch_size[0]
    n_exp = expert_demo.batch_size[0]

    # guard for tiny buffers
    batch_size = int(min(batch_size, n_pol, n_exp))
    eval_batch_size = int(min(eval_batch_size, n_pol, n_exp))

    if batch_size <= 0:
        return {
            "disc/loss": float("nan"),
            "disc/accuracy": float("nan"),
            "disc/acc_policy": float("nan"),
            "disc/acc_expert": float("nan"),
            "disc/policy_mean": float("nan"),
            "disc/expert_mean": float("nan"),
        }

    total_loss = 0.0
    total_acc = 0.0
    total_acc_pol = 0.0
    total_acc_exp = 0.0

    for _ in range(n_steps):
        idx_pol = torch.randint(0, n_pol, (batch_size,), device="cpu")
        idx_exp = torch.randint(0, n_exp, (batch_size,), device="cpu")

        pol_batch = policy_flat[idx_pol].select("observation", "action").to(device)
        exp_batch = expert_demo[idx_exp].select("observation", "action").to(device)

        # Forward
        pol_prob, pol_logits = _disc_prob_and_logits(discriminator, pol_batch, eps=eps)
        exp_prob, exp_logits = _disc_prob_and_logits(discriminator, exp_batch, eps=eps)

        # Targets: policy=1, expert=0
        loss_pol = F.binary_cross_entropy_with_logits(pol_logits, torch.ones_like(pol_logits))
        loss_exp = F.binary_cross_entropy_with_logits(exp_logits, torch.zeros_like(exp_logits))
        loss = loss_pol + loss_exp

        disc_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
        disc_optimizer.step()

        with torch.no_grad():
            acc_pol = (pol_prob > 0.5).float().mean().item()  # should be 1 for policy
            acc_exp = (exp_prob <= 0.5).float().mean().item()  # should be 1 for expert
            acc = 0.5 * (acc_pol + acc_exp)

        total_loss += loss.item()
        total_acc += acc
        total_acc_pol += acc_pol
        total_acc_exp += acc_exp

    # Eval on fresh minibatch for logging
    with torch.no_grad():
        idx_pol = torch.randint(0, n_pol, (eval_batch_size,), device="cpu")
        idx_exp = torch.randint(0, n_exp, (eval_batch_size,), device="cpu")
        pol_eval = policy_flat[idx_pol].select("observation", "action").to(device)
        exp_eval = expert_demo[idx_exp].select("observation", "action").to(device)

        pol_prob, _ = _disc_prob_and_logits(discriminator, pol_eval, eps=eps)
        exp_prob, _ = _disc_prob_and_logits(discriminator, exp_eval, eps=eps)

        pol_mean = pol_prob.mean().item()
        exp_mean = exp_prob.mean().item()
        pol_std = pol_prob.std().item() if pol_prob.numel() > 1 else 0.0
        exp_std = exp_prob.std().item() if exp_prob.numel() > 1 else 0.0

    steps = max(1, n_steps)
    return {
        "disc/loss": total_loss / steps,
        "disc/accuracy": total_acc / steps,
        "disc/acc_policy": total_acc_pol / steps,
        "disc/acc_expert": total_acc_exp / steps,
        "disc/policy_mean": pol_mean,
        "disc/policy_std": pol_std,
        "disc/expert_mean": exp_mean,
        "disc/expert_std": exp_std,
        "disc/prob_gap": pol_mean - exp_mean,
    }


@torch.no_grad()
def apply_ppofd_reward_shaping(
    tensordict_data: TensorDict,
    discriminator,
    *,
    device: str,
    lambda1: float,
    eps: float,
    chunk_size: int,
    intrinsic_clip: Optional[float] = None,
) -> Dict[str, float]:
    """
    Overwrites tensordict_data['next','reward'] with shaped reward:
      r_hat = r_env - lambda1 * log(D(s,a))
    Stores:
      - ['next','reward_env'] : original reward before shaping
      - ['next','intrinsic_bonus'] : lambda1 * (-log D)
      - ['disc_prob'] : D(s,a) for policy transitions

    Returns shaping stats for logging.
    """
    if lambda1 <= 0:
        return {}

    discriminator.eval()

    # reward tensor shape like [B,T,1] or [B,T]
    r_env = tensordict_data["next", "reward"]
    r_env_flat = r_env.reshape(-1)

    # Flatten transitions to feed discriminator
    flat = tensordict_data.reshape(-1)
    n = flat.batch_size[0]

    D_flat = torch.empty((n,), device=tensordict_data.device)

    # Batched inference for D(s,a)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sub = flat[start:end].select("observation", "action").to(device)
        prob, _ = _disc_prob_and_logits(discriminator, sub, eps=eps)
        D_flat[start:end] = prob

    D_clamped = D_flat.clamp(eps, 1.0 - eps)
    intrinsic_flat = (-torch.log(D_clamped)) * lambda1

    if intrinsic_clip is not None:
        intrinsic_flat = intrinsic_flat.clamp(-intrinsic_clip, intrinsic_clip)

    intrinsic = intrinsic_flat.reshape(r_env.shape)
    D_shaped = D_clamped.reshape(r_env.shape)

    # Save original and shaped rewards + diagnostic tensors
    tensordict_data["next", "reward_env"] = r_env.clone()
    tensordict_data["next", "intrinsic_bonus"] = intrinsic
    tensordict_data["disc_prob"] = D_shaped
    tensordict_data["next", "reward"] = tensordict_data["next", "reward_env"] + intrinsic

    # stats
    shaped_reward_flat = tensordict_data["next", "reward"].reshape(-1)
    return {
        "shaping/reward_env_mean": r_env_flat.mean().item(),
        "shaping/reward_shaped_mean": shaped_reward_flat.mean().item(),
        "shaping/D_policy_mean": D_clamped.mean().item(),
        "shaping/D_policy_std": (D_clamped.std().item() if D_clamped.numel() > 1 else 0.0),
        "shaping/intrinsic_mean": intrinsic_flat.mean().item(),
        "shaping/intrinsic_std": (intrinsic_flat.std().item() if intrinsic_flat.numel() > 1 else 0.0),
        "shaping/intrinsic_min": intrinsic_flat.min().item(),
        "shaping/intrinsic_max": intrinsic_flat.max().item(),
    }


def run_ppo(
    actor_critic_module: ActorCriticModule,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    eval_config: Optional[EvaluationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    seed: int = 0,
    expert_demonstration: Optional[TensorDict] = None,
):
    if logging_config is not None and (logging_frequency := logging_config.stats_interval):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")
        # discriminator / shaping
        wandb.define_metric("disc/*", step_metric="batch/n_updates")
        wandb.define_metric("shaping/*", step_metric="batch/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    eval_envs = make_eval_envs(env_constructors, eval_config)
    max_tasks = max([env.size() for env in eval_envs])
    max_candidates = max([env.simulator_factory[0].graph_spec.max_candidates for env in eval_envs])

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
            deactivate_vmap=True,
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
        batch_size=ppo_config.minibatch_size,
    )

    def env_workers():
        return [env_constructors[i % len(env_constructors)] for i in range(ppo_config.graphs_per_collection)]

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
            num_threads=ppo_config.workers,
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
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
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
        )
    else:
        raise ValueError(f"Unknown collector type: {ppo_config.collector}. " "Use 'sync' or 'multi_sync'.")

    collector.set_seed(seed)

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

    if optimizer is None:
        optimizer = torch.optim.Adam(loss_module.parameters())
    else:
        optimizer = optimizer(loss_module.parameters())
    training.info(f"Using optimizer: {optimizer}")

    # --- discriminator setup ---
    if not hasattr(actor_critic_module, "discriminator") or actor_critic_module.discriminator is None:
        raise ValueError("actor_critic_module.discriminator is required for POfD reward shaping.")

    discriminator = actor_critic_module.discriminator.to(ppo_config.update_device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=ppo_config.disc_lr)

    # Move expert demo to update_device once (optional; OK if small)
    if expert_demonstration is not None:
        expert_demonstration = expert_demonstration.to(ppo_config.update_device)

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)
        training.info(f"Using learning rate scheduler: {lr_scheduler}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    def update_policy(batch, loss_module, optimizer, ppo_config, expert_demonstration=None):
        loss_vals = loss_module(batch)
        loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

        # if expert_demonstration is not None:
        #     idx = torch.randint(0, expert_demonstration.batch_size[0], (ppo_config.minibatch_size,), device="cpu")
        #     exp_obs = expert_demonstration["observation"][idx]
        #     exp_act = expert_demonstration["action"][idx].long()

        #     td_exp = TensorDict({"observation": exp_obs}, batch_size=[ppo_config.minibatch_size]).to(loss_value.device)
        #     td_exp = loss_module.actor_network(td_exp)
        #     logits_exp = td_exp["logits"]

        #     logp = torch.nn.functional.log_softmax(logits_exp, dim=-1)
        #     loss_value = loss_value

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=ppo_config.max_grad_norm)
        optimizer.step()

        return loss_vals

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(advantage_module, mode="reduce-overhead", warmup=8)

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)
    n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
    if ppo_config.minibatch_size > states_per_collection:
        training.warning(f"Minibatch size <{ppo_config.minibatch_size}> is larger than states per collection <{states_per_collection}>. ")

    training.info(
        f"Running PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers."
    )
    eval_max_performance = 0.0
    batch_max_performance = 0.0
    if should_eval(0, eval_config):
        training.info("Running initial evaluation before training")
        metrics = run_evaluation(collector.policy, eval_envs, eval_config, 0)
        if eval_config.pickle_path is not None:
            if metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"] > eval_max_performance:
                eval_max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"]

    training.info("Starting PPO training loop")

    start_t = time.perf_counter()
    n_updates = 0
    n_samples = 0
    n_collections = 0
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

        # 1) reward preprocessing (your existing bagging)
        with torch.no_grad():
            if ppo_config.bagged_policy == "uniform":
                redistribute_rewards_uniform(tensordict_data)

        # 2) discriminator update + reward shaping (POfD)
        disc_log = {}
        if (expert_demonstration is not None) and (ppo_config.lambda1 > 0):
            policy_flat_for_disc = tensordict_data.reshape(-1).select("observation", "action")

            disc_log = train_discriminator(
                discriminator,
                disc_optimizer,
                policy_flat_for_disc,
                expert_demonstration,
                device=ppo_config.update_device,
                n_steps=ppo_config.disc_updates_per_collection,
                batch_size=ppo_config.disc_minibatch_size,
                eval_batch_size=ppo_config.disc_eval_batch_size,
                max_grad_norm=ppo_config.disc_max_grad_norm,
                eps=ppo_config.disc_eps,
            )

            shaping_log = apply_ppofd_reward_shaping(
                tensordict_data,
                discriminator,
                device=ppo_config.update_device,
                lambda1=ppo_config.lambda1,
                eps=ppo_config.disc_eps,
                chunk_size=ppo_config.disc_eval_batch_size,
                intrinsic_clip=ppo_config.intrinsic_clip,
            )
            disc_log.update(shaping_log)

        # 3) compute advantages using shaped reward
        with torch.no_grad():
            advantage_module(tensordict_data)

        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        flattened_data = tensordict_data.reshape(-1)
        samples_in_collection = flattened_data.shape[0]
        n_samples += samples_in_collection

        replay_buffer.extend(flattened_data)

        update_start_t = time.perf_counter()
        loss_module.actor_network.train()
        loss_module.critic_network.train()

        # Determine BC coefficient based on minimum improvement over baseline in the batch
        improvements = flattened_data["next", "observation", "aux", "improvement"]
        valid_improvement_mask = torch.isfinite(improvements) & (improvements > -100)
        valid_improvements = improvements[valid_improvement_mask]

        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                n_updates += 1
                batch = replay_buffer.sample(ppo_config.minibatch_size)
                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update_policy(batch, loss_module, optimizer, ppo_config, expert_demonstration=expert_demonstration)
                if should_log(n_updates, logging_config):
                    wandb_log = log_training_metrics(
                        flattened_data,
                        tensordict_data,
                        loss,
                        loss_module,
                        optimizer,
                        n_updates,
                        i,
                        n_samples,
                        extra_metrics=disc_log,
                    )
                    # Save best policy based on mean improvement of the batch
                    if logging_config.log_best_policy and round(wandb_log.get("batch/mean_improvement", -1), 2) > round(batch_max_performance, 2):
                        batch_max_performance = wandb_log["batch/mean_improvement"]
                        metrics = {}
                        # Check with evaluation envs to avoid overfitting to training envs
                        _ = evaluate_policy(n_collections, collector.policy, eval_envs, eval_config, "DETERMINISTIC", metrics)
                        if metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"] > eval_max_performance:
                            eval_max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"]
                            filename = f"{eval_max_performance:.3f}_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt"
                            checkpoint_path = os.path.join(logging_config.best_policy_dir, filename)
                            # Remove all old checkpoints with the same seed
                            pattern = os.path.join(logging_config.best_policy_dir, f"*_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt")
                            for old_file in glob.glob(pattern):
                                if os.path.abspath(old_file) != os.path.abspath(checkpoint_path):
                                    try:
                                        os.remove(old_file)
                                        training.info(f"Removed old checkpoint for seed {seed}: {old_file}")
                                    except OSError as e:
                                        training.warning(f"Failed to remove {old_file}: {e}")
                            training.info(f"New max performance: {eval_max_performance:.4f}. Saving checkpoint.")
                            if logging_config.best_policy_dir is not None:
                                save_checkpoint(
                                    n_collections,
                                    policy_module=collector.policy,
                                    value_module=loss_module.critic_network,
                                    discriminator_module=discriminator,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    filename=filename,
                                    checkpoint_dir=logging_config.best_policy_dir,
                                )
                        else:
                            training.info(
                                f"Skipping checkpoint save, eval max performance {metrics[f'eval/DETERMINISTIC']['mean_vsPolicy']:.2f} did not exceed previous best of {eval_max_performance:.2f}."
                            )
                    elif logging_config.log_best_policy and wandb_log.get("batch/mean_improvement", -1) > 0.0:
                        training.info(
                            f"Skipping env check and checkpointing, batch mean improvement {wandb_log.get('batch/mean_improvement', -1):.2f} did not exceed threshold of {batch_max_performance:.2f}."
                        )

        collector.update_policy_weights_(TensorDict.from_module(loss_module.actor_network).to(ppo_config.collect_device))
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if should_eval(n_collections, eval_config=eval_config):
            collector.policy.eval()
            metrics = run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples)
            # Save best policy based on evaluation performance
            if eval_config.pickle_path is not None:
                if metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"] > eval_max_performance:
                    eval_max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"]

                    filename = f"{eval_max_performance:.3f}_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt"
                    checkpoint_path = os.path.join(logging_config.best_policy_dir, filename)
                    # Remove all old checkpoints with the same seed
                    pattern = os.path.join(logging_config.best_policy_dir, f"*_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt")
                    for old_file in glob.glob(pattern):
                        if os.path.abspath(old_file) != os.path.abspath(checkpoint_path):
                            try:
                                os.remove(old_file)
                                training.info(f"Removed old checkpoint for seed {seed}: {old_file}")
                            except OSError as e:
                                training.warning(f"Failed to remove {old_file}: {e}")
                    training.info(f"New max performance: {eval_max_performance:.4f}. Saving checkpoint.")
                    if logging_config.best_policy_dir is not None:
                        save_checkpoint(
                            n_collections,
                            policy_module=collector.policy,
                            value_module=loss_module.critic_network,
                            discriminator_module=discriminator,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            filename=filename,
                            checkpoint_dir=logging_config.best_policy_dir,
                        )

        if should_checkpoint(n_collections, logging_config):
            training.info(f"Checkpointing at collection {n_collections}")
            save_checkpoint(
                n_collections,
                policy_module=collector.policy,
                value_module=loss_module.critic_network,
                discriminator_module=discriminator,
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
        run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples)

    save_checkpoint(n_collections, policy_module=collector.policy, value_module=loss_module.critic_network, discriminator_module=discriminator, optimizer=optimizer, lr_scheduler=lr_scheduler)

    collector.shutdown()


def _get_logits_from_tdout(td_out, key: str = "logits") -> torch.Tensor:
    """
    Extract logits from an expert/reference model output.
    Supports:
      - TensorDict with td_out[key]
      - raw tensor output
    """
    if isinstance(td_out, TensorDict):
        if key not in td_out.keys():
            raise KeyError(f"Expert model output missing key {key!r}. Keys: {list(td_out.keys(True))}")
        return td_out[key]
    if torch.is_tensor(td_out):
        return td_out
    raise TypeError(f"Expert model returned unsupported type: {type(td_out)}")


def _kl_from_logits(
    pol_logits: torch.Tensor,  # (..., H, A) for multi-head
    ref_logits: torch.Tensor,  # (..., H, A)
    *,
    reduce_heads: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """
    KL( pi_theta || pi_ref ) for categorical distributions from logits.
    For multi-head factorized policy:
      KL_total = sum_{head} KL_head
    Optionally 'mean' across heads to reduce scale sensitivity.
    Returns shape: (...)  (one KL per state/transition)
    """
    logp = F.log_softmax(pol_logits, dim=-1)  # (..., H, A)
    logq = F.log_softmax(ref_logits, dim=-1)  # (..., H, A)
    p = logp.exp()

    # KL per head: (..., H)
    kl_per_head = (p * (logp - logq)).sum(dim=-1)

    if reduce_heads == "sum":
        return kl_per_head.sum(dim=-1)  # (...,)
    elif reduce_heads == "mean":
        return kl_per_head.mean(dim=-1)  # (...,)
    else:
        raise ValueError(f"reduce_heads must be 'sum' or 'mean', got {reduce_heads!r}")


def kl_pi_ref_from_logits(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    *,
    action_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",  # "none" | "mean" | "sum"
    clip_kl: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute KL(π || π_ref) for categorical policies parameterized by logits.

    Args:
        policy_logits: (..., A) unnormalized logits from current policy πθ
        ref_logits:    (..., A) unnormalized logits from reference/expert πref
        action_mask:   optional boolean mask (..., A) where True = valid action.
                       If provided, invalid actions are masked out in BOTH dists.
        reduction:     "none" returns (...,) KL per sample;
                       "mean" returns scalar mean over all samples;
                       "sum" returns scalar sum over all samples.
        clip_kl:       optional upper bound for per-sample KL to avoid inf explosions.

    Returns:
        KL divergence per-sample or reduced scalar, as specified.
    """
    if policy_logits.shape != ref_logits.shape:
        raise ValueError(f"Shape mismatch: policy_logits {policy_logits.shape} vs ref_logits {ref_logits.shape}")

    # Flatten all leading dims so we always reduce over the last dim A
    A = policy_logits.shape[-1]
    pi_logits = policy_logits.reshape(-1, A)
    rf_logits = ref_logits.reshape(-1, A)

    m = None
    if action_mask is not None:
        m = action_mask.reshape(-1, A).to(dtype=torch.bool, device=pi_logits.device)
        # Use a very negative number instead of -inf to avoid NaNs in some edge cases
        neg = torch.finfo(pi_logits.dtype).min
        pi_logits = pi_logits.masked_fill(~m, neg)
        rf_logits = rf_logits.masked_fill(~m, neg)

    # p = softmax(pi_logits), log q = logsoftmax(ref_logits)
    p = F.softmax(pi_logits, dim=-1)
    logq = F.log_softmax(rf_logits, dim=-1)

    # F.kl_div(input=logq, target=p) computes: p * (log p - log q) elementwise
    # reduction="none" keeps shape (N, A)
    elem = F.kl_div(logq, p, reduction="none", log_target=False)
    kl_per_sample = elem.sum(dim=-1)  # (N,)

    if clip_kl is not None:
        kl_per_sample = kl_per_sample.clamp_max(float(clip_kl))

    if reduction == "none":
        return kl_per_sample
    if reduction == "mean":
        return kl_per_sample.mean()
    if reduction == "sum":
        return kl_per_sample.sum()

    raise ValueError(f"Unknown reduction={reduction!r}")


def run_ppo_action_kl(
    actor_critic_module: ActorCriticModule,
    expert_model: nn.Module,
    env_constructors: List[Callable[[], EnvBase]],
    ppo_config: PPOConfig,
    logging_config: Optional[LoggingConfig],
    eval_config: Optional[EvaluationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    seed: int = 0,
    expert_demonstration: Optional[TensorDict] = None,
):
    if logging_config is not None and (logging_frequency := logging_config.stats_interval):
        wandb.define_metric("batch/n_updates")
        wandb.define_metric("batch/n_samples", step_metric="batch/n_updates")
        wandb.define_metric("batch/n_collections", step_metric="batch/n_updates")
        wandb.define_metric("batch/*", step_metric="batch/n_updates")
        wandb.define_metric("grad_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("param_norm/*", step_metric="batch/n_updates")
        wandb.define_metric("eval/*", step_metric="batch/n_updates")

    print("Using PPO with config:", OmegaConf.to_yaml(ppo_config))

    eval_envs = make_eval_envs(env_constructors, eval_config)
    max_tasks = max([env.size() for env in eval_envs])
    max_candidates = max([env.simulator_factory[0].graph_spec.max_candidates for env in eval_envs])

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
            deactivate_vmap=True,
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
        batch_size=ppo_config.minibatch_size,
    )

    def env_workers():
        return [env_constructors[i % len(env_constructors)] for i in range(ppo_config.graphs_per_collection)]

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
            num_threads=ppo_config.workers,
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
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
            compile_policy=({"mode": "reduce-overhead"} if ppo_config.compile_policy else None),
        )
    else:
        raise ValueError(f"Unknown collector type: {ppo_config.collector}. " "Use 'sync' or 'multi_sync'.")

    collector.set_seed(seed)

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

    if optimizer is None:
        optimizer = torch.optim.Adam(loss_module.parameters())
    else:
        optimizer = optimizer(loss_module.parameters())
    training.info(f"Using optimizer: {optimizer}")

    loss_module = loss_module.to(ppo_config.update_device)
    advantage_module = advantage_module.to(ppo_config.update_device)

    def update_policy(batch, loss_module, optimizer, ppo_config, expert_model):
        loss_vals = loss_module(batch)
        loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

        # ----- KL regularization against expert/reference -----
        beta = getattr(ppo_config, "kl_beta", 0.1)  # add this to your PPOConfig
        if beta > 0.0:
            device = loss_value.device

            # Get current policy logits WITH grad
            td_pi = TensorDict({"observation": batch["observation"]}, batch_size=batch.batch_size).to(device)
            td_pi = loss_module.actor_network(td_pi)
            pi_logits = td_pi["logits"]  # shape [B,...,A]

            # Get reference logits WITHOUT grad (fixed expert)
            expert_model.eval()
            with torch.no_grad():
                td_ref = TensorDict({"observation": batch["observation"]}, batch_size=batch.batch_size).to(device)
                td_ref = expert_model(td_ref)  # assumes it returns {"logits": ...}
                ref_logits = td_ref["logits"]

            # Optional action mask (adjust key to your codebase)
            action_mask = None
            if ("observation", "action_mask") in batch.keys(True):
                action_mask = batch["observation", "action_mask"]
            elif "action_mask" in batch.keys():
                action_mask = batch["action_mask"]

            kl = kl_pi_ref_from_logits(
                pi_logits,
                ref_logits,
                action_mask=action_mask,
                reduction="mean",
                clip_kl=getattr(ppo_config, "kl_clip", None),
            )

            loss_value = loss_value + beta * kl
            loss_vals["loss_kl_ref"] = kl.detach()

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=ppo_config.max_grad_norm)
        optimizer.step()

        return loss_vals

    if ppo_config.compile_advantage:
        advantage_module = compile_with_warmup(advantage_module, mode="reduce-overhead", warmup=8)

    if ppo_config.compile_update:
        update = compile_with_warmup(update, mode="reduce-overhead", warmup=8)

    states_per_collection = min(ppo_config.states_per_collection, max_states_per_collection)
    n_batch = max(1, states_per_collection // ppo_config.minibatch_size)
    if ppo_config.minibatch_size > states_per_collection:
        training.warning(f"Minibatch size <{ppo_config.minibatch_size}> is larger than states per collection <{states_per_collection}>. ")

    training.info(
        f"Running PPO training with {ppo_config.num_collections} collections, "
        f"{max_states_per_collection} states saved per collection, "
        f"{states_per_collection} states used per collection, "
        f"{ppo_config.minibatch_size} minibatch size, "
        f"{ppo_config.epochs_per_collection} epochs per collection, "
        f"{n_batch} batches per epoch, "
        f"{ppo_config.workers} workers."
    )
    eval_max_performance = 0.0
    batch_max_performance = 0.0
    if should_eval(0, eval_config):
        training.info("Running initial evaluation before training")
        metrics = run_evaluation(collector.policy, eval_envs, eval_config, 0)
        if eval_config.pickle_path is not None:
            if metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"] > eval_max_performance:
                eval_max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"]

    training.info("Starting PPO training loop")

    start_t = time.perf_counter()
    n_updates = 0
    n_samples = 0
    n_collections = 0
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

        # 1) reward preprocessing (your existing bagging)
        with torch.no_grad():
            if ppo_config.bagged_policy == "uniform":
                redistribute_rewards_uniform(tensordict_data)

        # 3) compute advantages using shaped reward
        with torch.no_grad():
            advantage_module(tensordict_data)

        adv_end_t = time.perf_counter()
        adv_elapsed_time = adv_end_t - adv_start_t
        training.info(f"Computed advantages {i + 1} in {adv_elapsed_time:.2f} seconds")

        flattened_data = tensordict_data.reshape(-1)
        samples_in_collection = flattened_data.shape[0]
        n_samples += samples_in_collection

        replay_buffer.extend(flattened_data)

        update_start_t = time.perf_counter()
        loss_module.actor_network.train()
        loss_module.critic_network.train()

        # Determine BC coefficient based on minimum improvement over baseline in the batch
        improvements = flattened_data["next", "observation", "aux", "improvement"]
        valid_improvement_mask = torch.isfinite(improvements) & (improvements > -100)
        valid_improvements = improvements[valid_improvement_mask]

        for j in range(ppo_config.epochs_per_collection):
            for k in range(n_batch):
                n_updates += 1
                batch = replay_buffer.sample(ppo_config.minibatch_size)
                batch.to(ppo_config.update_device, non_blocking=True)
                loss = update_policy(batch, loss_module, optimizer, ppo_config, expert_model=expert_model)
                if should_log(n_updates, logging_config):
                    wandb_log = log_training_metrics(
                        flattened_data,
                        tensordict_data,
                        loss,
                        loss_module,
                        optimizer,
                        n_updates,
                        i,
                        n_samples,
                    )
                    # Save best policy based on mean improvement of the batch
                    if logging_config.log_best_policy and round(wandb_log.get("batch/mean_improvement", -1), 2) > round(batch_max_performance, 2):
                        batch_max_performance = wandb_log["batch/mean_improvement"]
                        metrics = {}
                        # Check with evaluation envs to avoid overfitting to training envs
                        _ = evaluate_policy(n_collections, collector.policy, eval_envs, eval_config, "DETERMINISTIC", metrics)
                        if metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"] > eval_max_performance:
                            eval_max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"]
                            filename = f"{eval_max_performance:.3f}_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt"
                            checkpoint_path = os.path.join(logging_config.best_policy_dir, filename)
                            # Remove all old checkpoints with the same seed
                            pattern = os.path.join(logging_config.best_policy_dir, f"*_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt")
                            for old_file in glob.glob(pattern):
                                if os.path.abspath(old_file) != os.path.abspath(checkpoint_path):
                                    try:
                                        os.remove(old_file)
                                        training.info(f"Removed old checkpoint for seed {seed}: {old_file}")
                                    except OSError as e:
                                        training.warning(f"Failed to remove {old_file}: {e}")
                            training.info(f"New max performance: {eval_max_performance:.4f}. Saving checkpoint.")
                            if logging_config.best_policy_dir is not None:
                                save_checkpoint(
                                    n_collections,
                                    policy_module=collector.policy,
                                    value_module=loss_module.critic_network,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    filename=filename,
                                    checkpoint_dir=logging_config.best_policy_dir,
                                )
                        else:
                            training.info(
                                f"Skipping checkpoint save, eval max performance {metrics[f'eval/DETERMINISTIC']['mean_vsPolicy']:.2f} did not exceed previous best of {eval_max_performance:.2f}."
                            )
                    elif logging_config.log_best_policy and wandb_log.get("batch/mean_improvement", -1) > 0.0:
                        training.info(
                            f"Skipping env check and checkpointing, batch mean improvement {wandb_log.get('batch/mean_improvement', -1):.2f} did not exceed threshold of {batch_max_performance:.2f}."
                        )

        collector.update_policy_weights_(TensorDict.from_module(loss_module.actor_network).to(ppo_config.collect_device))
        update_end_t = time.perf_counter()
        update_elapsed_time = update_end_t - update_start_t
        training.info(f"Updated policy {i + 1} in {update_elapsed_time:.2f} seconds")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if should_eval(n_collections, eval_config=eval_config):
            collector.policy.eval()
            metrics = run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples)
            # Save best policy based on evaluation performance
            if eval_config.pickle_path is not None:
                if metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"] > eval_max_performance:
                    eval_max_performance = metrics[f"eval/DETERMINISTIC"]["mean_vsPolicy"]

                    filename = f"{eval_max_performance:.3f}_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt"
                    checkpoint_path = os.path.join(logging_config.best_policy_dir, filename)
                    # Remove all old checkpoints with the same seed
                    pattern = os.path.join(logging_config.best_policy_dir, f"*_{logging_config.best_policy_name if logging_config.best_policy_name else 'checkpoint'}_{seed}.pt")
                    for old_file in glob.glob(pattern):
                        if os.path.abspath(old_file) != os.path.abspath(checkpoint_path):
                            try:
                                os.remove(old_file)
                                training.info(f"Removed old checkpoint for seed {seed}: {old_file}")
                            except OSError as e:
                                training.warning(f"Failed to remove {old_file}: {e}")
                    training.info(f"New max performance: {eval_max_performance:.4f}. Saving checkpoint.")
                    if logging_config.best_policy_dir is not None:
                        save_checkpoint(
                            n_collections,
                            policy_module=collector.policy,
                            value_module=loss_module.critic_network,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            filename=filename,
                            checkpoint_dir=logging_config.best_policy_dir,
                        )

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
        run_evaluation(collector.policy, eval_envs, eval_config, n_collections, n_updates, n_samples)

    save_checkpoint(n_collections, policy_module=collector.policy, value_module=loss_module.critic_network, optimizer=optimizer, lr_scheduler=lr_scheduler)

    collector.shutdown()
