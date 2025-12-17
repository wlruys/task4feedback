from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import time

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import GAE, VTrace
from torchrl._utils import compile_with_warmup

from task4feedback.logging import training
from task4feedback.ml.rl_utils import log_parameter_and_gradient_norms
from ..interface import Algorithm
from .config import PPOConfig


def joint_stats(td: TensorDict, loss_module: ClipPPOLoss) -> None:
    """Debug logging for joint probability statistics from old_ppo.py"""
    if not training.isEnabledFor(logging.DEBUG):
        return

    with torch.no_grad():
        prev_lp = td["sample_log_prob"].squeeze(-1)  # [N]

        # Get current log prob using the loss module's method
        cur_lp = loss_module.actor_network(td)
        if isinstance(cur_lp, tuple):
            cur_lp = cur_lp[0]
        if hasattr(cur_lp, "log_prob"):
            cur_lp = cur_lp.log_prob(td["action"])
        cur_lp = cur_lp.squeeze(-1)  # [N]

        act = td["action"]  # [N, num_heads]

        # If we have logits, compute explicit joint log-prob
        if "logits" in td.keys():
            logits = td["logits"]  # [N, num_heads, num_actions]
            logp_heads = F.log_softmax(logits, dim=-1)  # [N, num_heads, num_actions]
            gathered = logp_heads.gather(-1, act.unsqueeze(-1)).squeeze(-1)  # [N, num_heads]
            joint_lp_explicit = gathered.sum(-1)  # [N]
        else:
            joint_lp_explicit = None

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
        _log_scalar("prev_lp (stored)", prev_lp)
        _log_scalar("cur_lp (dist)", cur_lp)
        if joint_lp_explicit is not None:
            _log_scalar("cur_lp (explicit)", joint_lp_explicit)

        # Observation stats
        if ("observation", "nodes", "tasks", "attr") in td.keys(include_nested=True):
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

        # KL approximation
        kl_approx = prev_lp - cur_lp  # [N]
        training.debug("=== KL APPROX (sample-wise) ===")
        _log_scalar("kl_approx", kl_approx)
        if joint_lp_explicit is not None:
            kl_approx_explicit = prev_lp - joint_lp_explicit
            _log_scalar("kl_approx_explicit", kl_approx_explicit)

        # Logit stats
        if "logits" in td.keys():
            l = logits.detach()
            training.debug("logits shape=%s", tuple(l.shape))
            lmax = l.abs().amax().item()
            per_head_span = l.max(dim=-1).values - l.min(dim=-1).values  # [N, num_heads]
            training.debug(
                "=== LOGIT SCALE === |logits|_max=%.1f span per head mean=%.2f max=%.2f",
                lmax,
                per_head_span.mean().item(),
                per_head_span.max().item(),
            )


class PPOAlgorithm(Algorithm):
    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self.config = config
        self.advantage_module = None
        self.value_net = None
        self.replay_buffer = None

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        # Validate model has required components
        model.validate_for_algorithm("ppo")

        # Cache value/policy references for advantage computation
        critic = model.value
        actor = model.policy
        if critic is None:
            raise ValueError("PPO initialize(): value network is required for advantage computation")

        self.value_net = critic
        # Build separate advantage module with full configuration support
        self.advantage_module = self._build_advantage_module(critic, actor, device)
        if self.config.compile_advantage:
            self.advantage_module = compile_with_warmup(self.advantage_module, mode="reduce-overhead", warmup=8)

    def make_loss_module(self, model: torch.nn.Module) -> ClipPPOLoss:
        # Validate model has required components
        model.validate_for_algorithm("ppo")

        # Direct access with standard names
        actor = model.policy
        critic = model.value

        if actor is None or critic is None:
            raise ValueError("PPO requires both policy and value networks")

        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            clip_epsilon=self.config.clip_eps,
            entropy_bonus=True,
            entropy_coeff=self.config.ent_coef,
            critic_coeff=self.config.val_coef,
            loss_critic_type=self.config.value_norm,
            clip_value=self.config.clip_vloss,
            normalize_advantage=self.config.normalize_advantage,
        )

        # Configure loss module's value estimator with consistent gamma/lambda
        # Note: We also use a separate advantage module in process_batch for additional config support
        if self.config.advantage_type == "gae":
            loss_module.make_value_estimator(
                ValueEstimators.GAE,
                gamma=self.config.gamma,
                lmbda=self.config.lmbda,
            )
        elif self.config.advantage_type == "vtrace":
            loss_module.make_value_estimator(
                ValueEstimators.VTrace,
                gamma=self.config.gamma,
            )
        else:
            raise ValueError(f"Unsupported advantage type: {self.config.advantage_type}")

        return loss_module

    def make_replay_buffer(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[ReplayBuffer]:
        # PPO uses a replay buffer to store the collected batch and sample minibatches
        storage = LazyTensorStorage(max_size=max(1, self.config.states_per_collection), device=device)
        sampler = SamplerWithoutReplacement()

        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=self.config.minibatch_size,
        )
        return self.replay_buffer

    def process_batch(self, batch: TensorDict, device: torch.device) -> Tuple[TensorDict, float]:
        """Process batch and compute advantages with timing.

        Returns:
            Tuple of (processed_batch, advantage_computation_seconds)
        """
        batch = batch.to(device, non_blocking=True)

        # Compute advantages / value targets using separate advantage module
        # (supports additional config like vectorized, average_gae, compile, etc.)
        adv_start_t = time.perf_counter()
        with torch.inference_mode():
            self.advantage_module(batch)
        adv_elapsed_time = time.perf_counter() - adv_start_t

        training.info(f"Computed advantages in {adv_elapsed_time:.2f} seconds")

        return batch, adv_elapsed_time

    def _compute_collection_metrics(self, flattened_data: TensorDict) -> Dict[str, float]:
        # training.info(f"Keys in flattened_data: {flattened_data.keys(include_nested=True)}")
        metrics = {}
        with torch.no_grad():
            rewards = flattened_data["next", "reward"]
            
            # Check for improvement/time keys which might be specific to this env
            if ("next", "observation", "aux", "improvement") in flattened_data.keys(include_nested=True):
                improvements = flattened_data["next", "observation", "aux", "improvement"]
                valid_improvement_mask = torch.isfinite(improvements) & (improvements > -100)
                valid_improvements = improvements[valid_improvement_mask]
                
                if valid_improvements.numel() > 0:
                    metrics["batch/avg_improvement"] = valid_improvements.mean().item()
                    metrics["batch/max_improvement"] = valid_improvements.max().item()
                    metrics["batch/min_improvement"] = valid_improvements.min().item()
                    if valid_improvements.numel() > 1:
                        metrics["batch/std_improvement"] = valid_improvements.std().item()

                if ("next", "observation", "aux", "time") in flattened_data.keys(include_nested=True):
                    times = flattened_data["next", "observation", "aux", "time"]
                    valid_times = times[valid_improvement_mask].to(torch.float32)
                    if valid_times.numel() > 0:
                        metrics["batch/avg_time"] = valid_times.mean().item()
                        metrics["batch/min_time"] = valid_times.min().item()
                        metrics["batch/max_time"] = valid_times.max().item()

            # Reward metrics
            if rewards.numel() > 0:
                metrics["batch/avg_reward"] = rewards.mean().item()
                if rewards.numel() > 1:
                    metrics["batch/std_reward"] = rewards.std().item()

            # Advantage/Value metrics
            if "advantage" in flattened_data.keys():
                metrics["batch/advantage_mean"] = flattened_data["advantage"].mean().item()
                metrics["batch/advantage_std"] = flattened_data["advantage"].std().item()
            
            if "value_target" in flattened_data.keys():
                metrics["batch/mean_value_target"] = flattened_data["value_target"].mean().item()
                metrics["batch/std_value_target"] = flattened_data["value_target"].std().item()

            # Explained variance
            if "state_value" in flattened_data.keys() and "value_target" in flattened_data.keys():
                values = flattened_data["state_value"].squeeze(-1)
                targets = flattened_data["value_target"].squeeze(-1)
                if targets.numel() > 1:
                    var_targets = targets.var(unbiased=False)
                    if torch.isfinite(var_targets) and var_targets > 1e-8:
                        residual_var = (targets - values).var(unbiased=False)
                        metrics["batch/explained_variance"] = (1.0 - (residual_var / var_targets)).item()
        
        return metrics

    def update(
        self,
        loss_module: LossModule,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        # Single update step (minibatch)
        loss_vals = loss_module(batch)
        loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

        # KL early stopping per minibatch (from old_ppo.py)
        # Skip gradient update if KL divergence is too high to maintain training stability
        if loss_vals["kl_approx"] > 0.8:
            training.warning(f"High KL divergence detected: {loss_vals['kl_approx'].item():.4f}")
            training.warning("Skipping gradient update to maintain training stability.")
            optimizer.zero_grad()
            metrics = {k: v.mean().item() for k, v in loss_vals.items()}
            metrics["batch/lr"] = optimizer.param_groups[0]["lr"]
            metrics["ppo/kl_skipped"] = 1.0
            return metrics

        optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=self.config.max_grad_norm)
        optimizer.step()

        metrics = {k: v.mean().item() for k, v in loss_vals.items()}
        if getattr(self.config, "log_norms", False):
            metrics.update(log_parameter_and_gradient_norms(loss_module))
        metrics["batch/lr"] = optimizer.param_groups[0]["lr"]
        metrics["ppo/kl_skipped"] = 0.0

        return metrics

    def train_step(
        self,
        batch: TensorDict,
        loss_module: LossModule,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Optional[ReplayBuffer] = None,
        target_net_updater: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, float]]:

        # Process batch (compute GAE) with timing
        if device is None:
            device = self.config.update_device
        batch, advantage_seconds = self.process_batch(batch, device)

        # Compute collection metrics once (will be added to first update only)
        flattened_data = batch.reshape(-1)
        collection_metrics = self._compute_collection_metrics(flattened_data)

        # Use internal replay buffer for efficient minibatch sampling
        rb = replay_buffer if replay_buffer is not None else self.replay_buffer
        if rb is None:
            raise ValueError("PPO requires a replay buffer for efficient minibatching.")

        # Add flattened data to replay buffer
        rb.extend(flattened_data)

        metrics_list = []
        buffer_len = len(rb)
        effective_batch_size = min(self.config.minibatch_size, buffer_len)
        if effective_batch_size == 0:
            rb.empty()  # Clear buffer for next collection
            return []

        loss_module.actor_network.train()
        loss_module.critic_network.train()

        # Calculate number of batches per epoch (from old_ppo.py)
        n_batch = max(1, buffer_len // effective_batch_size)

        # Track update timing (from old_ppo.py)
        update_start_t = time.perf_counter()

        # Epoch-level training with KL early stopping
        for epoch in range(self.config.epochs_per_collection):
            epoch_kl_sum = 0.0
            epoch_updates = 0

            # Sample minibatches from replay buffer (explicit sampling like old_ppo.py)
            for _ in range(n_batch):
                sample = rb.sample(effective_batch_size)
                sample = sample.to(device, non_blocking=True)

                # Optional debug logging (from old_ppo.py)
                if training.isEnabledFor(logging.DEBUG):
                    joint_stats(sample, loss_module)

                metrics = self.update(loss_module, sample, optimizer)

                # Track KL for epoch-level early stopping
                epoch_kl_sum += metrics.get("kl_approx", 0.0)
                epoch_updates += 1

                metrics_list.append(metrics)

            # Check for epoch-level KL early stopping
            if self.config.target_kl is not None and epoch_updates > 0:
                avg_kl = epoch_kl_sum / epoch_updates
                if avg_kl > self.config.target_kl:
                    training.info(
                        f"Early stopping at epoch {epoch + 1}/{self.config.epochs_per_collection} "
                        f"due to KL divergence {avg_kl:.4f} > {self.config.target_kl:.4f}"
                    )
                    # Add early stop metric to last update
                    if metrics_list:
                        metrics_list[-1]["ppo/early_stopped_epoch"] = epoch + 1
                        metrics_list[-1]["ppo/final_kl"] = avg_kl
                    break

        update_elapsed_time = time.perf_counter() - update_start_t
        training.info(f"Updated policy in {update_elapsed_time:.2f} seconds")

        # Add collection metrics and timing to the first update only
        # This avoids bloating logs with duplicate collection-level statistics
        if metrics_list:
            collection_metrics.update({
                "timing/advantage_seconds": advantage_seconds,
                "timing/update_seconds": update_elapsed_time,
                "timing/collection_seconds": advantage_seconds + update_elapsed_time,
                "timing/effective_batch_size": effective_batch_size,
            })
            metrics_list[0].update(collection_metrics)

        # Clear buffer for next collection (PPO is on-policy)
        rb.empty()

        return metrics_list

    def _build_advantage_module(self, critic, actor, device):
        """Build advantage module with full configuration support.

        This is separate from the loss module's value estimator to support
        additional arguments like average_gae, vectorized, and compilation.
        Both use the same gamma/lambda from config for consistency.
        """
        if self.config.advantage_type == "gae":
            return GAE(
                gamma=self.config.gamma,
                lmbda=self.config.lmbda,
                value_network=critic,
                average_gae=False,
                device=device,
                vectorized=not self.config.compile_advantage,
                deactivate_vmap=True,
            )
        elif self.config.advantage_type == "vtrace":
            # VTrace with optional lambda support (from old_ppo.py)
            vtrace_kwargs = {
                "gamma": self.config.gamma,
                "value_network": critic,
                "actor_network": actor,
                "device": device,
                "deactivate_vmap": True,
            }
            # Add lambda if configured (for LSTM compatibility from old implementation)
            if hasattr(self.config, "vtrace_use_lambda") and self.config.vtrace_use_lambda:
                vtrace_kwargs["lmbda"] = self.config.lmbda
            return VTrace(**vtrace_kwargs)
        else:
            raise ValueError(f"Unsupported advantage type: {self.config.advantage_type}")
