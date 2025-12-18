import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import ValueEstimators
from torchrl.objectives.value import GAE, VTrace

from task4feedback.logging import training
from task4feedback.ml.rl_utils import log_parameter_and_gradient_norms
from ..interface import Algorithm
from ..compile_utils import _safe_compile_with_warmup
from .config import PPOConfig




class PPOAlgorithm(Algorithm):
    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self.config = config
        self.advantage_module = None
        self.value_net = None
        self.replay_buffer = None
        self._update_core = None

    def _compute_states_per_collection(self) -> int:
        """Compute effective states_per_collection from rollout_steps and graphs_per_collection."""
        if not hasattr(self.config, 'rollout_steps') or self.config.rollout_steps <= 0:
            raise ValueError("rollout_steps must be set and > 0 in config")
        return self.config.graphs_per_collection * self.config.rollout_steps

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        model.validate_for_algorithm("ppo")

        self.value_net = model.value
        if self.value_net is None:
            raise ValueError("PPO requires value network for advantage computation")

        self.advantage_module = self._build_advantage_module(model.value, model.policy, device)
        if self.config.compile_advantage:
            self.advantage_module = _safe_compile_with_warmup(
                self.advantage_module,
                warmup=8,
                mode="default",
                label="ppo/advantage_module",
            )
        if getattr(self.config, "compile_update", False):
            self._update_core = self._make_update_core()
            self._update_core = _safe_compile_with_warmup(
                self._update_core,
                warmup=8,
                mode="default",
                label="ppo/update_core",
            )
        else:
            self._update_core = self._make_update_core()

    @staticmethod
    def _make_update_core():
        def _core(loss_module: LossModule, batch: TensorDict):
            loss_vals = loss_module(batch)
            loss_value = (
                loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
            )
            skipped = loss_vals["kl_approx"].gt(0.8).any()
            loss_value = torch.where(skipped, torch.zeros_like(loss_value), loss_value)
            loss_value.backward()
            return loss_vals, skipped

        return _core

    def make_loss_module(self, model: torch.nn.Module) -> ClipPPOLoss:
        model.validate_for_algorithm("ppo")

        if model.policy is None or model.value is None:
            raise ValueError("PPO requires both policy and value networks")

        actor_network = model.policy
        critic_network = model.value
        if getattr(self.config, "compile_loss_networks", False):
            #actor_network = _compile_with_warmup(actor_network, warmup=8, mode="default")
            #critic_network = _safe_compile_with_warmup(critic_network, warmup=8, mode="default")
            actor_network = torch.compile(actor_network, mode="default")
            critic_network = torch.compile(critic_network, mode="default")

        loss_module = ClipPPOLoss(
            actor_network=actor_network,
            critic_network=critic_network,
            clip_epsilon=self.config.clip_eps,
            entropy_bonus=True,
            entropy_coeff=self.config.ent_coef,
            critic_coeff=self.config.val_coef,
            loss_critic_type=self.config.value_norm,
            clip_value=self.config.clip_vloss,
            normalize_advantage=self.config.normalize_advantage,
        )

        estimator_type = (
            ValueEstimators.GAE if self.config.advantage_type == "gae" else ValueEstimators.VTrace
        )
        loss_module.make_value_estimator(
            estimator_type,
            gamma=self.config.gamma,
            lmbda=self.config.lmbda if estimator_type == ValueEstimators.GAE else None,
        )

        return loss_module
    
    def _get_loss_metrics(self, loss_out):
        #Put batch in front of all keys in loss_out
        return {
            f"batch/{k}": v.item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_out.items()
        }
    
    def make_replay_buffer(self, batch_size: int, device: torch.device) -> ReplayBuffer:
        # Use computed states_per_collection based on rollout_steps
        states_per_collection = self._compute_states_per_collection()

        storage = LazyTensorStorage(
            max_size=max(1, states_per_collection), device=device
        )
        sampler = SamplerWithoutReplacement()

        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=self.config.minibatch_size,
        )
        return self.replay_buffer

    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        batch = batch.to(device, non_blocking=True)

        start_time = time.perf_counter()
        with torch.inference_mode():
            self.advantage_module(batch)
        elapsed = time.perf_counter() - start_time

        training.info(f"Computed advantages in {elapsed:.2f} seconds")
        return batch

    def update(
        self,
        loss_module: LossModule,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        optimizer.zero_grad()
        if self._update_core is None:
            self._update_core = self._make_update_core()

        loss_vals, skipped = self._update_core(loss_module, batch)
        if bool(skipped.detach().item()):
            training.warning(
                f"High KL divergence {loss_vals['kl_approx'].item():.4f}, skipping update"
            )
            with torch.no_grad():
                metrics = {k: v.mean().item() for k, v in loss_vals.items()}
                metrics["batch/lr"] = optimizer.param_groups[0]["lr"]
                metrics["batch/kl_skipped"] = 1.0
            optimizer.zero_grad()
            return metrics

        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=self.config.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            metrics = {k: v.mean().item() for k, v in loss_vals.items()}
            if getattr(self.config, "log_norms", False):
                metrics.update(log_parameter_and_gradient_norms(loss_module))
            metrics["batch/lr"] = optimizer.param_groups[0]["lr"]
            metrics["batch/kl_skipped"] = 0.0
        return metrics
    

    def train_step(
        self,
        batch: TensorDict,
        loss_module: LossModule,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Optional[ReplayBuffer] = None,
        target_net_updater: Optional[Any] = None,
        device: Optional[torch.device] = None,
        n_collections: int = 0,
        n_updates: int = 0,
        n_samples: int = 0,
    ) -> List[Dict[str, float]]:
        if device is None:
            device = self.config.update_device

        batch = self.process_batch(batch, device)
        flattened_data = batch.reshape(-1)
        collection_metrics = self._compute_collection_metrics(flattened_data)

        rb = replay_buffer if replay_buffer is not None else self.replay_buffer
        if rb is None:
            raise ValueError("PPO requires a replay buffer")

        rb.extend(flattened_data)

        buffer_len = len(rb)
        effective_batch_size = min(self.config.minibatch_size, buffer_len)
        if effective_batch_size == 0:
            rb.empty()
            return []

        loss_module.actor_network.train()
        loss_module.critic_network.train()

        n_batch = max(1, buffer_len // effective_batch_size)
        metrics_list = []
        update_start = time.perf_counter()

        for epoch in range(self.config.epochs_per_collection):
            epoch_updates = 0

            for _ in range(n_batch):
                sample = rb.sample(effective_batch_size).to(device, non_blocking=True)
                loss_metrics = self.update(loss_module, sample, optimizer)
                metrics = self._get_loss_metrics(loss_metrics)
                metrics_list.append(metrics)
                epoch_updates += 1

        update_elapsed = time.perf_counter() - update_start
        training.info(f"Updated policy in {update_elapsed:.2f} seconds")

        if metrics_list:
            shared_metrics = {
                **collection_metrics,
                "timing/update_seconds": update_elapsed,
            }
            if effective_batch_size > 0:
                shared_metrics["batch/effective_batch_size"] = effective_batch_size
            for metrics in metrics_list:
                metrics.update(shared_metrics)

        rb.empty()
        return metrics_list

    def _build_advantage_module(self, critic, actor, device):
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
            vtrace_kwargs = {
                "gamma": self.config.gamma,
                "value_network": critic,
                "actor_network": actor,
                "device": device,
                "deactivate_vmap": True,
            }
            if hasattr(self.config, "vtrace_use_lambda") and self.config.vtrace_use_lambda:
                vtrace_kwargs["lmbda"] = self.config.lmbda
            return VTrace(**vtrace_kwargs)
        else:
            raise ValueError(f"Unsupported advantage type: {self.config.advantage_type}")

    def _compute_collection_metrics(self, flattened_data: TensorDict) -> Dict[str, float]:
        metrics = {}
        with torch.no_grad():
            rewards = flattened_data["next", "reward"]

            if ("next", "observation", "aux", "improvement") in flattened_data.keys(
                include_nested=True
            ):
                improvements = flattened_data["next", "observation", "aux", "improvement"]
                valid_mask = torch.isfinite(improvements) & (improvements > -100)
                valid_improvements = improvements[valid_mask]

                if valid_improvements.numel() > 0:
                    metrics["batch/mean_improvement"] = valid_improvements.mean().item()
                    metrics["batch/max_improvement"] = valid_improvements.max().item()
                    metrics["batch/min_improvement"] = valid_improvements.min().item()
                    if valid_improvements.numel() > 1:
                        metrics["batch/std_improvement"] = valid_improvements.std().item()
                    metrics["batch/n_completed"] = float(valid_improvements.numel())

                    # Only log times for completed episodes
                    if ("next", "observation", "aux", "time") in flattened_data.keys(
                        include_nested=True
                    ):
                        times = flattened_data["next", "observation", "aux", "time"]
                        valid_times = times[valid_mask].to(torch.float32)
                        if valid_times.numel() > 0:
                            metrics["batch/mean_time"] = valid_times.mean().item()
                            metrics["batch/min_time"] = valid_times.min().item()
                            metrics["batch/max_time"] = valid_times.max().item()
                else:
                    # No episodes finished in this collection
                    training.debug(
                        "No episodes completed in collection batch (total steps: %d). "
                        "Consider increasing frames_per_batch if this persists.",
                        improvements.numel()
                    )
                    metrics["batch/n_completed"] = 0.0

            if rewards.numel() > 0:
                metrics["batch/mean_reward"] = rewards.mean().item()
                if rewards.numel() > 1:
                    metrics["batch/std_reward"] = rewards.std().item()

            if "advantage" in flattened_data.keys():
                metrics["batch/mean_advantage"] = flattened_data["advantage"].mean().item()
                metrics["batch/std_advantage"] = flattened_data["advantage"].std().item()

            if "value_target" in flattened_data.keys():
                metrics["batch/mean_value_target"] = flattened_data["value_target"].mean().item()
                metrics["batch/std_value_target"] = flattened_data["value_target"].std().item()

            if "state_value" in flattened_data.keys() and "value_target" in flattened_data.keys():
                values = flattened_data["state_value"].squeeze(-1)
                targets = flattened_data["value_target"].squeeze(-1)
                if targets.numel() > 1:
                    var_targets = targets.var(unbiased=False)
                    if torch.isfinite(var_targets) and var_targets > 1e-8:
                        residual_var = (targets - values).var(unbiased=False)
                        metrics["batch/explained_variance"] = (
                            1.0 - (residual_var / var_targets)
                        ).item()
        return metrics
