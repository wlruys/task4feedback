from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, SliceSampler
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


class PPOAlgorithm(Algorithm):
    def __init__(self, config: PPOConfig):
        super().__init__(config)
        self.config = config
        self.advantage_module = None
        self.replay_buffer = None

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        self.advantage_module = self._build_advantage_module(model, device)
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
        if self.config.advantage_type == "gae":
            loss_module.make_value_estimator(ValueEstimators.GAE)
        elif self.config.advantage_type == "vtrace":
            loss_module.make_value_estimator(ValueEstimators.VTrace)
        return loss_module

    def make_replay_buffer(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[ReplayBuffer]:
        # PPO uses a replay buffer to store the collected batch and sample minibatches
        storage = LazyTensorStorage(max_size=max(1, self.config.states_per_collection), device=device)
        
        if self.config.sample_slices:
            sampler = SliceSampler(
                strict_length=True,
                slice_len=self.config.slice_len,
                traj_key=("collector", "traj_ids"),
            )
        else:
            sampler = SamplerWithoutReplacement()
            
        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=sampler,
            batch_size=self.config.minibatch_size,
        )
        return self.replay_buffer

    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        batch = batch.to(device, non_blocking=True)
        
        # Compute advantages
        with torch.inference_mode():
            self.advantage_module(batch)
            
        return batch

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
        
        if loss_vals["kl_approx"] > 0.8:
            training.warning(f"High KL divergence detected: {loss_vals['kl_approx'].mean().item()}")
            optimizer.zero_grad()
            return {k: v.mean().item() for k, v in loss_vals.items()}

        optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=self.config.max_grad_norm)
        optimizer.step()
        
        metrics = {k: v.mean().item() for k, v in loss_vals.items()}
        metrics.update(log_parameter_and_gradient_norms(loss_module))
        metrics["batch/lr"] = optimizer.param_groups[0]["lr"]
        
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
        
        # Process batch (compute GAE)
        if device is None:
            device = self.config.update_device
        batch = self.process_batch(batch, device)
        
        # Metrics
        flattened_data = batch.reshape(-1)
        collection_metrics = self._compute_collection_metrics(flattened_data)
        
        # Use internal buffer or passed buffer
        # For PPO, we clear the buffer each time
        rb = replay_buffer if replay_buffer is not None else self.replay_buffer
        if rb is None:
            raise ValueError("PPO requires a replay buffer for minibatching.")
             
        rb.empty()
        rb.extend(flattened_data)
        
        metrics_list = []
        buffer_len = len(rb)
        effective_batch_size = min(self.config.minibatch_size, buffer_len)
        if effective_batch_size == 0:
            return []
            
        n_batch = max(1, buffer_len // effective_batch_size)
        
        loss_module.actor_network.train()
        loss_module.critic_network.train()
        
        for _ in range(self.config.epochs_per_collection):
            for _ in range(n_batch):
                sub_batch = rb.sample(effective_batch_size)
                sub_batch = sub_batch.to(self.config.update_device, non_blocking=True)
                
                metrics = self.update(loss_module, sub_batch, optimizer)
                metrics.update(collection_metrics)
                metrics_list.append(metrics)
                
        return metrics_list

    def _build_advantage_module(self, model, device):
        critic = getattr(model, "value_operator", getattr(model, "critic", getattr(model, "value_net", getattr(model, "value", None))))
        actor = getattr(model, "policy", getattr(model, "actor", None))
        
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
            kwargs = {
                "gamma": self.config.gamma,
                "value_network": critic,
                "actor_network": actor,
                "device": device,
                "deactivate_vmap": True,
            }
            # include_vtrace_lmbda logic from original code
            return VTrace(**kwargs)
        else:
            raise ValueError(f"Unsupported advantage type: {self.config.advantage_type}")
