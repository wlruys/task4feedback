"""QMIX Algorithm implementation following TorchRL best practices.

Based on:
- https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/qmix_vdn.py
- https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage, RandomSampler
from torchrl.envs import EnvBase
from torchrl.objectives import DQNLoss, ValueEstimators
from torchrl.objectives.common import LossModule
from torchrl.modules import EGreedyModule, QValueModule
import tensordict.nn as td_nn

from task4feedback.logging import training
from task4feedback.ml.rl_utils import log_parameter_and_gradient_norms
from ..compile_utils import _safe_compile_with_warmup
from ..interface import Algorithm
from .config import QMIXConfig


class _QValueGreedyPolicy(td_nn.TensorDictModuleBase):
    """Wrap a Q-value network with greedy action selection."""

    def __init__(
        self,
        qvalue_net: torch.nn.Module,
        action_space: str,
        action_spec,
    ) -> None:
        super().__init__()
        self.qvalue_net = qvalue_net
        self.qvalue_module = QValueModule(
            action_space=action_space,
            action_value_key="action_value",
            out_keys=["action", "action_value", "chosen_action_value"],
            spec=action_spec,
        )
        self.in_keys = self.qvalue_net.in_keys
        self.out_keys = self.qvalue_module.out_keys

    def forward(self, td: TensorDict) -> TensorDict:
        td = self.qvalue_net(td)
        td = self.qvalue_module(td)
        return td


class QMIXAlgorithm(Algorithm):
    """QMIX multi-agent RL algorithm implementation.

    Currently implements a simplified version using DQNLoss with multi-agent key mapping,
    leveraging the environment's multi-head structure.
    """

    def __init__(self, config: QMIXConfig):
        super().__init__(config)
        self.config = config
        self._exploration_module: Optional[EGreedyModule] = None
        self._update_core = None
        self.replay_buffer = None
        self._action_spec = None
        self._action_mask_key: tuple[str, ...] = ("observation", "aux", "candidate_mask")
        self._greedy_value_network: Optional[_QValueGreedyPolicy] = None

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        """Initialize QMIX-specific components."""
        model.validate_for_algorithm("qmix")

        # Grab action spec from env for exploration and greedy argmax
        if env_constructors:
            try:
                probe_env = env_constructors[0]()
                self._action_spec = probe_env.action_spec
                # The number of agents is the number of heads in the action spec
                if self._action_spec.shape:
                    self.n_agents = self._action_spec.shape[0]
                else:
                    self.n_agents = 1
                probe_env.close()
            except Exception as exc:
                training.warning("Failed to inspect env for action spec: %s", exc)

        # Optionally compile the update core for performance
        if getattr(self.config, "compile_update", False):
            self._update_core = _safe_compile_with_warmup(
                self._make_update_core(),
                warmup=8,
                mode="default",
                label="qmix/update_core",
            )
        else:
            self._update_core = self._make_update_core()

    def _get_greedy_value_network(self, model: torch.nn.Module) -> _QValueGreedyPolicy:
        if self._greedy_value_network is None:
            if model.qvalue is None:
                raise ValueError("QMIX requires a qvalue network for greedy policy construction.")
            self._greedy_value_network = _QValueGreedyPolicy(
                model.qvalue,
                action_space=self.config.action_space,
                action_spec=self._action_spec,
            )
        return self._greedy_value_network

    @staticmethod
    def _make_update_core():
        """Create the core update function for computing and backpropagating loss."""
        def _core(loss_module: LossModule, batch: TensorDict):
            loss_td = loss_module(batch)
            # Sum all loss components
            loss_sum = sum(
                loss for key, loss in loss_td.items()
                if key.startswith("loss")
            )
            loss_sum.backward()
            return loss_td

        return _core

    def make_loss_module(self, model: torch.nn.Module) -> DQNLoss:
        """Create QMIX loss module with proper key configuration.

        Args:
            model: The model containing the Q-value network

        Returns:
            Configured DQNLoss module (acting as QMIX/IQL)
        """
        model.validate_for_algorithm("qmix")

        qvalue = model.qvalue
        if qvalue is None:
            raise ValueError("QMIX requires qvalue network")

        # Optionally compile the Q-network for performance
        if getattr(self.config, "compile_loss_networks", False):
            qvalue = _safe_compile_with_warmup(
                qvalue,
                warmup=8,
                mode="default",
                label="qmix/qvalue",
            )
            model.qvalue = qvalue

        qvalue_with_greedy = self._get_greedy_value_network(model)

        # Create DQN loss with target network (delay_value=True)
        # For QMIX, we use DQNLoss with multi-agent key mapping
        loss_module = DQNLoss(
            value_network=qvalue_with_greedy,
            loss_function=self.config.loss_function,
            delay_value=True,  # Use separate target network
            action_space=self.config.action_space,
        )

        # Set keys for multi-agent structure
        # These keys match the environment's multi-agent output format
        # We set them BEFORE making the value estimator to avoid KeyError
        loss_module.set_keys(
            action_value="action_value",
            value="chosen_action_value",
            action="action",
            reward="reward",
            done="done",
            terminated="terminated",
        )

        # Configure the value estimator (TD0 for QMIX)
        loss_module.make_value_estimator(
            ValueEstimators.TD0,
            gamma=self.config.gamma
        )

        return loss_module

    def make_replay_buffer(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[ReplayBuffer]:
        """Create replay buffer for off-policy learning.

        Args:
            batch_size: Size of batches to sample
            device: Device for storing replay buffer

        Returns:
            Configured replay buffer
        """
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                self.config.replay_buffer_size,
                device=device
            ),
            sampler=RandomSampler(),
            batch_size=batch_size,
        )
        return self.replay_buffer

    def get_collection_policy(
        self,
        model: torch.nn.Module,
        device: torch.device,  # noqa: ARG002
        mode: str = "train",
    ) -> torch.nn.Module:
        """Get the policy for data collection with epsilon-greedy exploration.

        Args:
            model: The model containing the Q-value network
            device: Device to run the policy on
            mode: "train" or "eval" mode

        Returns:
            Policy module for collection
        """
        if model.qvalue is None:
            raise ValueError("QMIX collector requires a qvalue network.")

        # Create exploration module if not exists
        if self._exploration_module is None and mode == "train":
            self._exploration_module = EGreedyModule(
                eps_init=self.config.eps_init,
                eps_end=self.config.eps_end,
                annealing_num_steps=self.config.eps_decay,
                action_key="action",
                spec=None,
            )

        # For training, return Q-network
        if mode == "train" and self._exploration_module is not None:
            policy = model.qvalue
            self._exploration_module.train()
            return policy
        else:
            # For evaluation, use greedy policy
            policy = model.qvalue
            policy.eval()
            return policy

    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        """Process batch before training.

        Args:
            batch: Raw batch from replay buffer
            device: Device to move batch to

        Returns:
            Processed batch ready for loss computation
        """
        # Move to device
        batch = batch.to(device, non_blocking=True)

        # Expand rewards, done, terminated, and action to match the number of agents
        num_heads = self.n_agents
        B = batch.shape[0]

        # Expand observation and next observation to (B, N, ...)
        # This ensures the model sees an agent dimension and produces (B, N, C) Q-values
        if "observation" in batch.keys():
            obs = batch["observation"]
            if obs.batch_dims == 1:
                batch["observation"] = obs.unsqueeze(1).expand(B, num_heads).to_tensordict()
            elif obs.batch_dims == 2 and obs.shape[1] == 1:
                batch["observation"] = obs.expand(B, num_heads).to_tensordict()
        
        if ("next", "observation") in batch.keys(include_nested=True):
            next_obs = batch["next", "observation"]
            if next_obs.batch_dims == 1:
                batch["next", "observation"] = next_obs.unsqueeze(1).expand(B, num_heads).to_tensordict()
            elif next_obs.batch_dims == 2 and next_obs.shape[1] == 1:
                batch["next", "observation"] = next_obs.expand(B, num_heads).to_tensordict()

        for key in ["reward", "done", "terminated", "action"]:
            for prefix in [("next",), ()]:
                full_key = prefix + (key,)
                val = batch.get(full_key, None)
                if val is not None:
                    if val.ndim == 1:
                        # [batch] -> [batch, num_heads]
                        batch[full_key] = val.unsqueeze(1).expand(B, num_heads)
                    elif val.ndim == 2:
                        if val.shape[1] == 1:
                            # [batch, 1] -> [batch, num_heads]
                            batch[full_key] = val.expand(B, num_heads)
                        elif val.shape[1] == num_heads:
                            # Already has num_heads, just ensure it's 2D
                            batch[full_key] = val
                        else:
                            # Truncate or expand to match num_heads
                            if val.shape[1] > num_heads:
                                batch[full_key] = val[:, :num_heads]
                            else:
                                batch[full_key] = val.expand(B, num_heads)
                    elif val.ndim == 3 and val.shape[1] == 1:
                        # [batch, 1, 1] -> [batch, num_heads]
                        batch[full_key] = val.squeeze(-1).expand(B, num_heads)
                    elif val.ndim == 3 and val.shape[2] == 1:
                        # [batch, num_heads, 1] -> [batch, num_heads]
                        batch[full_key] = val.squeeze(-1)

        return batch

    def update(
        self,
        loss_module: DQNLoss,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Perform a single gradient update.

        Args:
            loss_module: QMIX loss module
            batch: Batch of transitions
            optimizer: Optimizer
            target_net_updater: Target network updater

        Returns:
            Dictionary of training metrics
        """
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Compute loss and backpropagate
        if self._update_core is None:
            self._update_core = self._make_update_core()

        loss_td = self._update_core(loss_module, batch)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(),
            self.config.max_grad_norm
        )

        # Update parameters
        optimizer.step()

        # Update target network if provided
        if target_net_updater is not None:
            target_net_updater.step()

        # Collect metrics
        metrics = self._get_loss_metrics(loss_td)
        metrics["batch/lr"] = optimizer.param_groups[0]["lr"]

        # Add epsilon to metrics if available
        if self._exploration_module is not None:
            metrics["batch/epsilon"] = float(self._exploration_module.eps)

        # Optionally log parameter and gradient norms
        if getattr(self.config, "log_norms", False):
            metrics.update(log_parameter_and_gradient_norms(loss_module))

        return metrics

    @staticmethod
    def _get_loss_metrics(loss_out: TensorDict) -> Dict[str, float]:
        """Extract metrics from loss output."""
        metrics = {}
        for k, v in loss_out.items():
            if isinstance(v, torch.Tensor):
                metrics[f"batch/{k}"] = v.mean().item() if v.numel() > 1 else v.item()
            else:
                metrics[f"batch/{k}"] = float(v)
        return metrics

    def _compute_collection_metrics(self, flattened_data: TensorDict) -> Dict[str, float]:
        """Compute metrics from collected data."""
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            # Basic reward metrics
            if ("next", "reward") in flattened_data.keys(include_nested=True):
                rewards = flattened_data["next", "reward"]
                if rewards.numel() > 0:
                    metrics["batch/mean_reward"] = rewards.mean().item()

            # Custom environment-specific metrics
            if ("next", "observation", "aux", "improvement") in flattened_data.keys(
                include_nested=True
            ):
                improvements = flattened_data["next", "observation", "aux", "improvement"]
                valid_mask = torch.isfinite(improvements) & (improvements > -100)
                valid_improvements = improvements[valid_mask]

                if valid_improvements.numel() > 0:
                    metrics["batch/mean_improvement"] = valid_improvements.mean().item()
                    metrics["batch/max_improvement"] = valid_improvements.max().item()
                    metrics["batch/n_completed"] = float(valid_improvements.numel())
                else:
                    metrics["batch/n_completed"] = 0.0

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
        """Perform a complete training step on collected batch."""
        # Get or use provided replay buffer
        rb = replay_buffer if replay_buffer is not None else self.replay_buffer
        if rb is None:
            raise ValueError("QMIX requires a replay buffer.")

        # Flatten collected data and add to replay buffer
        flattened_data = batch.reshape(-1)
        collection_metrics = self._compute_collection_metrics(flattened_data)
        rb.extend(flattened_data)

        # Determine effective batch size
        buffer_len = len(rb)
        effective_batch_size = min(self.config.batch_size, buffer_len)

        # Don't train if buffer is empty
        if effective_batch_size == 0:
            return []

        # Use configured device if not provided
        if device is None:
            device = self.config.update_device

        # Perform multiple gradient updates per collection
        metrics_list = []
        update_start = time.perf_counter()

        for _ in range(self.config.updates_per_collection):
            # Sample batch from replay buffer
            sub_batch = rb.sample(batch_size=effective_batch_size)

            # Process batch
            sub_batch = self.process_batch(sub_batch, device)

            # Perform gradient update
            metrics = self.update(
                loss_module,
                sub_batch,
                optimizer,
                target_net_updater
            )
            metrics_list.append(metrics)

        # Add shared metrics
        update_elapsed = time.perf_counter() - update_start
        if metrics_list:
            shared_metrics = {
                **collection_metrics,
                "timing/update_seconds": update_elapsed,
                "batch/effective_batch_size": float(effective_batch_size),
                "batch/replay_buffer_size": float(buffer_len),
            }
            for metrics in metrics_list:
                metrics.update(shared_metrics)

        return metrics_list
