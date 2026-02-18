"""DQN Algorithm implementation following TorchRL best practices.

Based on:
- https://github.com/pytorch/rl/blob/main/tutorials/sphinx-tutorials/coding_dqn.py
- https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_atari.py
- https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py
"""

import time
from typing import Any, Callable, Dict, List, Optional

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
from .config import DQNConfig


class _QValueGreedyPolicy(td_nn.TensorDictModuleBase):
    """Wrap a Q-value network with greedy action selection."""

    def __init__(
        self,
        qvalue_net: torch.nn.Module,
        action_space: str,
        action_spec,
        action_mask_key: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__()
        self.qvalue_net = qvalue_net
        self.qvalue_module = QValueModule(
            action_space=action_space,
            action_value_key="action_value",
            action_mask_key=action_mask_key,
            out_keys=["action", "action_value", "chosen_action_value"],
            spec=action_spec,
        )
        self.in_keys = self.qvalue_net.in_keys
        self.out_keys = self.qvalue_module.out_keys

    def forward(self, td: TensorDict) -> TensorDict:
        td = self.qvalue_net(td)
        td = self.qvalue_module(td)
        return td


class _DQNExplorationPolicy(torch.nn.Module):
    """Compose greedy policy with epsilon-greedy exploration."""

    def __init__(
        self,
        greedy_policy: _QValueGreedyPolicy,
        exploration_module: EGreedyModule,
    ) -> None:
        super().__init__()
        self.greedy_policy = greedy_policy
        self.exploration_module = exploration_module

    def forward(self, td: TensorDict) -> TensorDict:
        td = self.greedy_policy(td)
        td = self.exploration_module(td)
        return td


class DQNAlgorithm(Algorithm):
    """Deep Q-Network algorithm implementation.

    Supports both single-agent and multi-agent (Independent Q-Learning) scenarios.
    """

    def __init__(self, config: DQNConfig):
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
        """Initialize DQN-specific components."""
        model.validate_for_algorithm("dqn")

        # Grab action spec from env for exploration and greedy argmax
        if env_constructors:
            try:
                probe_env = env_constructors[0]()
                self._action_spec = probe_env.action_spec
                probe_env.close()
            except Exception as exc:
                training.warning("Failed to inspect env for action spec: %s", exc)

        # Optionally compile the update core for performance
        if getattr(self.config, "compile_update", False):
            self._update_core = _safe_compile_with_warmup(
                self._make_update_core(),
                warmup=8,
                mode="default",
                label="dqn/update_core",
            )
        else:
            self._update_core = self._make_update_core()

    @staticmethod
    def _make_update_core():
        """Create the core update function for computing and backpropagating loss."""
        def _core(loss_module: LossModule, batch: TensorDict):
            loss_td = loss_module(batch)
            # Sum all loss components (usually just "loss" for DQN)
            loss_sum = sum(
                loss for key, loss in loss_td.items()
                if key.startswith("loss")
            )
            loss_sum.backward()
            return loss_td

        return _core

    def _get_greedy_value_network(self, model: torch.nn.Module) -> _QValueGreedyPolicy:
        if self._greedy_value_network is None:
            if model.qvalue is None:
                raise ValueError("DQN requires a qvalue network for greedy policy construction.")
            self._greedy_value_network = _QValueGreedyPolicy(
                model.qvalue,
                action_space=self.config.action_space,
                action_spec=self._action_spec,
                action_mask_key=self._action_mask_key,
            )
        return self._greedy_value_network

    def make_loss_module(self, model: torch.nn.Module) -> DQNLoss:
        """Create DQN loss module with proper key configuration.

        Args:
            model: The model containing the Q-value network

        Returns:
            Configured DQNLoss module
        """
        model.validate_for_algorithm("dqn")

        qvalue = model.qvalue
        if qvalue is None:
            raise ValueError("DQN requires qvalue network")

        # Optionally compile the Q-network for performance
        if getattr(self.config, "compile_loss_networks", False):
            qvalue = _safe_compile_with_warmup(
                qvalue,
                warmup=8,
                mode="default",
                label="dqn/qvalue",
            )
            model.qvalue = qvalue

        qvalue_with_greedy = self._get_greedy_value_network(model)

        # Create DQN loss with target network (delay_value=True) and greedy action selection
        loss_module = DQNLoss(
            value_network=qvalue_with_greedy,
            loss_function=self.config.loss_function,
            delay_value=True,  # Use separate target network
            action_space=self.config.action_space,
        )

        # Set keys for multi-agent or custom observation structures
        # Note: These keys should match your environment's output format
        if hasattr(self.config, 'multi_agent') and self.config.multi_agent:
            # Multi-agent IQL pattern
            loss_module.set_keys(
                action_value=("agents", "action_value"),
                value=("agents", "chosen_action_value"),
                action="action",
                reward=("agents", "reward"),
                done=("agents", "done"),
                terminated=("agents", "terminated"),
            )
        else:
            # Standard single-agent DQN
            # Keys can be customized based on environment
            loss_module.set_keys(
                action="action",
                done="done",
                terminated="terminated",
                action_value="action_value",
                value="chosen_action_value",
            )

        # Configure the value estimator (TD0 for DQN)
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
            Policy module for collection (with exploration in train mode)
        """
        if model.qvalue is None:
            raise ValueError("DQN collector requires a qvalue network.")

        greedy_policy = self._get_greedy_value_network(model)

        if mode != "train":
            greedy_policy.eval()
            return greedy_policy

        if self._exploration_module is None:
            spec = self._action_spec
            if spec is not None and device is not None:
                spec = spec.to(device)
            self._exploration_module = EGreedyModule(
                eps_init=self.config.eps_init,
                eps_end=self.config.eps_end,
                annealing_num_steps=self.config.eps_decay,
                action_key="action",
                action_mask_key=self._action_mask_key,
                spec=spec,
                device=device,
            )

        exploring_policy = _DQNExplorationPolicy(greedy_policy, self._exploration_module)
        exploring_policy.train()
        return exploring_policy

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

        # For multi-agent scenarios, expand done/terminated to match agent dimensions
        if hasattr(self.config, 'multi_agent') and self.config.multi_agent:
            # Expand scalar done signals to match agent dimensions
            for key in ["done", "terminated"]:
                if ("next", key) in batch.keys(include_nested=True):
                    done_val = batch.get(("next", key))
                    if done_val.ndim < batch.get("action").ndim:
                        # Expand to match agent dimension
                        done_val = done_val.unsqueeze(-1).expand(
                            batch.get_item_shape(("agents", "reward"))
                        )
                        batch.set(("next", "agents", key), done_val)

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
            loss_module: DQN loss module
            batch: Batch of transitions
            optimizer: Optimizer for Q-network
            target_net_updater: Target network updater (soft or hard)

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
        """Extract metrics from loss output.

        Args:
            loss_out: TensorDict output from loss module

        Returns:
            Dictionary of scalar metrics
        """
        metrics = {}
        for k, v in loss_out.items():
            if isinstance(v, torch.Tensor):
                metrics[f"batch/{k}"] = v.mean().item() if v.numel() > 1 else v.item()
            else:
                metrics[f"batch/{k}"] = float(v)
        return metrics

    def _compute_collection_metrics(self, flattened_data: TensorDict) -> Dict[str, float]:
        """Compute metrics from collected data.

        Args:
            flattened_data: Flattened batch of collected transitions

        Returns:
            Dictionary of collection metrics
        """
        metrics: Dict[str, float] = {}

        with torch.no_grad():
            # Basic reward metrics
            if ("next", "reward") in flattened_data.keys(include_nested=True):
                rewards = flattened_data["next", "reward"]
                if rewards.numel() > 0:
                    metrics["batch/mean_reward"] = rewards.mean().item()
                    if rewards.numel() > 1:
                        metrics["batch/std_reward"] = rewards.std().item()

            # Custom environment-specific metrics
            # (e.g., improvement, time, etc.)
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
                else:
                    training.debug(
                        "No episodes completed in collection batch. "
                        "Consider increasing frames_per_batch if this persists."
                    )
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
        """Perform a complete training step on collected batch.

        Args:
            batch: Collected batch of transitions
            loss_module: DQN loss module
            optimizer: Optimizer
            replay_buffer: Replay buffer for storing transitions
            target_net_updater: Target network updater
            device: Device for training
            n_collections: Number of collections so far
            n_updates: Number of updates so far
            n_samples: Number of samples so far

        Returns:
            List of metric dictionaries (one per update)
        """
        # Get or use provided replay buffer
        rb = replay_buffer if replay_buffer is not None else self.replay_buffer
        if rb is None:
            raise ValueError("DQN requires a replay buffer.")

        # Flatten collected data and add to replay buffer
        flattened_data = batch.reshape(-1)
        collection_metrics = self._compute_collection_metrics(flattened_data)
        rb.extend(flattened_data)

        # Update epsilon schedule based on frames collected
        if self._exploration_module is not None:
            try:
                frames = len(flattened_data)
            except Exception:
                frames = 1
            self._exploration_module.step(max(1, int(frames)))

        # Determine effective batch size
        buffer_len = len(rb)
        effective_batch_size = min(self.config.batch_size, buffer_len)

        # Don't train until we have enough data
        if effective_batch_size == 0 or buffer_len < self.config.initial_random_frames:
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

            # Process batch (move to device, handle multi-agent, etc.)
            sub_batch = self.process_batch(sub_batch, device)

            # Perform gradient update
            metrics = self.update(
                loss_module,
                sub_batch,
                optimizer,
                target_net_updater
            )
            metrics_list.append(metrics)

        # Add shared metrics to all update metrics
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
