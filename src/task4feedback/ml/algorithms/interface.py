from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.data import ReplayBuffer
from torchrl.envs import EnvBase
from torchrl.objectives.common import LossModule

from .base import AlgorithmConfig


class Algorithm(ABC):
    """Base class for RL algorithms."""

    def __init__(self, config: AlgorithmConfig):
        self.config = config

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        """Initialize algorithm-specific components (advantage modules, etc)."""
        pass

    @abstractmethod
    def make_loss_module(self, model: torch.nn.Module) -> LossModule:
        """Create loss module."""
        pass

    def make_collector(
        self,
        env_constructors: List[Callable[[], EnvBase]],
        policy: torch.nn.Module,
        device: torch.device,
        storing_device: torch.device,
        seed: int | None = None,
    ) -> DataCollectorBase:
        """Create data collector with automatic frames_per_batch limiting."""
        from .collectors import make_collector

        # Compute states_per_collection from rollout_steps and graphs_per_collection
        if not hasattr(self.config, 'rollout_steps') or self.config.rollout_steps <= 0:
            raise ValueError("rollout_steps must be set and > 0 in config")

        states_per_collection = self.config.graphs_per_collection * self.config.rollout_steps
        frames_per_batch = max(1, states_per_collection)
        total_frames = self.config.num_collections * states_per_collection
        compile_policy = {"mode": "default"} if getattr(self.config, "compile_policy", False) else None

        return make_collector(
            env_constructors=env_constructors,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            workers=self.config.workers,
            sync=getattr(self.config, "collector_sync", True),
            reset_at_each_iter=False,  # Use continuous rollouts when rollout_steps > 0
            seed=seed,
            policy_device=device,
            storing_device=storing_device,
            compile_policy=compile_policy,
            num_threads=self.config.workers,
            cat_results="stack",
            init_random_frames=getattr(self.config, "initial_random_frames", None),
        )

    def get_collection_policy(
        self,
        model: torch.nn.Module,
        device: torch.device,
        mode: str = "train",
    ) -> torch.nn.Module:
        """Select or build the policy module used for data collection."""
        kind = getattr(self.config, "collector_actor", "auto")

        if kind == "policy":
            policy = getattr(model, "policy", None)
        elif kind == "qvalue":
            policy = getattr(model, "qvalue", None)
        else:
            policy = getattr(model, "policy", None) or getattr(model, "qvalue", None) or model

        if policy is None:
            raise ValueError("No suitable policy found for collection.")

        policy.train(mode == "train")
        return policy

    @abstractmethod
    def make_replay_buffer(self, batch_size: int, device: torch.device) -> Optional[ReplayBuffer]:
        """Create replay buffer (None if not needed)."""
        pass

    @abstractmethod
    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        """Process batch before training (compute advantages, move to device, etc)."""
        pass

    @abstractmethod
    def update(
        self,
        loss_module: LossModule,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Perform single gradient update, return metrics."""
        pass

    @abstractmethod
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
        """Perform complete training step on collected batch, return list of metrics."""
        pass
