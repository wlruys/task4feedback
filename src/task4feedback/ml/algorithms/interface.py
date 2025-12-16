from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from tensordict import TensorDict
from torchrl.collectors import DataCollectorBase
from torchrl.data import ReplayBuffer
from torchrl.envs import EnvBase
from torchrl.objectives.common import LossModule

from .base import AlgorithmConfig


class Algorithm(ABC):
    """
    Abstract base class for RL algorithms.
    """

    def __init__(self, config: AlgorithmConfig):
        self.config = config

    def initialize(
        self,
        model: torch.nn.Module,
        device: torch.device,
        env_constructors: Optional[List[Callable[[], EnvBase]]] = None,
    ):
        """
        Initialize the algorithm with the model and device.
        Useful for setting up advantage modules or other model-dependent components.
        """
        pass

    @abstractmethod
    def make_loss_module(self, model: torch.nn.Module) -> LossModule:
        """
        Create and return the loss module for the algorithm.
        """
        pass

    def make_collector(
        self,
        env_constructors: List[Callable[[], EnvBase]],
        policy: torch.nn.Module,
        device: torch.device,
        storing_device: torch.device,
    ) -> DataCollectorBase:
        """
        Create and return the data collector (default implementation).

        Uses the unified collector factory. All behavior is configured via the config object.
        Algorithms typically don't need to override this unless they have special requirements.
        """
        from .collectors import make_collector

        frames_per_batch = max(1, self.config.states_per_collection)
        total_frames = self.config.num_collections * self.config.states_per_collection

        compile_policy = {"mode": "reduce-overhead"} if getattr(self.config, "compile_policy", False) else None

        return make_collector(
            env_constructors=env_constructors,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            workers=self.config.workers,
            sync=getattr(self.config, "collector_sync", True),
            reset_at_each_iter=getattr(self.config, "collector_reset_at_each_iter", True),
            policy_device=device,
            storing_device=storing_device,
            compile_policy=compile_policy,
        )

    @abstractmethod
    def make_replay_buffer(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[ReplayBuffer]:
        """
        Create and return the replay buffer (if needed).
        Returns None if the algorithm does not use a replay buffer.
        """
        pass

    @abstractmethod
    def process_batch(self, batch: TensorDict, device: torch.device) -> TensorDict:
        """
        Process a batch of data before updating.
        This can include moving to device, expanding rewards, etc.
        """
        pass

    @abstractmethod
    def update(
        self,
        loss_module: LossModule,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
        target_net_updater: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Perform a single update step.
        Returns a dictionary of metrics to log.
        """
        pass

    def train_step(
        self,
        batch: TensorDict,
        loss_module: LossModule,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Optional[ReplayBuffer] = None,
        target_net_updater: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, float]]:
        """
        Perform a training step on the collected batch.
        This handles the difference between on-policy (update on batch) and off-policy (add to buffer, sample, update).
        
        Args:
            batch: The batch of data collected from the environment.
            loss_module: The loss module.
            optimizer: The optimizer.
            replay_buffer: The replay buffer (optional, for off-policy).
            target_net_updater: The target network updater (optional).
            device: The device to perform updates on.
            
        Returns:
            A list of dictionaries containing metrics for each update step performed.
        """
        raise NotImplementedError("Algorithm must implement train_step.")

    def update_on_collection(
        self,
        loss_module: LossModule,
        batch: TensorDict,
        optimizer: torch.optim.Optimizer,
    ) -> List[Dict[str, float]]:
        """
        Perform updates on a collected batch (for on-policy algorithms).
        Returns a list of metrics dictionaries (one per update).
        """
        raise NotImplementedError("This algorithm does not support update_on_collection.")
