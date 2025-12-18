from dataclasses import dataclass
from typing import Optional

from .base import AlgorithmConfig


@dataclass
class BaseRLConfig(AlgorithmConfig):
    """Base configuration for all RL algorithms."""

    # Metadata
    name: str = "base"
    implementation: str = "torchrl"

    # Environment & Collection
    workers: int = 1
    threads_per_worker: int = 1
    graphs_per_collection: int = 10
    states_per_collection: int = 1920
    num_collections: int = 1000
    timeout: int = 86400

    # Devices
    collect_device: str = "cpu"
    update_device: str = "cpu"
    storing_device: str = "cpu"

    # Core Learning
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    collector_actor: str = "auto"  # policy, qvalue, qvalue_epsilon_greedy, or auto

    # Performance
    compile_policy: bool = False
    compile_update: bool = False
    compile_loss_networks: bool = False
    log_norms: bool = False


@dataclass
class OnPolicyConfig(BaseRLConfig):
    """Configuration for on-policy algorithms (PPO, A2C)."""

    type: str = "on_policy"
    rollout_steps: int = 0  # 0 = auto from states_per_collection / workers
    minibatch_size: int = 64
    epochs_per_collection: int = 4
    normalize_advantage: bool = False
    collector_reset_at_each_iter: bool = False


@dataclass
class OffPolicyConfig(BaseRLConfig):
    """Configuration for off-policy algorithms (SAC, DQN)."""

    type: str = "off_policy"
    replay_buffer_size: int = 1000000
    batch_size: int = 256
    updates_per_collection: int = 1
    collector_reset_at_each_iter: bool = True
