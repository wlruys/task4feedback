from dataclasses import dataclass
from typing import Union
from ..config import UnifiedConfig

@dataclass
class SACConfig(UnifiedConfig):
    """Configuration for Soft Actor-Critic (SAC) algorithm.

    Supports both single categorical and multi-head categorical action spaces
    for discrete action SAC. Multi-head support enables batched task mapping
    where multiple independent categorical decisions are made simultaneously.

    Key parameters:
        alpha_init: Initial entropy temperature coefficient
        target_entropy: Target entropy for auto-tuning ("auto" or float)
        num_qvalue_nets: Number of Q-networks for double Q-learning
        loss_function: Q-value loss ("smooth_l1" or "l2")
        replay_buffer_size: Size of experience replay buffer
        updates_per_collection: Number of gradient updates per collection step
    """
    name: str = "sac"
    type: str = "off_policy"

    # SAC specific defaults
    gamma: float = 0.99
    rollout_steps: int = 0  # Off-policy doesn't use rollouts

    # Collector defaults for SAC (off-policy)
    collector_reset_at_each_iter: bool = False  # Let episodes run naturally

    # Entropy regularization
    alpha_init: float = 1.0
    target_entropy: Union[str, float] = "auto"  # "auto" sets to -log(num_actions) * num_heads

    # Q-learning parameters
    num_qvalue_nets: int = 2  # Double Q-learning
    loss_function: str = "smooth_l1"
    target_update_polyak: float = 0.995

    # Replay buffer and training
    replay_buffer_size: int = 1000000
    batch_size: int = 256
    updates_per_collection: int = 64

    # Learning rates
    lr: float = 3e-4  # Actor learning rate
    q_lr: float = 3e-4  # Q-network learning rate
    alpha_lr: float = 3e-4  # Temperature learning rate

