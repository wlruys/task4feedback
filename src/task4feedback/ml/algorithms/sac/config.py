from dataclasses import dataclass
from typing import Union

from ..config import OffPolicyConfig


@dataclass
class SACConfig(OffPolicyConfig):
    """Soft Actor-Critic (SAC) configuration.

    Supports single and multi-head categorical action spaces for discrete SAC.
    """

    name: str = "sac"

    # Entropy Regularization
    alpha_init: float = 1.0
    target_entropy: Union[str, float] = "auto"  # "auto" = -log(num_actions) * num_heads

    # Q-Learning
    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"
    target_update_polyak: float = 0.995

    # Replay & Training
    replay_buffer_size: int = 1000000
    batch_size: int = 256
    updates_per_collection: int = 64

    # Learning Rates
    lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # Defaults
    gamma: float = 0.99
