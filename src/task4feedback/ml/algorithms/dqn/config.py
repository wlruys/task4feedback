from dataclasses import dataclass

from ..config import OffPolicyConfig


@dataclass
class DQNConfig(OffPolicyConfig):
    """Deep Q-Network (DQN) configuration."""

    name: str = "dqn"

    # Exploration
    eps_init: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 100000

    # Defaults
    gamma: float = 0.99
    collector_actor: str = "qvalue_epsilon_greedy"
