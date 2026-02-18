from dataclasses import dataclass
from ..config import OffPolicyConfig


@dataclass
class QMIXConfig(OffPolicyConfig):
    """QMIX (Monotonic Value Function Factorisation) configuration.

    Based on TorchRL's multi-agent IQL/QMIX implementations.
    """

    name: str = "qmix"

    # Q-Learning parameters
    loss_function: str = "smooth_l1"  # "smooth_l1" or "l2"
    target_update_polyak: float = 0.995  # Soft update coefficient (tau)
    action_space: str = "categorical"

    # Replay buffer (larger for multi-agent)
    replay_buffer_size: int = 1_000_000
    batch_size: int = 256
    updates_per_collection: int = 64
    initial_random_frames: int = 10000

    # Learning rate
    lr: float = 3e-4  # Learning rate for Q-network

    # Core RL
    gamma: float = 0.99
    rollout_steps: int = 256

    # Exploration (epsilon-greedy)
    eps_init: float = 1.0  # Initial exploration rate
    eps_end: float = 0.05  # Final exploration rate
    eps_decay: int = 500_000  # Steps to anneal epsilon

    # Multi-agent configuration
    multi_agent: bool = True  # Enable multi-agent mode
    collector_actor: str = "qvalue"  # Use Q-network for collection
