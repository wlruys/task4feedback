from dataclasses import dataclass

from ..config import OffPolicyConfig


@dataclass
class DQNConfig(OffPolicyConfig):
    """Deep Q-Network (DQN) configuration.

    Based on TorchRL's DQN and multi-agent IQL implementations.
    """

    name: str = "dqn"

    # Exploration (epsilon-greedy)
    eps_init: float = 1.0  # Initial exploration rate
    eps_end: float = 0.05  # Final exploration rate
    eps_decay: int = 100000  # Steps to decay from eps_init to eps_end

    # Q-Learning
    loss_function: str = "smooth_l1"  # "l2" or "smooth_l1"
    target_update_polyak: float = 0.995  # Soft update coefficient (tau)
    action_space: str = "categorical"  # Action space type

    # Replay & Training
    replay_buffer_size: int = 1000000  # Maximum replay buffer size
    batch_size: int = 256  # Minibatch size for training
    updates_per_collection: int = 64  # Gradient updates per collection
    initial_random_frames: int = 10000  # Random exploration warmup frames

    # Learning Rates
    lr: float = 3e-4  # Learning rate for Q-network

    # Defaults
    gamma: float = 0.99  # Discount factor
    collector_actor: str = "qvalue"  # Use Q-network for collection

    # Multi-agent (for Independent Q-Learning)
    multi_agent: bool = False  # Enable multi-agent IQL mode
