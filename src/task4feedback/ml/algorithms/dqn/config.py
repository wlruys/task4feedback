from dataclasses import dataclass
from ..config import UnifiedConfig

@dataclass
class DQNConfig(UnifiedConfig):
    name: str = "dqn"
    type: str = "off_policy"

    # DQN specific defaults
    gamma: float = 0.99

    # Collector defaults for DQN (off-policy)
    collector_reset_at_each_iter: bool = False  # Let episodes run naturally

