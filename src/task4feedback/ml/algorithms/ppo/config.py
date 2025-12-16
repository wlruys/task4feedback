from dataclasses import dataclass
from ..config import UnifiedConfig

@dataclass
class PPOConfig(UnifiedConfig):
    name: str = "ppo"
    # PPO specific defaults that might differ from UnifiedConfig
    gamma: float = 1.0
    lmbda: float = 0.99
    rollout_steps: int = 250

    # Collector defaults for PPO
    collector_reset_at_each_iter: bool = False  # PPO uses rollouts by default

