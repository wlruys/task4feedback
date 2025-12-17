from dataclasses import dataclass
from ..config import UnifiedConfig

@dataclass
class PPOConfig(UnifiedConfig):
    name: str = "ppo"
    # PPO specific defaults that might differ from UnifiedConfig
    gamma: float = 0.99
    lmbda: float = 0.95
    rollout_steps: int = 250
    normalize_advantage: bool = True

    # Collector defaults for PPO
    collector_reset_at_each_iter: bool = False  # PPO uses rollouts by default

