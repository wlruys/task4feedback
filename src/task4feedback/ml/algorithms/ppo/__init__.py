from .algorithm import PPOAlgorithm
from .config import PPOConfig
from ..registry import register_algorithm, register_algorithm_requirements

# Register PPO algorithm and its model requirements
register_algorithm("ppo", PPOAlgorithm)
register_algorithm_requirements("ppo", {"policy", "value"})

__all__ = ["PPOAlgorithm", "PPOConfig"]
