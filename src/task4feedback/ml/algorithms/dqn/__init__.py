from .algorithm import DQNAlgorithm
from .config import DQNConfig
from ..registry import register_algorithm, register_algorithm_requirements

# Register DQN algorithm and its model requirements
register_algorithm("dqn", DQNAlgorithm)
register_algorithm_requirements("dqn", {"qvalue"})

__all__ = ["DQNAlgorithm", "DQNConfig"]
