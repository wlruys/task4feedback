from .algorithm import SACAlgorithm
from .config import SACConfig
from ..registry import register_algorithm, register_algorithm_requirements

# Register SAC algorithm and its model requirements
register_algorithm("sac", SACAlgorithm)
register_algorithm_requirements("sac", {"policy", "qvalue"})

__all__ = ["SACAlgorithm", "SACConfig"]
