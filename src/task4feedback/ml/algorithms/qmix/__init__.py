from .algorithm import QMIXAlgorithm
from .config import QMIXConfig
from ..registry import register_algorithm, register_algorithm_requirements

# Register QMIX
register_algorithm("qmix", QMIXAlgorithm)
register_algorithm_requirements("qmix", {"qvalue"})

__all__ = ["QMIXAlgorithm", "QMIXConfig"]
