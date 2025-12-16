from typing import Dict, Set, Type

from .interface import Algorithm

ALGORITHM_REGISTRY: Dict[str, Type[Algorithm]] = {}
ALGORITHM_MODEL_REQUIREMENTS: Dict[str, Set[str]] = {}


def register_algorithm(name: str, cls: Type[Algorithm]):
    """
    Register an algorithm class with a name.
    """
    if name in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm {name} is already registered.")
    ALGORITHM_REGISTRY[name] = cls


def register_algorithm_requirements(name: str, required_components: Set[str]):
    """
    Register model component requirements for an algorithm.

    Args:
        name: Algorithm name (e.g., 'ppo', 'sac', 'dqn')
        required_components: Set of required model component names
                            (e.g., {'policy', 'value'} for PPO)
    """
    ALGORITHM_MODEL_REQUIREMENTS[name] = required_components


def get_algorithm(name: str) -> Type[Algorithm]:
    """
    Get an algorithm class by name.
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm {name} not found in registry.")
    return ALGORITHM_REGISTRY[name]


def get_algorithm_requirements(name: str) -> Set[str]:
    """
    Get required model components for an algorithm.

    Args:
        name: Algorithm name (e.g., 'ppo', 'sac', 'dqn')

    Returns:
        Set of required component names. Empty set if not registered.
    """
    return ALGORITHM_MODEL_REQUIREMENTS.get(name, set())
