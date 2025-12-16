from typing import Optional, Set

from torch.nn import Module
from torchrl.modules import ProbabilisticActor, ValueOperator


class UnifiedRLModule(Module):
    """Container for RL network components with standardized interface.

    Attributes:
        policy: Policy network (actor) for action selection
        value: Value network (critic) for state value estimation
        qvalue: Q-value network for action-value estimation
    """

    def __init__(
        self,
        policy: Optional[Module | ProbabilisticActor] = None,
        value: Optional[Module | ValueOperator] = None,
        qvalue: Optional[Module | ValueOperator] = None,
    ):
        super().__init__()
        # Use standard names only - no aliases
        self.policy = policy
        self.value = value
        self.qvalue = qvalue

    def validate_for_algorithm(self, algorithm_name: str):
        """Validate model has required components for algorithm.

        Uses registry to determine requirements - no hardcoded logic.

        Args:
            algorithm_name: Name of the algorithm (e.g., 'ppo', 'sac', 'dqn')

        Raises:
            ValueError: If algorithm is unknown or required components are missing
        """
        from task4feedback.ml.algorithms.registry import get_algorithm_requirements

        required = get_algorithm_requirements(algorithm_name)
        if not required:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        missing = []
        for component in required:
            if getattr(self, component, None) is None:
                missing.append(component)

        if missing:
            available = self.available_components()
            raise ValueError(
                f"{algorithm_name} requires {sorted(required)} network(s), "
                f"but missing: {sorted(missing)}. Available: {sorted(available)}"
            )

    def available_components(self) -> Set[str]:
        """Return set of available (non-None) components."""
        available = set()
        for name in ["policy", "value", "qvalue"]:
            if getattr(self, name, None) is not None:
                available.add(name)
        return available

    def forward(self, x):
        """Forward pass through available components."""
        out = ()
        if self.policy:
            out += (self.policy(x),)
        if self.value:
            out += (self.value(x),)
        if self.qvalue:
            out += (self.qvalue(x),)
        return out
