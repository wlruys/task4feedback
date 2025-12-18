from dataclasses import dataclass
from typing import Optional

from ..config import OnPolicyConfig


@dataclass
class PPOConfig(OnPolicyConfig):
    """PPO algorithm configuration."""

    name: str = "ppo"

    # PPO Core
    clip_eps: float = 0.2
    clip_vloss: bool = True
    target_kl: Optional[float] = None

    # Loss Coefficients
    ent_coef: float = 0.001
    val_coef: float = 0.5
    value_norm: str = "l1"

    # Advantage Estimation
    lmbda: float = 0.95
    advantage_type: str = "gae"  # "gae" or "vtrace"
    vtrace_use_lambda: bool = False
    compile_advantage: bool = False

    # Policy
    collector: str = "multi_sync"

    # Defaults
    gamma: float = 0.998
    rollout_steps: int = 16
    normalize_advantage: bool = True
