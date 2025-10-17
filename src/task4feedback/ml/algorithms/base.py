from dataclasses import dataclass
from torch.nn import Module


@dataclass
class LoggingConfig:
    stats_interval: int = 1000
    save_interval: int = 100
    checkpoint_interval: int = 100
    best_policy_dir: str = None
    best_policy_name: str = None
    log_best_policy: bool = False


@dataclass
class AlgorithmConfig:
    pass
