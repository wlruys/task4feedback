from dataclasses import dataclass
from torch.nn import Module


@dataclass
class LoggingConfig:
    stats_interval: int = 1000
    save_interval: int = 100
    checkpoint_interval: int = 100


@dataclass
class AlgorithmConfig:
    pass
