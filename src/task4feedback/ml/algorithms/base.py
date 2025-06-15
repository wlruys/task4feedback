from dataclasses import dataclass
from torch.nn import Module


@dataclass
class LoggingConfig:
    stats_interval: int = 1000
    save_interval: int = 100
    eval_interval: int = 100
    dpi: int = 100
    bitrate: int = 50


@dataclass
class AlgorithmConfig:
    pass
