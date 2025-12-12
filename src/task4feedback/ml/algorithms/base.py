from dataclasses import dataclass
from torch.nn import Module


@dataclass
class LoggingConfig:
    stats_interval: int = 1000
    save_interval: int = 100
    checkpoint_interval: int = 100
    best_policy_dir: str = None
    best_policy_name: str = None
    # wandb.watch configuration (can cause 10-100x slowdown for large models)
    watch_model: bool = False  # Enable wandb.watch (default: off for performance)
    watch_log_freq: int = 100  # Log frequency when watch is enabled
    watch_log_mode: str = "gradients"  # "gradients" | "parameters" | "all"


@dataclass
class AlgorithmConfig:
    pass
