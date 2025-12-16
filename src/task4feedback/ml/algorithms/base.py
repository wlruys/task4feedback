from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
from torch.nn import Module

from task4feedback.logging import training
from task4feedback.utils.atomic import atomic_write_text

if TYPE_CHECKING:
    from task4feedback.ml.eval import EvaluationConfig


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
    # Collector configuration
    collector_sync: bool = True  # Use synchronous collectors (True for stability, False for async throughput)
    collector_reset_at_each_iter: bool = True  # Whether to reset environments each iteration


def _best_metric_path(best_dir: Path) -> Path:
    return Path(best_dir) / "best_metric.json"


def load_best_performance(best_dir: Optional[str]) -> float:
    if best_dir is None:
        return 0.0
    path = _best_metric_path(Path(best_dir))
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text())
        return float(data.get("best_metric_value", data.get("best_mean_vs_EFT", 0.0)))
    except Exception as exc:
        training.warning(f"Failed to load existing best metric from {path}: {exc}")
        return 0.0


def save_best_performance(best_dir: Optional[str], value: float, checkpoint_name: str, metric_name: Optional[str] = None) -> None:
    if best_dir is None:
        return
    path = _best_metric_path(Path(best_dir))
    payload = {
        "best_metric_value": value,
        "best_metric_name": metric_name,
        "best_mean_vs_EFT": value,  # Backward compatibility
        "checkpoint": checkpoint_name,
        "updated_at": time.time(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, indent=2))
    except Exception as exc:
        training.warning(f"Failed to persist best metric to {path}: {exc}")


def should_log(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should log based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.stats_interval == 0


def should_eval(
    n_updates: int,
    eval_config: Optional[EvaluationConfig],
) -> bool:
    """Check if we should evaluate based on the current update count and logging configuration."""
    if eval_config is None:
        return False
    return eval_config.eval_interval > 0 and n_updates % eval_config.eval_interval == 0


def should_checkpoint(
    n_updates: int,
    logging_config: Optional[LoggingConfig],
) -> bool:
    """Check if we should checkpoint based on the current update count and logging configuration."""
    if logging_config is None:
        return False
    return n_updates % logging_config.checkpoint_interval == 0
