import logging

from task4feedback.logging import training


def get_helper_logger(name: str) -> logging.Logger:
    """Return an experiment-utils-specific logger scoped under training logger."""
    return training.getChild(f"exp_utils.{name}")
