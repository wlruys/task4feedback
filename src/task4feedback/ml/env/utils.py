"""Utility functions for environment implementations."""

import torch


def sample_vector(d: int = 8, sample: bool = True) -> torch.Tensor:
    """
    Sample a random vector for RLE (Randomized Linear Ensemble) features.

    Args:
        d: Dimension of the vector (default: 8)
        sample: If True, sample from standard normal; if False, return zeros

    Returns:
        Tensor of shape (d,) with sampled or zero values
    """
    if sample:
        return torch.randn(d, dtype=torch.float32)
    else:
        return torch.zeros(d, dtype=torch.float32)


def tasks_to_steps(n_tasks: int, max_candidates: int) -> int:
    """
    Convert number of tasks to simulator steps.

    In the simulator, multiple candidates can be mapped per step. This function
    converts a task count to the corresponding number of simulator steps.

    Args:
        n_tasks: Number of tasks to map
        max_candidates: Maximum number of candidates that can be mapped per step

    Returns:
        Number of simulator steps needed
    """
    return n_tasks * max_candidates


def steps_to_tasks(steps: int, max_candidates: int) -> int:
    """
    Convert simulator steps to number of tasks.

    Args:
        steps: Number of simulator steps
        max_candidates: Maximum number of candidates that can be mapped per step

    Returns:
        Approximate number of tasks
    """
    return steps // max_candidates
