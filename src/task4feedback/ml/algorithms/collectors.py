"""
Shared collector utilities for RL algorithms.

This module provides a unified factory function to create data collectors with the new TorchRL API,
reducing code duplication. All algorithm-specific behavior is configured via parameters.
"""

from typing import Callable, List, Optional, Union
import torch
from torchrl.collectors import Collector, MultiCollector
from torchrl.envs import EnvBase

from task4feedback.logging import training


def make_collector(
    env_constructors: List[Callable[[], EnvBase]],
    policy: torch.nn.Module,
    frames_per_batch: int,
    total_frames: Optional[int] = None,
    workers: int = 1,
    sync: bool = True,
    reset_at_each_iter: bool = True,
    policy_device: Union[str, torch.device] = "cpu",
    storing_device: Union[str, torch.device] = "cpu",
    env_device: Union[str, torch.device] = "cpu",
    use_buffers: bool = True,
    compile_policy: Optional[dict] = None,
    num_threads: Optional[int] = None,
    cat_results: Optional[str] = None,
    **kwargs,
):
    """
    Factory function to create collectors using the modern TorchRL API.

    Args:
        env_constructors: List of environment constructor callables.
        policy: Policy module for action selection.
        frames_per_batch: Number of frames to collect per batch.
        total_frames: Total frames to collect (optional, for auto-stopping).
        workers: Number of parallel workers (1 for single-process).
        sync: If True, use synchronous collection (recommended for on-policy).
              If False, use asynchronous collection (recommended for off-policy).
        reset_at_each_iter: Whether to reset environments at each iteration.
        policy_device: Device for policy inference.
        storing_device: Device for storing collected data.
        env_device: Device for environment execution.
        use_buffers: Whether to use buffers for data collection.
        compile_policy: Dict with compilation settings (e.g., {"mode": "reduce-overhead"}).
        num_threads: Number of threads for multi-process collectors.
        cat_results: How to concatenate results for multi-collectors ("stack" or None).
        **kwargs: Additional collector-specific arguments.

    Returns:
        A Collector or MultiCollector instance.
    """
    # Prepare environment workers
    env_workers = [
        env_constructors[i % len(env_constructors)]
        for i in range(workers)
    ]

    # Common collector kwargs
    collector_kwargs = {
        "frames_per_batch": frames_per_batch,
        "reset_at_each_iter": reset_at_each_iter,
        "policy_device": policy_device,
        "storing_device": storing_device,
        "env_device": env_device,
        "use_buffers": use_buffers,
        **kwargs,
    }

    if total_frames is not None:
        collector_kwargs["total_frames"] = total_frames

    if compile_policy is not None:
        collector_kwargs["compile_policy"] = compile_policy

    # Single-process collector
    if workers == 1:
        training.debug(f"Creating single-process Collector (frames_per_batch={frames_per_batch})")
        return Collector(
            env_workers[0],
            policy=policy,
            **collector_kwargs,
        )

    # Multi-process collector
    training.debug(
        f"Creating multi-process {'sync' if sync else 'async'} collector "
        f"(workers={workers}, frames_per_batch={frames_per_batch})"
    )

    multi_kwargs = collector_kwargs.copy()
    if num_threads is not None:
        multi_kwargs["num_threads"] = num_threads
    if cat_results is not None:
        multi_kwargs["cat_results"] = cat_results

    return MultiCollector(
        env_workers,
        policy,
        sync=sync,
        **multi_kwargs,
    )


