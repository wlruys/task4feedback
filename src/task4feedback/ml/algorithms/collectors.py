"""Collector factory used by RL algorithms.

Keep this thin: pick `Collector` vs `MultiCollector` and apply sane defaults.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
from torchrl.collectors import Collector, MultiCollector
from torchrl.envs import EnvBase

from task4feedback.logging import training


def _seeded_env_constructor(env_fn: Callable[[], EnvBase], seed: int) -> Callable[[], EnvBase]:
    """Wrap an env constructor and immediately apply a deterministic seed."""

    def _make() -> EnvBase:
        env = env_fn()
        try:
            env.set_seed(seed)
        except Exception:
            # Some custom env wrappers may not expose set_seed; in that case just
            # return the env and rely on the collector-level seeding.
            pass
        return env

    return _make


def make_collector(
    env_constructors: List[Callable[[], EnvBase]],
    policy: torch.nn.Module,
    frames_per_batch: int,
    total_frames: Optional[int] = None,
    workers: int = 1,
    sync: bool = True,
    reset_at_each_iter: bool = True,
    seed: int | None = None,
    policy_device: Union[str, torch.device] = "cpu",
    storing_device: Union[str, torch.device] = "cpu",
    env_device: Union[str, torch.device] = "cpu",
    use_buffers: bool = True,
    compile_policy: Optional[dict] = None,
    num_threads: Optional[int] = None,
    cat_results: str = "stack",
    
    **kwargs,
):
    if not env_constructors:
        raise ValueError("env_constructors must contain at least one callable.")

    # Build per-worker env constructors (and seed them immediately).
    env_workers: List[Callable[[], EnvBase]] = []
    for i in range(max(1, int(workers))):
        base_fn = env_constructors[i % len(env_constructors)]
        if seed is None:
            env_workers.append(base_fn)
        else:
            env_workers.append(_seeded_env_constructor(base_fn, int(seed) + i))

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

    if workers == 1:
        training.debug(f"Creating single-process Collector (frames_per_batch={frames_per_batch})")
        collector = Collector(
            env_workers[0],
            policy=policy,
            **collector_kwargs,
        )
        if seed is not None:
            try:
                collector.set_seed(int(seed))
            except Exception as exc:
                training.debug("Collector.set_seed failed: %s", exc)
        return collector

    training.debug(
        f"Creating multi-process {'sync' if sync else 'async'} collector "
        f"(workers={workers}, frames_per_batch={frames_per_batch})"
    )

    multi_kwargs = collector_kwargs.copy()
    if num_threads is not None:
        multi_kwargs["num_threads"] = num_threads
    # Always stack results: PPO advantage estimation depends on preserving the
    # (time, env) structure instead of concatenating across workers.
    multi_kwargs["cat_results"] = cat_results

    collector = MultiCollector(
        env_workers,
        policy,
        sync=sync,
        **multi_kwargs,
    )

    if seed is not None:
        try:
            collector.set_seed(int(seed))
        except Exception as exc:
            training.debug("MultiCollector.set_seed failed: %s", exc)

    return collector

