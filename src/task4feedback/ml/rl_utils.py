"""
Core RL utilities: advantage/reward helpers, logging helpers, and checkpoint persistence.

This module centralizes pieces that were previously mixed into `util.py` so that
training code can import narrowly scoped helpers while keeping legacy
`task4feedback.ml.util` imports working via shims.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import git
import torch
import torch.nn as nn
from tensordict import TensorDict

from task4feedback.logging import training
from task4feedback.utils.atomic import atomic_torch_save, atomic_write_text


def compute_advantage(td: TensorDict) -> TensorDict:
    """Compute cumulative rewards and advantages per trajectory."""
    with torch.no_grad():
        state_values = td["state_value"].view(-1)
        traj_ids = td["collector", "traj_ids"].view(-1)
        rewards = td["next", "reward"].view(-1)

        cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
        for traj in traj_ids.unique():
            mask = traj_ids == traj
            traj_rewards = rewards[mask]
            traj_cum_rewards = torch.flip(torch.cumsum(torch.flip(traj_rewards, dims=[0]), dim=0), dims=[0])
            cumulative_rewards[mask] = traj_cum_rewards.to(torch.float32)

        td["value_target"] = cumulative_rewards.unsqueeze(1)
        td["advantage"] = cumulative_rewards - state_values
    return td


def redistribute_rewards_uniform(td: TensorDict) -> TensorDict:
    """
    Redistribute each non-zero (bagged) reward uniformly across the preceding zero-reward steps
    of the same trajectory.
    """
    rewards = td.get(("next", "reward"))
    traj_ids = td.get(("collector", "traj_ids"))
    device = rewards.device

    W, T, _ = rewards.shape
    new_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)

    with torch.no_grad():
        for w in range(W):
            local_r = rewards[w, :, 0]
            local_traj = traj_ids[w]
            buffer = torch.zeros_like(local_r, dtype=torch.float32, device=device)

            for traj in local_traj.unique():
                mask = local_traj == traj
                if not mask.any():
                    continue

                traj_r = local_r[mask]
                r_rev = traj_r.flip(0)
                groups = torch.cumsum(r_rev != 0, dim=0)
                n_groups = int(groups.max().item()) + 1

                sum_per_group = torch.zeros(n_groups, device=device)
                cnt_per_group = torch.zeros(n_groups, device=device)
                sum_per_group.scatter_add_(0, groups, r_rev)
                cnt_per_group.scatter_add_(0, groups, torch.ones_like(r_rev))

                avg_per_group = sum_per_group / cnt_per_group
                share_rev = avg_per_group[groups]

                share = share_rev.flip(0)
                buffer[mask] = share

            new_rewards[w, :, 0] = buffer

        td.set(("next", "reward"), new_rewards)

    return td


def compute_gae(tensordict_data: TensorDict, gamma: float = 0.99, lam: float = 0.95) -> TensorDict:
    """
    Vectorized GAE computation that handles multiple trajectories efficiently.
    """
    with torch.no_grad():
        values = tensordict_data["state_value"].squeeze(-1)
        rewards = tensordict_data["next", "reward"].squeeze(-1)
        dones = tensordict_data["next", "done"].squeeze(-1)
        traj_ids = tensordict_data["collector", "traj_ids"].squeeze(-1)

        advantages = torch.zeros_like(values)
        value_targets = torch.zeros_like(values)

        unique_traj_ids = torch.unique(traj_ids)

        for traj_id in unique_traj_ids:
            traj_mask = traj_ids == traj_id
            traj_values = values[traj_mask]
            traj_rewards = rewards[traj_mask]
            traj_dones = dones[traj_mask]

            T = len(traj_values)
            next_values = torch.zeros_like(traj_values)
            next_values[:-1] = traj_values[1:] * (~traj_dones[:-1])

            deltas = traj_rewards + gamma * next_values - traj_values
            traj_advantages = torch.zeros_like(traj_values)
            gae = 0.0

            for t in reversed(range(T)):
                gae = deltas[t] + gamma * lam * gae * (~traj_dones[t])
                traj_advantages[t] = gae

            advantages[traj_mask] = traj_advantages
            value_targets[traj_mask] = traj_advantages + traj_values

        tensordict_data["advantage"] = advantages.unsqueeze(-1)
        tensordict_data["value_target"] = value_targets.unsqueeze(-1)

        return tensordict_data


def logits_to_action(logits: torch.Tensor, action):
    probs = torch.distributions.Categorical(logits=logits)
    return probs.log_prob(action), probs.entropy()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self, verbose: bool = True, name: Optional[str] = None):
        self.verbose = verbose
        self.name = name
        self.interval: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.verbose:
            name_str = f" {self.name}" if self.name else ""
            print(f"Timer{name_str}: {self.interval:.4f} seconds")

    def get_elapsed(self) -> float:
        """Returns the elapsed time in seconds."""
        return self.interval


def compute_model_fingerprint(model: nn.Module) -> str:
    """
    Generate unique fingerprint of model architecture.

    The fingerprint captures the model structure (layer names, shapes, dtypes)
    to detect incompatible checkpoints being loaded into wrong models.

    Args:
        model: PyTorch model to fingerprint

    Returns:
        16-character hex string uniquely identifying this architecture
    """
    arch_spec = {
        "class": model.__class__.__name__,
        # "param_count": sum(p.numel() for p in model.parameters()),
        # "layers": [
        #     {
        #         "name": name,
        #         "shape": list(param.shape),
        #         "dtype": str(param.dtype),
        #     }
        #     for name, param in model.named_parameters()
        # ],
    }
    arch_json = json.dumps(arch_spec, sort_keys=True)
    return hashlib.sha256(arch_json.encode()).hexdigest()[:16]


def log_parameter_and_gradient_norms(model) -> Dict[str, float]:
    """Log parameter and gradient norms to wandb."""
    param_norms: Dict[str, float] = {}
    grad_norms: Dict[str, float] = {}

    total_param_norm = 0.0
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_norm = param.detach().norm().item()
        param_norms[f"param_norm/{name}"] = param_norm
        total_param_norm += param_norm**2

        if param.grad is not None:
            grad_norm = param.grad.detach().norm().item()
            grad_norms[f"grad_norm/{name}"] = grad_norm
            total_grad_norm += grad_norm**2

    total_param_norm = total_param_norm**0.5
    total_grad_norm = total_grad_norm**0.5

    return {
        **param_norms,
        **grad_norms,
        "param_norm/total": total_param_norm,
        "grad_norm/total": total_grad_norm,
    }


def save_checkpoint(
    step: int,
    policy_module,
    value_module,
    optimizer,
    lr_scheduler=None,
    extras: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[str] = None,
    filename: Optional[str] = None,
    performance_metrics: Optional[Dict[str, float]] = None,
    wandb=None,
) -> Path:
    """
    Persist training state to disk with git metadata and performance metrics.

    Args:
        step: Training step number
        policy_module: Policy network module
        value_module: Value network module
        optimizer: Optimizer instance
        lr_scheduler: Optional learning rate scheduler
        extras: Optional extra data to save
        checkpoint_dir: Directory to save checkpoint (defaults to wandb dir or env var)
        filename: Optional filename (defaults to checkpoint_{step}.pt)
        performance_metrics: Optional performance metrics dict (e.g., {"mean_vs_EFT": 1.23})
        wandb: Optional wandb module for logging

    Returns:
        Path to saved checkpoint file
    """
    try:
        # Compute fingerprints for architecture validation
        policy_fingerprint = compute_model_fingerprint(policy_module)
        value_fingerprint = compute_model_fingerprint(value_module)

        state = dict(
            step=step,
            policy_module=policy_module.state_dict(),
            value_module=value_module.state_dict(),
            optimizer=optimizer.state_dict(),
            rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            extras=extras or {},
            performance_metrics=performance_metrics or {},  # Bundle metrics in checkpoint
            model_fingerprints={  # Architecture validation
                "policy": policy_fingerprint,
                "value": value_fingerprint,
            },
            saved_at=time.time(),  # Timestamp for tracking
            commit_hash=git.Repo(search_parent_directories=True).head.object.hexsha,
            commit_dirty=git.Repo(search_parent_directories=True).is_dirty(),
        )
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
        elif wandb is not None and wandb.run is not None and wandb.run.dir is not None:
            checkpoint_dir = Path(wandb.run.dir)
        else:
            checkpoint_dir = Path(os.environ.get("HYDRA_RUNTIME_OUTPUT_DIR", "."))

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / (filename if filename is not None else f"checkpoint_{step}.pt")

        atomic_torch_save(checkpoint_file, state)
        training.info(f"Checkpoint saved to {checkpoint_file}")

        # Also save lightweight metadata for quick queries without loading full checkpoint
        metadata_file = checkpoint_file.with_suffix('.json')
        metadata = {
            "step": step,
            "checkpoint_file": checkpoint_file.name,
            "performance_metrics": performance_metrics or {},
            "model_fingerprints": state["model_fingerprints"],
            "saved_at": state["saved_at"],
            "commit_hash": state["commit_hash"],
            "commit_dirty": state["commit_dirty"],
        }
        try:
            import json
            atomic_write_text(metadata_file, json.dumps(metadata, indent=2))
        except Exception as e:
            training.warning(f"Failed to save metadata file {metadata_file}: {e}")

        return checkpoint_file

    except Exception as e:  # pragma: no cover - passthrough logging
        training.error(f"Failed to save checkpoint at step {step}: {e}")
        raise


__all__ = [
    "compute_advantage",
    "redistribute_rewards_uniform",
    "compute_gae",
    "logits_to_action",
    "count_parameters",
    "Timer",
    "compute_model_fingerprint",
    "log_parameter_and_gradient_norms",
    "save_checkpoint",
]

