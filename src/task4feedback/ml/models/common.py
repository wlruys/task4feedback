from __future__ import annotations

import math
import torch

from task4feedback.ml.models.nn_utils import _flatten_last_dim

__all__ = ["flatten_task_grid", "build_aux_features"]

def flatten_task_grid(x: torch.Tensor, length: int, width: int, in_channels: int) -> tuple[torch.Tensor, tuple[int, ...], bool]:
    """Reshape task features `(…, T, C)` into `(B, C, H, W)` while tracking batch shape."""
    single = x.dim() == 2
    if single:
        x = x.unsqueeze(0)

    *batch_shape, T, Cin = x.shape
    if single:
        batch_shape = []
    if T != length * width:
        raise ValueError(f"tasks={T} differs from H*W={length*width}")
    if Cin != in_channels:
        raise ValueError(f"in_channels mismatch: expected {in_channels}, got {Cin}")

    B = 1
    if batch_shape:
        B = math.prod(int(d) for d in batch_shape)
    h = x.reshape(B, length, width, Cin).permute(0, 3, 1, 2).contiguous()
    return h, tuple(batch_shape), single


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(x.dtype).unsqueeze(-1) #[B, C, 1]
    counts = mask_f.sum(dim=1).clamp_min(1.0) #[B, 1]
    return (x * mask_f).sum(dim=1) / counts #[B, C]

def get_aux_feature_dim(
    *,
    add_device_load: bool = False,
    add_progress: bool = False,
    n_devices: int = 5,
) -> int:
    """Compute total number of auxiliary features."""
    n = 0
    if add_device_load:
        n += 3 * n_devices
    if add_progress:
        n += 2
    return n


def build_aux_features(
    obs,
    *,
    add_device_load: bool = False,
    add_progress: bool = False,
) -> tuple[torch.Tensor | None, tuple[int, ...] | None, torch.Tensor | None]:
    """Collect auxiliary features (device load/progress) and flatten to `(B, F)` when present."""
    progress_feat: torch.Tensor | None = None
    device_feat: torch.Tensor | None = None

    if add_progress:
        progress = obs["aux", "progress"]
        baseline = obs["aux", "baseline"]
        time = obs["aux", "time"]
        perc = time / baseline

        if progress.ndim == 0:
            progress = progress.view(1, 1)
        elif progress.ndim == 1:
            progress = progress.unsqueeze(-1)
        if perc.ndim == 0:
            perc = perc.view(1, 1)
        elif perc.ndim == 1:
            perc = perc.unsqueeze(-1)

        progress_feat = torch.cat([progress, perc], dim=-1)

    if add_device_load:
        device_load = obs["aux", "device_load"]
        device_memory = obs["aux", "device_memory"]
        device_feat = torch.cat([device_load, device_memory], dim=-1)
        
        if device_feat.ndim == 0:
            device_feat = device_feat.view(1, 1)
        elif device_feat.ndim == 1:
            device_feat = device_feat.unsqueeze(0)

    if progress_feat is None and device_feat is None:
        return None, None, None

    if device_feat is None:
        parts = [progress_feat]
    elif progress_feat is None:
        parts = [device_feat]
    else:
        parts = [progress_feat, device_feat]

    if len(parts) == 1:
        feat = parts[0]
    else:
        feat = torch.cat(parts, dim=-1)
    flat, batch_shape, _ = _flatten_last_dim(feat)
    return flat, batch_shape, feat
