from dataclasses import dataclass
import math
from typing import List, Optional, Self, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def kaiming_init(layer, a=0.01, mode="fan_in", nonlinearity="leaky_relu"):
    """
    Initializes a layer with Kaiming He initialization.
    """
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def xavier_init(layer, gain=1.0):
    """
    Initializes a layer with Xavier initialization.
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def orthogonal_init(layer, gain=1.0):
    """
    Initializes a layer with orthogonal initialization.
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


def init_weights(m):
    """
    Initializes LayerNorm layers.
    """
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def _zero_last_linear(seq: nn.Sequential):
    last = None
    for m in reversed(seq):
        if isinstance(m, nn.Linear):
            last = m
            break
    assert last is not None
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def _tiny_last_linear(seq: nn.Sequential, std: float = 1e-4):
    last = None
    for m in reversed(seq):
        if isinstance(m, nn.Linear):
            last = m
            break
    assert last is not None
    nn.init.normal_(last.weight, std=std)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


@dataclass
class FeatureDimConfig:
    task_feature_dim: int = 12
    data_feature_dim: int = 5
    device_feature_dim: int = 12
    task_data_edge_dim: int = 3
    task_device_edge_dim: int = 2
    task_task_edge_dim: int = 1

    @staticmethod
    def from_observer(observer):
        return FeatureDimConfig(
            task_feature_dim=observer.task_feature_dim,
            data_feature_dim=observer.data_feature_dim,
            device_feature_dim=observer.device_feature_dim,
            task_data_edge_dim=observer.task_data_edge_dim,
            task_device_edge_dim=observer.task_device_edge_dim,
            task_task_edge_dim=observer.task_task_edge_dim,
        )

    @staticmethod
    def from_config(other: Self, **overrides):
        return FeatureDimConfig(
            task_feature_dim=overrides.get("task_feature_dim", other.task_feature_dim),
            data_feature_dim=overrides.get("data_feature_dim", other.data_feature_dim),
            device_feature_dim=overrides.get("device_feature_dim", other.device_feature_dim),
            task_data_edge_dim=overrides.get("task_data_edge_dim", other.task_data_edge_dim),
            task_device_edge_dim=overrides.get("task_device_edge_dim", other.task_device_edge_dim),
            task_task_edge_dim=overrides.get("task_task_edge_dim", other.task_task_edge_dim),
        )


@dataclass
class LayerConfig:
    hidden_channels: int = 16
    n_heads: int = 1
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _compute_num_downsampling_layers(length: int, width: int, minimum_resolution: int) -> int:
    h, w = int(length), int(width)
    layers = 0
    while min(h, w) >= 2 * minimum_resolution:
        h //= 2
        w //= 2
        layers += 1
    return layers


def _init_deconv_bilinear_(deconv: nn.ConvTranspose2d) -> None:
    if deconv.groups != 1:
        return
    k_h, k_w = deconv.kernel_size
    s_h, s_w = deconv.stride
    if not (k_h == k_w == s_h == s_w):
        return
    if deconv.in_channels != deconv.out_channels:
        return

    k = k_h
    factor = (k + 1) // 2
    center = factor - 1 if (k % 2 == 1) else factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    filt1d = 1 - torch.abs(og - center) / factor
    filt2d = torch.outer(filt1d, filt1d)

    with torch.no_grad():
        w = deconv.weight
        w.zero_()
        for c in range(deconv.in_channels):
            w[c, c, :, :] = filt2d
        if deconv.bias is not None:
            deconv.bias.zero_()


def _align_and_concat(up_feat: torch.Tensor, enc_feat: torch.Tensor) -> torch.Tensor:
    uh, uw = up_feat.shape[-2:]
    eh, ew = enc_feat.shape[-2:]
    dh, dw = eh - uh, ew - uw
    if dh > 0 or dw > 0:
        pad = [max(dw // 2, 0), max(dw - dw // 2, 0), max(dh // 2, 0), max(dh - dh // 2, 0)]
        up_feat = F.pad(up_feat, pad)
    elif dh < 0 or dw < 0:
        top = (-dh) // 2
        left = (-dw) // 2
        up_feat = up_feat[..., top : top + eh, left : left + ew]
    return torch.cat([up_feat, enc_feat], dim=1)


def _flatten_to_BCHW(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Accept (C,H,W) or (*batch, C, H, W). Return (B, C, H, W),
    """
    if x.dim() == 3:
        return x.unsqueeze(0), (), 1
    elif x.dim() >= 4:
        *batch, C, H, W = x.shape
        B = math.prod(int(d) for d in batch) if batch else 1
        return x.reshape(B, C, H, W), tuple(batch), B
    else:
        raise ValueError(f"Expected (C,H,W) or (*batch,C,H,W), got {tuple(x.shape)}")


def _unflatten_from_B(xB: torch.Tensor, batch_shape: Tuple[int, ...]) -> torch.Tensor:
    """Inverse of _flatten_to_BCHW for the *batch* part; keeps (C,H,W) intact."""
    return xB.squeeze(0) if not batch_shape else xB.view(*batch_shape, *xB.shape[1:])


def _flatten_last_dim(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...], int]:
    """
    Accept (P) or (*batch, P). Return (B, P), batch_shape, B.
    """
    if x.dim() == 1:
        return x.unsqueeze(0), (), 1
    elif x.dim() >= 2:
        *batch, P = x.shape
        B = math.prod(int(d) for d in batch) if batch else 1
        return x.reshape(B, P), tuple(batch), B
    else:
        raise ValueError(f"Expected (..., P), got {tuple(x.shape)}")


def _expand_to_batch(x: torch.Tensor, B: int) -> torch.Tensor:
    """
    If `x` has a leading batch dimension of 1, expand it to `B`.
    Accepts scalars/1D/2D+; returns a tensor with the same trailing shape.
    """
    if B <= 0:
        raise ValueError(f"Expected B > 0, got {B}")
    if x.dim() == 0:
        x = x.view(1, 1)
    elif x.dim() == 1:
        x = x.unsqueeze(0)
    if x.size(0) == 1 and B > 1:
        return x.expand((B,) + x.shape[1:])
    return x


def _choose_gn_groups(C: int) -> int:
    for g in (8, 4, 2):
        if C % g == 0:
            return g
    return 1
