"""Distribution classes for action sampling in RL policies.

This module contains custom distribution implementations that extend PyTorch's
base distributions with features like masking, multi-head support, etc.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical, Independent, constraints


class MultiHeadCategoricalMasked(Independent):
    """Multi-head categorical distribution with dynamic head masking.

    Args:
        logits: Unnormalized log probabilities of shape (..., num_heads, num_actions)
        probs: Normalized probabilities of shape (..., num_heads, num_actions)
        head_mask: Boolean mask of shape (..., num_heads). True = active, False = masked
        inactive_action: Action index to use for masked heads (default: 0)
        validate_args: Whether to validate distribution arguments
        reinterpreted_batch_ndims: Number of batch dimensions to reinterpret as event dims (default: 1)

    Example:
        >>> logits = torch.randn(4, 3, 5)  # batch=4, heads=3, actions=5
        >>> mask = torch.tensor([[True, True, False], ...])  # 3rd head masked
        >>> dist = MultiHeadCategoricalMasked(logits=logits, head_mask=mask, inactive_action=0)
        >>> actions = dist.sample()  # Shape: (4, 3), with actions[:, 2] == 0
    """

    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_rsample = False

    def __init__(
        self,
        logits: Tensor | None = None,
        probs: Tensor | None = None,
        head_mask: Tensor | None = None,
        inactive_action: int = 0,
        validate_args: bool | None = None,
        reinterpreted_batch_ndims: int = 1,
    ) -> None:
        # Process mask first to apply to logits before creating the distribution
        if head_mask is not None and logits is not None:
            mask = head_mask.to(device=logits.device, dtype=torch.bool)
            # Expand mask to match logits batch shape if needed
            if mask.shape != logits.shape[:-1]:
                mask = mask.expand(logits.shape[:-1])

            masked_logits = logits.clone()
            num_actions = logits.shape[-1]
            inactive_mask = ~mask

            if inactive_mask.any():
                masked_logits = torch.where(
                    inactive_mask.unsqueeze(-1),
                    torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype),
                    masked_logits
                )
                action_mask = torch.zeros(num_actions, dtype=torch.bool, device=logits.device)
                action_mask[inactive_action] = True

                inactive_action_mask = inactive_mask.unsqueeze(-1) & action_mask

                masked_logits = torch.where(
                    inactive_action_mask,
                    torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                    masked_logits
                )

            logits = masked_logits

        base = Categorical(logits=logits, probs=probs, validate_args=validate_args)

        super().__init__(
            base,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            validate_args=validate_args,
        )

        self._inactive = inactive_action
        self._logits = base.logits
        self._device = self._logits.device
        self._batch_shape = base.batch_shape

        if head_mask is None:
            self._mask = None 
        else:
            self._mask = head_mask.to(device=self._device, dtype=torch.bool)
            if self._mask.shape != self._batch_shape:
                self._mask = self._mask.expand(self._batch_shape)

    def _expand_mask(self, shape: tuple[int, ...]) -> Tensor:
        m = self._mask
        lead = len(shape) - m.ndim
        if lead > 0:
            # Add leading dimensions and expand
            m = m.view((1,) * lead + m.shape).expand(shape)
        return m

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        out = self.base_dist.sample(sample_shape)
        if self._mask is None:
            return out

        mask = self._expand_mask(out.shape) if out.shape != self._mask.shape else self._mask
        return out.masked_fill_(~mask, self._inactive)

    @property
    def mode(self) -> Tensor:
        m = self._logits.argmax(dim=-1)
        if self._mask is None:
            return m
        return m.masked_fill_(~self._mask, self._inactive)

    @property
    def logits(self) -> Tensor:
        """Return unnormalized log probabilities."""
        return self._logits

    @property
    def probs(self) -> Tensor:
        """Return normalized probabilities."""
        return self.base_dist.probs

    @property
    def deterministic_sample(self) -> Tensor:
        """Return deterministic sample (same as mode)."""
        return self.mode

    @property
    def mean(self) -> Tensor:
        m = self.base_dist.mean
        if self._mask is None:
            return m
        return m.masked_fill(~self._mask, float(self._inactive))

    def log_prob(self, value: Tensor) -> Tensor:
        if self._mask is None:
            return self.base_dist.log_prob(value).sum(dim=-1)

        # Expand mask if needed
        mask = self._expand_mask(value.shape) if value.shape != self._mask.shape else self._mask

        # Set masked positions to 0 (arbitrary valid action) for log_prob computation
        lp = self.base_dist.log_prob(value.masked_fill(~mask, 0))

        # Zero out log probabilities for masked heads, then sum
        return lp.mul_(mask.float()).sum(dim=-1)

    def entropy(self) -> Tensor:
        ent = self.base_dist.entropy()
        if self._mask is None:
            return ent.sum(dim=-1)
        return ent.mul_(self._mask.float()).sum(dim=-1)


def MultiHeadCategorical(**kwargs):
    """Helper function to create multi-head categorical without masking.

    Args:
        **kwargs: Arguments passed to Categorical constructor

    Returns:
        Independent(Categorical) distribution with reinterpreted_batch_ndims=1
    """
    base = torch.distributions.Categorical(**kwargs)
    return torch.distributions.Independent(base, 1)
