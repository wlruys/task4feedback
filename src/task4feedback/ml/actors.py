from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from tensordict import TensorDict

from task4feedback.ml.tdgraph import to_key


class EpsilonGreedyQValueActor(nn.Module):
    """Minimal epsilon-greedy actor for Q-value networks.

    This is intended for value-based methods (e.g., DQN) where action selection
    should be argmax-with-exploration rather than a probabilistic policy. The
    actor leaves the Q-value TensorDict entries untouched and only adds an
    ``action`` key.
    """

    def __init__(
        self,
        qvalue_network: nn.Module,
        *,
        qvalue_key: Sequence[str] | str = ("action_value",),
        action_key: Sequence[str] | str = ("action",),
        mask_key: Sequence[str] | str | None = ("observation", "aux", "candidate_mask"),
        eps_init: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 100000,
    ) -> None:
        super().__init__()
        self.qvalue_network = qvalue_network
        self.qvalue_key = to_key(qvalue_key)
        self.action_key = to_key(action_key)
        self.mask_key = to_key(mask_key) if mask_key is not None else None

        self.eps_init = float(eps_init)
        self.eps_end = float(eps_end)
        self.eps_decay = max(1, int(eps_decay))
        self.register_buffer("_steps", torch.zeros(1, dtype=torch.long), persistent=False)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, float(self._steps.item()) / float(self.eps_decay))
        return float(self.eps_init + (self.eps_end - self.eps_init) * frac)

    def _apply_mask(self, q: torch.Tensor, td: TensorDict) -> torch.Tensor:
        if self.mask_key is None:
            return q

        mask = td.get(self.mask_key, None)
        if mask is None:
            return q

        mask = mask.to(device=q.device, dtype=torch.bool)
        target_shape = q.shape

        # Expand mask to match q-value shape (broadcast across action dimension if needed)
        while mask.ndim < q.ndim:
            mask = mask.unsqueeze(-1)
        if mask.shape != target_shape:
            try:
                mask = mask.expand(target_shape)
            except RuntimeError:
                return q

        return q.masked_fill(~mask, float("-inf"))

    @torch.no_grad()
    def forward(self, td: TensorDict) -> TensorDict:
        td = self.qvalue_network(td)
        q = td.get(self.qvalue_key, None)
        if q is None:
            raise KeyError(f"Q-value key {self.qvalue_key} not found in TensorDict.")

        q_masked = self._apply_mask(q, td)
        num_actions = q_masked.shape[-1]

        greedy = q_masked.argmax(dim=-1)
        random_actions = torch.randint(
            low=0,
            high=num_actions,
            size=greedy.shape,
            device=q_masked.device,
            dtype=greedy.dtype,
        )

        eps = self.epsilon if self.training else self.eps_end
        explore_mask = torch.rand_like(greedy, dtype=torch.float) < eps
        action = torch.where(explore_mask, random_actions, greedy)

        td.set(self.action_key, action)

        # Step count increments by number of env steps collected
        self._steps += int(greedy.numel())
        return td


__all__ = ["EpsilonGreedyQValueActor"]
