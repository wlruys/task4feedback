from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import inspect

import torch
import torch.nn as nn
import tensordict.nn as td_nn
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig
from tensordict import TensorDictBase

from .tdgraph import Key, TDNode, to_key


def _filter_runtime_kwargs(target: Any, runtime: dict) -> dict:
    if not runtime:
        return {}
    try:
        sig = inspect.signature(target.__init__ if inspect.isclass(target) else target)
    except Exception:
        return dict(runtime)

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(runtime)

    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in runtime.items() if k in allowed}


class ModuleNode(TDNode):
    """Generic TDNode wrapper for arbitrary nn.Modules.

    - Instantiates `module` from Hydra if needed, filtering runtime kwargs.
    - Uses module-provided `in_keys` / `out_keys` when not specified.
    - Wraps plain modules in a TensorDictModule for key-based IO.
    """

    def __init__(
        self,
        *,
        module: DictConfig | nn.Module,
        in_keys: Sequence[Sequence[str]] | None = None,
        out_keys: Sequence[Sequence[str]] | None = None,
        runtime: dict | None = None,
        **_ignored,
    ):
        super().__init__()
        self._runtime = runtime or {}

        if isinstance(module, nn.Module):
            self.module = module
        else:
            target = None
            try:
                target = get_class(module["_target_"])
            except Exception:
                target = None
            kwargs = _filter_runtime_kwargs(target, self._runtime) if target else dict(self._runtime)
            self.module = instantiate(module, **kwargs, _recursive_=False)

        if in_keys is None:
            in_keys = getattr(self.module, "in_keys", None)
        if out_keys is None:
            out_keys = getattr(self.module, "out_keys", None)

        if in_keys is None or out_keys is None:
            raise ValueError(
                f"{self.__class__.__name__} requires in_keys/out_keys either in YAML or on the module."
            )

        self.in_keys: list[Key] = [to_key(k) for k in in_keys]
        self.out_keys: list[Key] = [to_key(k) for k in out_keys]

        self.requires = list(self.in_keys)
        self.provides = list(self.out_keys)

        if isinstance(self.module, td_nn.TensorDictModuleBase):
            self.td_module = self.module
        else:
            self.td_module = td_nn.TensorDictModule(
                self.module,
                in_keys=[tuple(k) for k in self.in_keys],
                out_keys=[tuple(k) for k in self.out_keys],
                inplace=True,
            )

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        out = self.td_module(td)
        return out if out is not None else td


class ObservationKeyExtractorNode(TDNode):
    """Extracts a nested key from observation into a public key."""

    def __init__(
        self,
        *,
        key_path: Sequence[str],
        out_key: Sequence[str],
        runtime: dict | None = None,
        **_ignored,
    ):
        super().__init__()
        self.key_path = to_key(key_path)
        self.out_key = to_key(out_key)
        self.requires = [("observation",)]
        self.provides = [self.out_key]

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        obs = td.get(("observation",))
        value = obs.get(self.key_path)
        td.set(self.out_key, value)
        return td


class NamespacedLSTMNode(TDNode):
    """Explicit LSTM node with public rnn state keys.

    This is optional and currently unused in configs.
    """

    def __init__(
        self,
        *,
        node_name: str,
        input_key: Sequence[str] = ("embed",),
        output_key: Sequence[str] = ("embed_rnn",),
        hidden_size: int = 256,
        reset_on_done: bool = True,
        runtime: dict | None = None,
        **_ignored,
    ):
        super().__init__()
        self.node_name = str(node_name)
        self.input_key = to_key(input_key)
        self.output_key = to_key(output_key)
        self.hidden_size = int(hidden_size)
        self.reset_on_done = bool(reset_on_done)

        self.h_key: Key = ("rnn", self.node_name, "h")
        self.c_key: Key = ("rnn", self.node_name, "c")

        self.requires = [self.input_key, ("done",)]
        self.provides = [self.output_key, self.h_key, self.c_key]

        # Placeholder cell; real input size inferred on first forward.
        self.cell = nn.LSTMCell(1, self.hidden_size)
        self._input_size: Optional[int] = None

    def _ensure_cell(self, embed: torch.Tensor) -> None:
        if self._input_size is not None:
            return
        self._input_size = int(embed.shape[-1])
        self.cell = nn.LSTMCell(self._input_size, self.hidden_size)

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        embed = td.get(self.input_key)
        self._ensure_cell(embed)

        # Flatten all batch dims (including candidates) into B.
        batch_shape = embed.shape[:-1]
        x_flat = embed.reshape(-1, embed.shape[-1])

        # Fetch or init hidden state.
        h = td.get(self.h_key, default=None)
        c = td.get(self.c_key, default=None)
        if h is None or c is None:
            h = x_flat.new_zeros((x_flat.shape[0], self.hidden_size))
            c = x_flat.new_zeros((x_flat.shape[0], self.hidden_size))
        else:
            h = h.reshape(-1, self.hidden_size)
            c = c.reshape(-1, self.hidden_size)

        done = td.get(("done",), default=None)
        if done is not None and self.reset_on_done:
            done_exp = done
            while done_exp.ndim < len(batch_shape):
                done_exp = done_exp.unsqueeze(-1)
            done_exp = done_exp.expand(batch_shape).reshape(-1).to(torch.bool)
            if done_exp.any():
                keep = (~done_exp).unsqueeze(-1)
                h = h * keep
                c = c * keep

        h_new, c_new = self.cell(x_flat, (h, c))
        out = h_new.reshape(*batch_shape, self.hidden_size)

        td.set(self.output_key, out)
        td.set(self.h_key, h_new.reshape(*batch_shape, self.hidden_size))
        td.set(self.c_key, c_new.reshape(*batch_shape, self.hidden_size))
        return td
