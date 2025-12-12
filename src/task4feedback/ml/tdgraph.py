from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict, TensorDictBase
import tensordict.nn as td_nn


# Canonical nested key representation used throughout.
Key = Tuple[str, ...]


def to_key(k: Any) -> Key:
    """Normalize YAML / Hydra key representations to a tuple of strings."""
    if k is None:
        raise TypeError("Key cannot be None")
    if isinstance(k, tuple):
        return tuple(str(x) for x in k)
    if isinstance(k, str):
        return (k,)
    if isinstance(k, Sequence):
        return tuple(str(x) for x in k)
    raise TypeError(f"Unsupported key type: {type(k)!r}")


@dataclass
class NamespacingConf:
    local_prefix: Key = ("net",)
    public_keys: List[Key] = field(default_factory=list)
    public_prefixes: List[Key] = field(default_factory=lambda: [("rnn",)])

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | DictConfig | None) -> "NamespacingConf":
        if cfg is None:
            return cls()
        local_prefix = to_key(OmegaConf.select(cfg, "local_prefix", default=("net",)))
        public_keys = [to_key(k) for k in OmegaConf.select(cfg, "public_keys", default=[])]
        public_prefixes = [to_key(k) for k in OmegaConf.select(cfg, "public_prefixes", default=[("rnn",)])]
        return cls(local_prefix=local_prefix, public_keys=public_keys, public_prefixes=public_prefixes)

    def is_public(self, key: Key) -> bool:
        if key in self.public_keys:
            return True
        for prefix in self.public_prefixes:
            if key[: len(prefix)] == prefix:
                return True
        return False

    def storage_key(self, graph_name: str, node_name: str, logical_key: Key) -> Key:
        if self.is_public(logical_key):
            return logical_key
        return (graph_name, *self.local_prefix, node_name, *logical_key)


@dataclass
class ValidationConf:
    check_missing: bool = True
    check_collisions: bool = True
    check_cycles: bool = True
    topo_tiebreak: str = "lexicographic"
    dry_run: bool = False

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | DictConfig | None) -> "ValidationConf":
        if cfg is None:
            return cls()
        return cls(
            check_missing=bool(OmegaConf.select(cfg, "check_missing", default=True)),
            check_collisions=bool(OmegaConf.select(cfg, "check_collisions", default=True)),
            check_cycles=bool(OmegaConf.select(cfg, "check_cycles", default=True)),
            topo_tiebreak=str(OmegaConf.select(cfg, "topo_tiebreak", default="lexicographic")),
            dry_run=bool(OmegaConf.select(cfg, "dry_run", default=False)),
        )


class TDNode(nn.Module):
    """Minimal node contract for TDGraph."""

    requires: List[Key]
    provides: List[Key]

    def forward(self, td: TensorDictBase) -> TensorDictBase:  # pragma: no cover
        raise NotImplementedError


class KeyMappedNode(nn.Module):
    """Adapter that remaps logical keys ↔ storage keys for a single node."""

    def __init__(
        self,
        node: TDNode,
        *,
        graph_name: str,
        node_name: str,
        namespacing: NamespacingConf,
        providers: Mapping[Key, str] | None = None,
    ):
        super().__init__()
        self.node = node
        self.graph_name = str(graph_name)
        self.node_name = str(node_name)
        self.namespacing = namespacing
        self.providers = providers or {}

        self.requires_logical = [to_key(k) for k in getattr(node, "requires", [])]
        self.provides_logical = [to_key(k) for k in getattr(node, "provides", [])]

        self.requires_storage = {
            k: namespacing.storage_key(
                self.graph_name,
                str(self.providers.get(k, self.node_name)),
                k,
            )
            for k in self.requires_logical
        }
        self.provides_storage = {
            k: namespacing.storage_key(self.graph_name, self.node_name, k)
            for k in self.provides_logical
        }
        # Precompute pairs for hot-path iteration.
        self._requires_pairs = tuple(self.requires_storage.items())
        self._provides_pairs = tuple(self.provides_storage.items())
        # Cache scratch TensorDict to avoid per-step allocations; keep it cleared to
        # avoid retaining references to previous-step tensors.
        self._scratch: TensorDict | None = None
        self._scratch_batch_size: tuple[int, ...] | None = None
        self._scratch_device: torch.device | None = None
        self._scratch_placeholder: torch.Tensor | None = None

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        scratch = self._scratch
        batch_size = tuple(td.batch_size)
        device = td.device
        if scratch is None or self._scratch_batch_size != batch_size or self._scratch_device != device:
            scratch = TensorDict({}, batch_size=td.batch_size, device=device)
            placeholder = (
                torch.empty((*batch_size, 0), device=device)
                if device is not None
                else torch.empty((*batch_size, 0))
            )
            scratch_set = scratch.set
            for logical_k, _ in self._requires_pairs:
                scratch_set(logical_k, placeholder)
            self._scratch = scratch
            self._scratch_batch_size = batch_size
            self._scratch_device = device
            self._scratch_placeholder = placeholder
        else:
            placeholder = self._scratch_placeholder
        td_get = td.get
        scratch_set = scratch.set
        for logical_k, storage_k in self._requires_pairs:
            scratch_set(logical_k, td_get(storage_k))

        out = self.node(scratch)
        if out is None:
            out = scratch

        out_get = out.get
        td_set = td.set
        for logical_k, storage_k in self._provides_pairs:
            td_set(storage_k, out_get(logical_k))

        # Ensure scratch does not retain references to large tensors between calls.
        if placeholder is not None:
            for logical_k, _ in self._requires_pairs:
                scratch_set(logical_k, placeholder)
            if out is scratch:
                for logical_k, _ in self._provides_pairs:
                    scratch_set(logical_k, placeholder)
        return td


class TDGraph(nn.Module):
    """TensorDict-keyed DAG"""

    def __init__(
        self,
        *,
        name: str,
        inputs: Sequence[Key],
        nodes: Mapping[str, TDNode],
        outputs: Sequence[Key] | None,
        namespacing: NamespacingConf,
        validation: ValidationConf,
    ):
        super().__init__()
        self.name = str(name)
        self.inputs = [to_key(k) for k in inputs]
        self.outputs = [to_key(k) for k in outputs] if outputs else None
        self.namespacing = namespacing
        self.validation = validation

        # TorchRL / TensorDictModule compatibility.
        # Graph inputs/outputs are logical public keys, which map to themselves.
        def _as_nested(keys: Sequence[Key]) -> List[Any]:
            out: List[Any] = []
            for k in keys:
                out.append(k[0] if len(k) == 1 else k)
            return out

        self.in_keys = _as_nested(self.inputs)
        if self.outputs is not None:
            self.out_keys = _as_nested(self.outputs)
        else:
            self.out_keys = []

        self.nodes = nn.ModuleDict(nodes)

        self._order = self._compile()
        self._exec_modules = nn.ModuleList()
        for n in self._order:
            node = self.nodes[n]
            logical_in = [to_key(x) for x in getattr(node, "requires", [])]
            logical_out = [to_key(x) for x in getattr(node, "provides", [])]

            storage_in = [
                self.namespacing.storage_key(
                    self.name, str(self._providers.get(k, n)), k
                )
                for k in logical_in
            ]
            storage_out = [
                self.namespacing.storage_key(self.name, n, k) for k in logical_out
            ]

            mod: nn.Module
            base_module = getattr(node, "module", None)
            if isinstance(base_module, nn.Module):
                mod = td_nn.TensorDictModule(
                    base_module,
                    in_keys=_as_nested(storage_in),
                    out_keys=_as_nested(storage_out),
                    inplace=True,
                )
            elif isinstance(node, td_nn.TensorDictModuleBase) and hasattr(node, "module"):
                base_module = getattr(node, "module")
                mod = td_nn.TensorDictModule(
                    base_module,
                    in_keys=_as_nested(storage_in),
                    out_keys=_as_nested(storage_out),
                    inplace=True,
                )
            else:
                # Fallback to KeyMappedNode only if node touches internal keys.
                if any(
                    not self.namespacing.is_public(k)
                    for k in logical_in + logical_out
                ):
                    mod = KeyMappedNode(
                        node,
                        graph_name=self.name,
                        node_name=n,
                        namespacing=self.namespacing,
                        providers=self._providers,
                    )
                else:
                    mod = node

            self._exec_modules.append(mod)
        # Tuple iteration is cheaper than ModuleList iteration.
        self._exec = tuple(self._exec_modules)

    @property
    def execution_order(self) -> List[str]:
        return list(self._order)

    def debug_dump(self) -> str:
        lines = [f"TDGraph(name={self.name!r})"]
        for n in self._order:
            node = self.nodes[n]
            lines.append(
                f"- {n}: requires={getattr(node,'requires',[])} provides={getattr(node,'provides',[])}"
            )
        return "\n".join(lines)

    def _compile(self) -> List[str]:
        # Collect provides and check collisions.
        provides: Dict[Key, str] = {}
        for node_name, node in self.nodes.items():
            for k in [to_key(x) for x in getattr(node, "provides", [])]:
                if self.validation.check_collisions and k in provides:
                    raise ValueError(
                        f"[{self.name}] collision: logical key {k} provided by both "
                        f"{provides[k]!r} and {node_name!r}"
                    )
                provides[k] = node_name

        # Store provider map for KeyMappedNode requires resolution.
        self._providers = dict(provides)

        # Build dependency graph.
        deps: Dict[str, set[str]] = {n: set() for n in self.nodes.keys()}
        rev: Dict[str, set[str]] = {n: set() for n in self.nodes.keys()}

        input_set = set(self.inputs)
        for node_name, node in self.nodes.items():
            for req in [to_key(x) for x in getattr(node, "requires", [])]:
                if req in input_set:
                    continue
                provider = provides.get(req)
                if provider is None:
                    if self.validation.check_missing:
                        raise ValueError(
                            f"[{self.name}] missing key {req} required by {node_name!r}"
                        )
                    continue
                deps[node_name].add(provider)
                rev[provider].add(node_name)

        # Kahn topo sort with deterministic tie-break.
        ready = sorted([n for n, d in deps.items() if not d])
        order: List[str] = []

        while ready:
            if self.validation.topo_tiebreak == "lexicographic":
                n = ready.pop(0)
            else:
                raise ValueError(
                    f"Unsupported topo_tiebreak {self.validation.topo_tiebreak!r}"
                )
            order.append(n)
            for child in sorted(rev[n]):
                deps[child].discard(n)
                if not deps[child]:
                    if child not in ready:
                        ready.append(child)
            ready.sort()

        if self.validation.check_cycles and len(order) != len(self.nodes):
            remaining = [n for n, d in deps.items() if d]
            raise ValueError(f"[{self.name}] cycle detected involving {remaining}")

        # Validate outputs if requested.
        if self.outputs and self.validation.check_missing:
            available = set(self.inputs) | set(provides.keys())
            for out_k in self.outputs:
                if out_k not in available:
                    raise ValueError(
                        f"[{self.name}] declared output {out_k} not produced"
                    )

        return order

    def forward(self, td: TensorDictBase) -> TensorDictBase:
        exec_modules = self._exec
        for mod in exec_modules:
            td = mod(td)
        return td

    @classmethod
    def from_config(
        cls,
        conf: DictConfig | Mapping[str, Any],
        *,
        namespacing: NamespacingConf,
        validation: ValidationConf,
        runtime: dict | None = None,
    ) -> "TDGraph":
        name = OmegaConf.select(conf, "name")
        if name is None:
            raise ValueError("Graph config must set 'name'")
        inputs = [to_key(k) for k in OmegaConf.select(conf, "inputs", default=[])]
        outputs = OmegaConf.select(conf, "outputs", default=None)
        if outputs is not None:
            outputs = [to_key(k) for k in outputs]

        nodes_conf = OmegaConf.select(conf, "nodes", default={})
        nodes: Dict[str, TDNode] = {}
        for node_name, node_cfg in nodes_conf.items():
            node = instantiate(node_cfg, runtime=runtime, _recursive_=False)
            if not isinstance(node, TDNode):
                raise TypeError(
                    f"Node {node_name!r} in graph {name!r} is not a TDNode: {type(node)!r}"
                )
            nodes[str(node_name)] = node

        return cls(
            name=str(name),
            inputs=inputs,
            nodes=nodes,
            outputs=outputs,
            namespacing=namespacing,
            validation=validation,
        )
