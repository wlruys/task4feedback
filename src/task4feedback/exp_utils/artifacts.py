from __future__ import annotations
from .definitions import *

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Literal

import torch
from omegaconf import DictConfig, OmegaConf

from .env import NormalizationDetails
from .run_name import cfg_hash
from task4feedback.utils.atomic import (
    atomic_pickle_dump,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_text,
)

logger = logging.getLogger(__name__)

CacheKind = Literal["eval", "model", "normalization"]

ARTIFACT_SIG_VERSION = "v3"

# Missing nodes are treated as empty configs.
SIGNATURE_COMPONENT_PATHS: Dict[str, str] = {
    "graph": "graph",
    "system": "system",
    "action": "env.action",
    "env": "env",
    "runtime": "eng.runtime",
    "reward": "env.reward",
    "feature": "feature",
    "models": "models",
    "normalization": "feature.normalization",
    "algorithm": "algorithm",
    "optimizer": "optimizer",
}


@dataclass(frozen=True)
class CacheSpec:
    components: Tuple[str, ...]
    include_seed: bool


# Easy-to-edit cache invalidation specs per artifact kind.
CACHE_SPECS: Dict[CacheKind, CacheSpec] = {
    "eval": CacheSpec(
        components=("graph", "system", "runtime", "env"),
        include_seed=True,
    ),
    "model": CacheSpec(
        components=(
            "graph",
            "system",
            "runtime",
            "env",
            "reward",
            "feature",
            "models",
            "algorithm",
            "optimizer",
            "lr_scheduler",
            "noise",
        ),
        include_seed=True,
    ),
    "normalization": CacheSpec(
        components=("graph", "system", "runtime", "env", "feature", "normalization"),
        include_seed=False,
    ),
}


CACHE_ROOTS: Dict[CacheKind, Path] = {
    "eval": Path("cached_evals"),
    "model": Path("cached_models"),
    "normalization": Path("cached_norms"),
}


@dataclass(frozen=True)
class ArtifactSignature:
    """Expanded signature containing per-component hashes."""

    version: str
    components: Dict[str, str]
    seed: int
    graph_type: str

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "ArtifactSignature":
        seed = int(OmegaConf.select(cfg, "seed", default=0) or 0)
        graph_type = OmegaConf.select(cfg, "graph.type", default=None)
        if not graph_type:
            graph_type = OmegaConf.select(cfg, "graph", default={}).get("type", "unknown_graph")
        graph_type = str(graph_type)

        components: Dict[str, str] = {}
        for name, path in SIGNATURE_COMPONENT_PATHS.items():
            node = _select_cfg_node(cfg, path)
            try:
                components[name] = cfg_hash(node)
            except Exception:
                components[name] = ""
        return cls(
            version=ARTIFACT_SIG_VERSION,
            components=components,
            seed=seed,
            graph_type=graph_type,
        )

    def short(self, component: str, n: int = 8) -> str:
        return short_hash(self.components.get(component, ""), n=n)


@dataclass(frozen=True)
class CacheSignature:
    kind: CacheKind
    version: str
    components: Dict[str, str]
    subset_hash: str
    seed: Optional[int]


@dataclass(frozen=True)
class CacheContext:
    kind: CacheKind
    dir: Path
    signature: ArtifactSignature
    cache_signature: CacheSignature


def ensure_cache_dir(cfg: DictConfig, kind: CacheKind) -> CacheContext:
    """Create (if needed) and return the cache directory for `kind`."""
    if kind not in CACHE_SPECS:
        raise ValueError(f"Unknown cache kind: {kind}")

    signature = ArtifactSignature.from_cfg(cfg)
    spec = CACHE_SPECS[kind]

    subset_components = {k: signature.components.get(k, "") for k in spec.components}
    seed = signature.seed if spec.include_seed else None
    subset_hash = _hash_payload({"kind": kind, "components": subset_components, "seed": seed})

    cache_sig = CacheSignature(
        kind=kind,
        version=ARTIFACT_SIG_VERSION,
        components=subset_components,
        subset_hash=subset_hash,
        seed=seed,
    )

    subdir = _cache_subdir_name(spec, signature, subset_hash)
    root = CACHE_ROOTS[kind]
    target = root / signature.graph_type / subdir
    if seed is not None:
        target = target / f"seed-{seed:03d}"

    target.mkdir(parents=True, exist_ok=True)

    ctx = CacheContext(kind=kind, dir=target, signature=signature, cache_signature=cache_sig)
    _write_signature(ctx)
    _write_config_files(ctx, cfg)
    return ctx


def eval_cache_context(cfg: DictConfig) -> CacheContext:
    return ensure_cache_dir(cfg, "eval")


def model_cache_context(cfg: DictConfig) -> CacheContext:
    return ensure_cache_dir(cfg, "model")


def normalization_cache_context(cfg: DictConfig) -> CacheContext:
    return ensure_cache_dir(cfg, "normalization")


def save_normalization_state(
    cfg: DictConfig,
    normalization: Optional[NormalizationDetails],
    *,
    ctx: Optional[CacheContext] = None,
) -> Optional[CacheContext]:
    """Persist observation normalization state into the normalization cache."""
    if normalization is None:
        return ctx
    ctx = ctx or normalization_cache_context(cfg)

    norm_path = ctx.dir / "normalization.pt"
    states = {name: _cpu_state(state) for name, state in normalization.states.items()}
    atomic_torch_save(norm_path, states)

    meta_path = ctx.dir / "normalization.json"
    payload = {
        "subset_hash": ctx.cache_signature.subset_hash,
        "path": norm_path.name,
        "saved_at": time.time(),
    }
    atomic_write_text(meta_path, json.dumps(payload, indent=2))
    return ctx


def load_normalization_state(
    cfg: DictConfig,
    *,
    ctx: Optional[CacheContext] = None,
) -> Optional[NormalizationDetails]:
    """Load observation normalization state from the normalization cache."""
    ctx = ctx or normalization_cache_context(cfg)
    norm_path = ctx.dir / "normalization.pt"
    if not norm_path.exists():
        return None
    try:
        states = torch.load(norm_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Failed to load normalization at %s: %s", norm_path, exc)
        return None
    return NormalizationDetails(states=states)


def short_hash(h: str, n: int = 8) -> str:
    if not h:
        return "none"
    return h.split("-", 1)[-1][:n]


def _select_cfg_node(cfg: DictConfig, path: str) -> DictConfig:
    node = OmegaConf.select(cfg, path, default=None)
    if node is None:
        return OmegaConf.create({})
    if isinstance(node, DictConfig):
        return node
    return OmegaConf.create(node)


def _hash_payload(payload: Dict[str, Any], *, n: int = 16) -> str:
    data = {"version": ARTIFACT_SIG_VERSION, **payload}
    as_json = json.dumps(data, sort_keys=True, default=str)
    h = hashlib.blake2b(as_json.encode(), digest_size=n).hexdigest()
    return f"{ARTIFACT_SIG_VERSION}-{h}"


def _cache_subdir_name(spec: CacheSpec, signature: ArtifactSignature, subset_hash: str) -> str:
    parts = [f"{name}-{signature.short(name)}" for name in spec.components]
    parts.append(f"sig-{short_hash(subset_hash)}")
    return "_".join(parts)


def _write_signature(ctx: CacheContext) -> None:
    path = ctx.dir / "signature.json"
    payload = {
        "kind": ctx.kind,
        "sig_version": ctx.signature.version,
        "subset_hash": ctx.cache_signature.subset_hash,
        "seed": ctx.cache_signature.seed,
        "graph_type": ctx.signature.graph_type,
        "components": ctx.cache_signature.components,
        "all_components": ctx.signature.components,
        "created_at": time.time(),
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


def _write_config_files(ctx: CacheContext, cfg: DictConfig) -> None:
    spec = CACHE_SPECS[ctx.kind]
    # Per-component YAMLs
    for name in spec.components:
        node = _select_cfg_node(cfg, SIGNATURE_COMPONENT_PATHS.get(name, name))
        path = ctx.dir / f"{name}.yaml"
        if not path.exists():
            # Don't resolve sub-nodes as they may contain relative interpolations that break
            atomic_write_text(path, OmegaConf.to_yaml(node, resolve=False))

    # Full config for convenience
    full_path = ctx.dir / "full_config.yaml"
    if not full_path.exists():
        # Don't resolve to avoid issues with relative interpolations
        atomic_write_text(full_path, OmegaConf.to_yaml(cfg, resolve=False))


def _cpu_state(state: Dict[str, Any]) -> Dict[str, Any]:
    processed: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            processed[key] = value.detach().cpu()
        else:
            processed[key] = value
    return processed
