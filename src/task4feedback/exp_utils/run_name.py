from __future__ import annotations
from .definitions import *
import hashlib, datetime, re, json
from omegaconf import DictConfig, OmegaConf
from typing import Sequence

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(parts: Sequence[str], *, maxlen: int = 60) -> str:
    raw = "-".join(str(p) for p in parts if p not in ("", None))
    res = _SLUG_RE.sub("-", raw)[:maxlen].strip("-")
    return res or "run"


def cfg_hash(cfg: DictConfig, *, n=16, version="v2") -> str:
    """
    Generate deterministic hash of config with collision resistance.

    Args:
        cfg: Config to hash
        n: Hash size in bytes (default 16 = 32 hex chars, ~2^128 space)
        version: Hash version for schema migration

    Returns:
        Versioned hash string (e.g., "v2-abc123...")

    Note:
        Uses resolve=False to hash config structure, not runtime-resolved values.
        This ensures configs hash consistently regardless of interpolation context.
    """
    # Use resolve=False to hash structure, not runtime values
    container = OmegaConf.to_container(cfg, resolve=False)

    # Include version in hash input for schema migration
    payload = {"version": version, "config": container}
    as_json = json.dumps(payload, sort_keys=True)

    hash_bytes = hashlib.blake2b(as_json.encode(), digest_size=n)
    return f"{version}-{hash_bytes.hexdigest()}"

def make_run_name(cfg: DictConfig, include_hash: bool = True) -> str:
    """
    Generate human-readable run name with optional hash.

    Args:
        cfg: Config to generate name from
        include_hash: Whether to include config hash (default True)

    Returns:
        Run name string (e.g., "jacobi-cnn-seed042-20251209-143052-abc123")
    """
    parts = [
        cfg.get("graph", {}).get("type", ""),      # Graph type (e.g., "jacobi")
        (cfg.get("models", None) or cfg.get("network", {})).get("name", ""),  # Model arch (e.g., "cnn")
        f"seed{cfg.get('seed', 0):03d}",           # Explicit seed with zero-padding
    ]
    slug = slugify(parts)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if include_hash:
        h = cfg_hash(cfg)
        # Extract just the hex part (after "v2-" prefix) and take first 8 chars
        hash_part = h.split("-", 1)[-1][:8]
        return f"{slug}-{date}-{hash_part}"
    return f"{slug}-{date}"
