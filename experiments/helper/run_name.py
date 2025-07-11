# run_name.py
from __future__ import annotations
import hashlib, datetime, re, json
from omegaconf import DictConfig, OmegaConf
from typing import Sequence

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(parts: Sequence[str], *, maxlen: int = 60) -> str:
    raw = "-".join(str(p) for p in parts if p not in ("", None))
    res = _SLUG_RE.sub("-", raw)[:maxlen].strip("-")
    return res or "run"


def cfg_hash(cfg: DictConfig, *, n=8) -> str:
    as_json = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
    return hashlib.blake2b(as_json.encode(), digest_size=n).hexdigest()

def make_run_name(cfg: DictConfig) -> str:
    slug = slugify(
        [
            cfg.get("graph", {}).get("name", ""),
            cfg.get("network", {}).get("name", ""),
            cfg.get("reward", {}).get("name", ""),
        ]
    )

    h = cfg_hash(cfg)
    date = datetime.datetime.now().strftime("%m%d-%H%M%S")
    return f"{slug}-{date}-{h}"
