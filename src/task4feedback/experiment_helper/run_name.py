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


def calculate_ratio(interior, boundary):
    """
    Given interior and boundary ratios wrt computation time, calculate the corresponding
    arithmetic intensity and boundary width.
    Returns:
    - arithmetic_intensity: str
    - boundary_width: str

    """
    if interior < boundary:
        raise ValueError("Interior must be greater than or equal to Boundary")

    if interior not in [100, 10, 1, 0.1] or boundary not in [100, 10, 1, 0.1]:
        raise ValueError("Interior and Boundary must be one of [100, 10, 1, 0.1]")

    val_intensity = 595.5555555 / interior
    arithmetic_intensity = f"{val_intensity:.7f}"[:-1]

    val_boundary = 0.25 / (interior / boundary)
    boundary_width = f"{val_boundary}"

    return arithmetic_intensity, boundary_width


def make_folder_name(cfg: DictConfig):
    """
    Create a folder name based on configuration.
    Returns:
    - folder_name: str
    - graph_name: str
    - interior_ratio: str
    - boundary_ratio: str
    """

    if cfg.graph.config.get("r_interior") is not None and cfg.graph.config.get("r_boundary") is not None:
        interior_ratio = cfg.graph.config.r_interior
        boundary_ratio = cfg.graph.config.r_boundary
    else:

        def closest_ratio_string(value: float) -> str:
            mapping = {100: "100", 10: "10", 1: "1", 0.1: "0.1"}
            closest = min(mapping.keys(), key=lambda x: abs(value - x))
            return mapping[closest]

        interior_ratio = 595.5555555 / (cfg.graph.config.arithmetic_intensity)
        boundary_ratio = interior_ratio * cfg.graph.config.boundary_width * 4

        interior_ratio = closest_ratio_string(interior_ratio)
        boundary_ratio = closest_ratio_string(boundary_ratio)

    if OmegaConf.select(cfg, "graph.config.workload_args.traj_type") is not None:
        graph_name = cfg.graph.config.workload_args.traj_type
    else:
        graph_name = "static"

    if cfg.graph.env.change_duration:
        if cfg.graph.config.workload_args.traj_type == "circle":
            graph_name = "ncircle"
        elif cfg.graph.config.workload_args.traj_type == "corners":
            graph_name = "noise"
    if cfg.graph.config.steps > 256:
        graph_name = "l" + graph_name

    return (
        f"{cfg.graph.config.n}w_{cfg.graph.config.steps}lvl_{cfg.system.n_devices-1}gpu_{graph_name}_{interior_ratio}-{boundary_ratio}_{int(cfg.system.mem/1e9)}GB",
        graph_name,
        interior_ratio,
        boundary_ratio,
    )
