#!/usr/bin/env python3
import os
import sys
import json
import time
import math
import random
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import hydra

cs = ConfigStore.instance()
cs.store(name="base", node={})

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def get_cpu_affinity() -> Optional[str]:
    try:
        if hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0))
            ranges = []
            start = prev = None
            for c in cpus:
                if start is None:
                    start = prev = c
                elif c == prev + 1:
                    prev = c
                else:
                    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                    start = prev = c
            if start is not None:
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            return ",".join(ranges)
    except Exception:
        pass
    return None

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

_TERMINATE = False
def _sig_handler(signum, frame):
    global _TERMINATE
    _TERMINATE = True
    print(f"[{now_utc_iso()}] Received signal {signum}. Requesting graceful shutdown...", flush=True)

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

@hydra.main(version_base=None, config_name="base", config_path=None)
def main(cfg: DictConfig) -> None:
    duration = float(cfg.get("duration", 5.0))
    busy = bool(cfg.get("busy", False))
    seed = int(cfg.get("seed", 0))
    fail_prob = float(cfg.get("fail_prob", 0.0))

    wb = cfg.get("wandb", {}) or {}
    wb_name = str(wb.get("name", f"dummy_{os.getpid()}"))
    wb_tags = list(wb.get("tags", [])) if isinstance(wb.get("tags", []), (list, tuple)) else []

    print("=" * 80)
    print(f"[{now_utc_iso()}] PID={os.getpid()}  CWD={os.getcwd()}")
    print(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    aff = get_cpu_affinity()
    print(f"CPU affinity (Linux): {aff if aff else 'N/A'}")
    print(f"Will run for ~{duration:.2f}s | busy={busy} | seed={seed} | fail_prob={fail_prob}")
    print(f"wandb.name={wb_name}  wandb.tags={wb_tags}")
    print("=" * 80, flush=True)

    random.seed(seed)

    if fail_prob > 0 and random.random() < fail_prob:
        print(f"[{now_utc_iso()}] Simulating failure (fail_prob={fail_prob})", flush=True)
        sys.exit(1)

    t0 = time.time()
    if busy:
        iters = 0
        while time.time() - t0 < duration:
            _ = math.sin(iters * 0.0001) * math.cos(iters * 0.0003)
            iters += 1
            if _TERMINATE:
                print(f"[{now_utc_iso()}] Terminated during busy loop.", flush=True)
                sys.exit(130)
    else:
        end = t0 + duration
        while time.time() < end:
            if _TERMINATE:
                print(f"[{now_utc_iso()}] Terminated during sleep.", flush=True)
                sys.exit(130)
            time.sleep(min(0.2, end - time.time()))

    elapsed = time.time() - t0
    metrics = {
        "timestamp": now_utc_iso(),
        "pid": os.getpid(),
        "duration_sec": elapsed,
        "busy": busy,
        "seed": seed,
        "cpu_affinity": aff,
        "tags": wb_tags,
        "wandb_name": wb_name,
        "loss": round(1.0 / (1.0 + elapsed), 6),
        "accuracy": round(min(0.99, 0.5 + 0.5 * (elapsed / max(1.0, duration))), 6),
    }

    out_dir = Path("runs") / wb_name
    ensure_dir(out_dir)
    dump_json(out_dir / "metrics.json", metrics)

    print(f"[{now_utc_iso()}] Done. Wrote metrics to {out_dir/'metrics.json'}")
    print(f"Summary: {json.dumps({'wandb_name': wb_name, 'elapsed': elapsed}, sort_keys=True)}", flush=True)

if __name__ == "__main__":
    main()