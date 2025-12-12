from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to `path` (parallel-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".tmp-{path.name}-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to `path` (parallel-safe)."""
    atomic_write_bytes(path, text.encode())


def atomic_torch_save(path: Path, obj: Any) -> None:
    """Atomically torch.save to `path` (parallel-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".tmp-{path.name}-", dir=str(path.parent))
    try:
        os.close(fd)
        torch.save(obj, tmp_name)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    """Atomically pickle.dump to `path` (parallel-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".tmp-{path.name}-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass

