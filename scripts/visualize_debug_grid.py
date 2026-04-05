#!/usr/bin/env python3
"""
Render grid-based visualizations from debug artifacts produced by:
  - DilationStateGNN debug dump (`dilation_debug*`)
  - CandidateGNNObserver shared-read dump (`shared_read_*`)

Examples:
  python scripts/visualize_debug_grid.py --debug-dir debug/dilation_debug_1
  python scripts/visualize_debug_grid.py --debug-dir debug --all-runs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="ascii", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(v: str, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _natural_key(text: str) -> list[object]:
    parts = re.split(r"(\d+)", text)
    out: list[object] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        else:
            out.append(part)
    return out


def _extract_k(path: Path) -> int:
    m = re.search(r"k(\d+)_edges\.csv$", path.name)
    return int(m.group(1)) if m else -1


def _draw_grid(ax: plt.Axes, rows: int, cols: int) -> None:
    for r in range(rows + 1):
        ax.plot([-0.5, cols - 0.5], [r - 0.5, r - 0.5], color="#dddddd", linewidth=0.8, zorder=0)
    for c in range(cols + 1):
        ax.plot([c - 0.5, c - 0.5], [-0.5, rows - 0.5], color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(labelsize=7)


def _draw_directed_edge(
    ax: plt.Axes,
    src_col: float,
    src_row: float,
    dst_col: float,
    dst_row: float,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float = 0.9,
) -> None:
    dx = dst_col - src_col
    dy = dst_row - src_row
    d = math.sqrt(dx * dx + dy * dy)
    if d <= 1e-9:
        return

    shorten = 0.22
    end_x = dst_col - shorten * dx / d
    end_y = dst_row - shorten * dy / d
    start_x = src_col + 0.08 * dx / d
    start_y = src_row + 0.08 * dy / d

    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linestyle": linestyle,
            "lw": linewidth,
            "alpha": alpha,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 8,
        },
        zorder=2,
    )


def _pick_label(row: dict[str, str], node_label: str) -> str:
    if node_label == "task":
        return row.get("task_id", "")
    if node_label == "node":
        return row.get("node_id", row.get("local_idx", ""))
    if node_label == "both":
        nid = row.get("node_id", row.get("local_idx", ""))
        tid = row.get("task_id", "")
        if tid != "":
            return f"{nid}/{tid}"
        return nid
    return ""


def _render_dilation_dir(
    run_dir: Path,
    out_dir: Path,
    *,
    node_label: str,
    point_size: float,
    show_offshell: bool,
    focus_node_id: Optional[int],
    focus_task_id: Optional[int],
) -> int:
    nodes_path = run_dir / "nodes.csv"
    if not nodes_path.exists():
        return 0

    nodes_raw = _read_csv(nodes_path)
    if len(nodes_raw) == 0:
        return 0

    nodes = []
    by_node_id = {}
    for row in nodes_raw:
        node_id = _to_int(row.get("node_id", "-1"))
        batch_id = _to_int(row.get("batch_id", "0"), 0)
        r = _to_int(row.get("row", "-1"))
        c = _to_int(row.get("col", "-1"))
        entry = {
            "node_id": node_id,
            "batch_id": batch_id,
            "task_id": _to_int(row.get("task_id", "-1")),
            "row": r,
            "col": c,
            "_raw": row,
        }
        nodes.append(entry)
        by_node_id[node_id] = entry

    valid_nodes = [n for n in nodes if n["row"] >= 0 and n["col"] >= 0]
    if len(valid_nodes) == 0:
        print(f"[skip] no grid coords in {run_dir}")
        return 0

    report_path = run_dir / "report.json"
    rows = max(n["row"] for n in valid_nodes) + 1
    cols = max(n["col"] for n in valid_nodes) + 1
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="ascii"))
            shape = report.get("grid_shape")
            if isinstance(shape, list) and len(shape) == 2:
                rows = max(rows, int(shape[0]))
                cols = max(cols, int(shape[1]))
        except Exception:
            pass

    nodes_by_batch: dict[int, list[dict[str, int]]] = defaultdict(list)
    for n in valid_nodes:
        nodes_by_batch[n["batch_id"]].append(n)

    focus_nid: Optional[int] = None
    if focus_node_id is not None:
        if focus_node_id not in by_node_id:
            raise ValueError(f"focus node_id {focus_node_id} not found in {nodes_path}")
        focus_nid = int(focus_node_id)
    elif focus_task_id is not None:
        matches = sorted(n["node_id"] for n in valid_nodes if int(n["task_id"]) == int(focus_task_id))
        if len(matches) == 0:
            raise ValueError(f"focus task_id {focus_task_id} not found in {nodes_path}")
        focus_nid = int(matches[0])

    edge_files = sorted(run_dir.glob("k*_edges.csv"), key=lambda p: (_extract_k(p), _natural_key(p.name)))
    if len(edge_files) == 0:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    produced = 0
    styles = {
        "exact": ("black", "-", 0.8),
        "missing_vs_bfs": ("red", "--", 1.2),
        "extra_vs_bfs": ("royalblue", ":", 1.2),
    }
    shell_styles = {
        "eq": ("black", "-", 0.9),
        "lt": ("orange", "--", 1.4),
        "gt": ("purple", "-.", 1.2),
    }

    for edge_file in edge_files:
        k = _extract_k(edge_file)
        edges = _read_csv(edge_file)
        edges_by_batch: dict[int, list[dict[str, str]]] = defaultdict(list)

        for e in edges:
            src = _to_int(e.get("src", "-1"))
            dst = _to_int(e.get("dst", "-1"))
            if src not in by_node_id or dst not in by_node_id:
                continue
            src_batch = by_node_id[src]["batch_id"]
            dst_batch = by_node_id[dst]["batch_id"]
            if src_batch != dst_batch:
                continue
            edges_by_batch[src_batch].append(e)

        for batch_id, batch_nodes in nodes_by_batch.items():
            if focus_nid is not None and by_node_id[focus_nid]["batch_id"] != batch_id:
                continue

            fig_w = max(6.0, 0.7 * cols)
            fig_h = max(6.0, 0.7 * rows)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            _draw_grid(ax, rows, cols)

            shell_counts = {"eq": 0, "lt": 0, "gt": 0, "unknown": 0}
            drawn_edges = 0
            for e in edges_by_batch.get(batch_id, []):
                src = _to_int(e.get("src", "-1"))
                dst = _to_int(e.get("dst", "-1"))
                status = e.get("status", "exact")
                if src not in by_node_id or dst not in by_node_id:
                    continue
                if focus_nid is not None and src != focus_nid and dst != focus_nid:
                    continue
                src_n = by_node_id[src]
                dst_n = by_node_id[dst]
                shell_rel = "unknown"
                if k > 0 and min(src_n["row"], src_n["col"], dst_n["row"], dst_n["col"]) >= 0:
                    manhattan = abs(src_n["row"] - dst_n["row"]) + abs(src_n["col"] - dst_n["col"])
                    if manhattan == k:
                        shell_rel = "eq"
                    elif manhattan < k:
                        shell_rel = "lt"
                    else:
                        shell_rel = "gt"

                shell_counts[shell_rel] = shell_counts.get(shell_rel, 0) + 1
                if not show_offshell and shell_rel in ("lt", "gt"):
                    continue

                if shell_rel in shell_styles:
                    color, linestyle, lw = shell_styles[shell_rel]
                else:
                    color, linestyle, lw = styles.get(status, ("gray", "-", 0.7))
                _draw_directed_edge(
                    ax,
                    src_n["col"],
                    src_n["row"],
                    dst_n["col"],
                    dst_n["row"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=lw,
                )
                drawn_edges += 1

            x = [n["col"] for n in batch_nodes]
            y = [n["row"] for n in batch_nodes]
            if focus_nid is None:
                ax.scatter(x, y, s=point_size, color="white", edgecolors="black", linewidths=0.8, zorder=3)
            else:
                other_nodes = [n for n in batch_nodes if n["node_id"] != focus_nid]
                fx = [by_node_id[focus_nid]["col"]]
                fy = [by_node_id[focus_nid]["row"]]
                if len(other_nodes) > 0:
                    ox = [n["col"] for n in other_nodes]
                    oy = [n["row"] for n in other_nodes]
                    ax.scatter(ox, oy, s=point_size * 0.85, color="#f2f2f2", edgecolors="#888888", linewidths=0.6, zorder=3)
                ax.scatter(fx, fy, s=point_size * 1.35, color="#ffd54f", edgecolors="black", linewidths=1.0, zorder=4)

            if node_label != "none":
                for n in batch_nodes:
                    if focus_nid is not None and n["node_id"] != focus_nid:
                        continue
                    label = _pick_label(n["_raw"], node_label)
                    if label:
                        ax.text(n["col"] + 0.06, n["row"] - 0.10, label, fontsize=6, color="black", zorder=4)

            n_exact = sum(1 for e in edges_by_batch.get(batch_id, []) if e.get("status") == "exact")
            n_miss = sum(1 for e in edges_by_batch.get(batch_id, []) if e.get("status") == "missing_vs_bfs")
            n_extra = sum(1 for e in edges_by_batch.get(batch_id, []) if e.get("status") == "extra_vs_bfs")
            focus_note = ""
            if focus_nid is not None:
                focus_note = f" focus=node:{focus_nid}/task:{by_node_id[focus_nid]['task_id']}"
            ax.set_title(
                f"{run_dir.name} | k={k} | batch={batch_id}{focus_note}\n"
                f"exact={n_exact} missing={n_miss} extra={n_extra} drawn={drawn_edges}\n"
                f"|d|==k:{shell_counts['eq']}  |d|<k:{shell_counts['lt']}  |d|>k:{shell_counts['gt']}",
                fontsize=10,
            )

            fig.tight_layout()
            out_path = out_dir / f"{run_dir.name}_k{k}_batch{batch_id}.png"
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            produced += 1

    return produced


def _render_shared_read_dir(
    run_dir: Path,
    out_dir: Path,
    *,
    node_label: str,
    point_size: float,
    focus_node_id: Optional[int],
    focus_task_id: Optional[int],
) -> int:
    cand_path = run_dir / "candidates.csv"
    edge_path = run_dir / "edges.csv"
    if not cand_path.exists() or not edge_path.exists():
        return 0

    candidates = _read_csv(cand_path)
    edges = _read_csv(edge_path)
    if len(candidates) == 0:
        return 0

    nodes: dict[int, dict[str, str]] = {}
    for row in candidates:
        idx = _to_int(row.get("local_idx", "-1"))
        if idx < 0:
            continue
        nodes[idx] = row

    valid = [row for row in nodes.values() if _to_int(row.get("row", "-1")) >= 0 and _to_int(row.get("col", "-1")) >= 0]
    if len(valid) == 0:
        print(f"[skip] no grid coords in {run_dir}")
        return 0

    focus_idx: Optional[int] = None
    if focus_node_id is not None:
        if focus_node_id not in nodes:
            raise ValueError(f"focus node_id/local_idx {focus_node_id} not found in {cand_path}")
        focus_idx = int(focus_node_id)
    elif focus_task_id is not None:
        matches = sorted(idx for idx, row in nodes.items() if _to_int(row.get("task_id", "-1")) == int(focus_task_id))
        if len(matches) == 0:
            raise ValueError(f"focus task_id {focus_task_id} not found in {cand_path}")
        focus_idx = int(matches[0])

    rows = max(_to_int(v.get("row", "-1")) for v in valid) + 1
    cols = max(_to_int(v.get("col", "-1")) for v in valid) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_w = max(6.0, 0.7 * cols)
    fig_h = max(6.0, 0.7 * rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_grid(ax, rows, cols)

    styles = {
        "match": ("black", "-", 1.0),
        "missing": ("red", "--", 1.4),
        "extra": ("royalblue", ":", 1.4),
    }

    for e in edges:
        src = _to_int(e.get("src_idx", "-1"))
        dst = _to_int(e.get("dst_idx", "-1"))
        if src not in nodes or dst not in nodes:
            continue
        if focus_idx is not None and src != focus_idx and dst != focus_idx:
            continue
        src_n = nodes[src]
        dst_n = nodes[dst]
        src_r = _to_float(src_n.get("row", "-1"), -1.0)
        src_c = _to_float(src_n.get("col", "-1"), -1.0)
        dst_r = _to_float(dst_n.get("row", "-1"), -1.0)
        dst_c = _to_float(dst_n.get("col", "-1"), -1.0)
        if min(src_r, src_c, dst_r, dst_c) < 0:
            continue
        color, linestyle, lw = styles.get(e.get("status", "match"), ("gray", "-", 0.8))
        ax.plot([src_c, dst_c], [src_r, dst_r], color=color, linestyle=linestyle, linewidth=lw, alpha=0.9, zorder=2)

    if focus_idx is None:
        x = [_to_float(v.get("col", "-1"), -1.0) for v in valid]
        y = [_to_float(v.get("row", "-1"), -1.0) for v in valid]
        ax.scatter(x, y, s=point_size, color="white", edgecolors="black", linewidths=0.8, zorder=3)
    else:
        other = [v for idx, v in nodes.items() if idx != focus_idx and _to_int(v.get("row", "-1")) >= 0 and _to_int(v.get("col", "-1")) >= 0]
        if len(other) > 0:
            ox = [_to_float(v.get("col", "-1"), -1.0) for v in other]
            oy = [_to_float(v.get("row", "-1"), -1.0) for v in other]
            ax.scatter(ox, oy, s=point_size * 0.85, color="#f2f2f2", edgecolors="#888888", linewidths=0.6, zorder=3)
        fx = _to_float(nodes[focus_idx].get("col", "-1"), -1.0)
        fy = _to_float(nodes[focus_idx].get("row", "-1"), -1.0)
        ax.scatter([fx], [fy], s=point_size * 1.35, color="#ffd54f", edgecolors="black", linewidths=1.0, zorder=4)

    if node_label != "none":
        if focus_idx is None:
            for v in valid:
                label = _pick_label(v, node_label)
                if label:
                    ax.text(_to_float(v.get("col", "-1")) + 0.06, _to_float(v.get("row", "-1")) - 0.10, label, fontsize=6, color="black", zorder=4)
        else:
            v = nodes[focus_idx]
            label = _pick_label(v, node_label)
            if label:
                ax.text(_to_float(v.get("col", "-1")) + 0.06, _to_float(v.get("row", "-1")) - 0.10, label, fontsize=6, color="black", zorder=4)

    n_match = sum(1 for e in edges if e.get("status") == "match")
    n_missing = sum(1 for e in edges if e.get("status") == "missing")
    n_extra = sum(1 for e in edges if e.get("status") == "extra")
    focus_note = ""
    if focus_idx is not None:
        focus_note = f" focus=idx:{focus_idx}/task:{_to_int(nodes[focus_idx].get('task_id', '-1'))}"
    ax.set_title(
        f"{run_dir.name} | shared read{focus_note}\n"
        f"match={n_match} missing={n_missing} extra={n_extra}",
        fontsize=10,
    )

    fig.tight_layout()
    out_path = out_dir / f"{run_dir.name}_shared_read.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return 1


def _find_run_dirs(root: Path) -> tuple[list[Path], list[Path]]:
    dilation: list[Path] = []
    shared: list[Path] = []

    if (root / "nodes.csv").exists() and len(list(root.glob("k*_edges.csv"))) > 0:
        return [root], []
    if (root / "candidates.csv").exists() and (root / "edges.csv").exists():
        return [], [root]

    for d in root.iterdir():
        if not d.is_dir():
            continue
        if re.match(r"^dilation_debug(?:_\d+)?$", d.name) and (d / "nodes.csv").exists():
            dilation.append(d)
        if re.match(r"^shared_read", d.name) and (d / "candidates.csv").exists():
            shared.append(d)

    dilation.sort(key=lambda p: _natural_key(p.name))
    shared.sort(key=lambda p: _natural_key(p.name))
    return dilation, shared


def _select_runs(runs: list[Path], *, all_runs: bool, max_runs: int) -> list[Path]:
    if len(runs) == 0:
        return []
    if all_runs:
        return runs
    return runs[-max_runs:]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize debug graph dumps on a 2D grid.")
    p.add_argument("--debug-dir", type=Path, required=True, help="Debug run dir or parent dir containing debug runs.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for images. Default: <run_dir>/grid_viz",
    )
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all discovered runs under --debug-dir. Default is latest run(s) only.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="When not using --all-runs, process up to this many latest runs per type.",
    )
    p.add_argument(
        "--node-label",
        choices=["none", "node", "task", "both"],
        default="task",
        help="Node label style.",
    )
    p.add_argument("--point-size", type=float, default=44.0, help="Node marker size.")
    p.add_argument(
        "--show-offshell",
        action="store_true",
        help="Draw edges whose Manhattan grid distance is not exactly k (for Jacobi debug).",
    )
    p.add_argument(
        "--focus-node-id",
        type=int,
        default=None,
        help="Show only edges incident to this node id (dilation) or local_idx (shared-read).",
    )
    p.add_argument(
        "--focus-task-id",
        type=int,
        default=None,
        help="Show only edges incident to this task id.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.focus_node_id is not None and args.focus_task_id is not None:
        raise ValueError("Use only one of --focus-node-id or --focus-task-id.")

    root = args.debug_dir
    if not root.exists():
        raise FileNotFoundError(f"debug directory not found: {root}")

    dilation_runs, shared_runs = _find_run_dirs(root)
    dilation_runs = _select_runs(dilation_runs, all_runs=args.all_runs, max_runs=max(1, int(args.max_runs)))
    shared_runs = _select_runs(shared_runs, all_runs=args.all_runs, max_runs=max(1, int(args.max_runs)))

    if len(dilation_runs) == 0 and len(shared_runs) == 0:
        print(f"[warn] no supported debug runs found in {root}")
        return 0

    produced = 0
    for run in dilation_runs:
        out_dir = (args.output_root / run.name) if args.output_root is not None else (run / "grid_viz")
        n = _render_dilation_dir(
            run,
            out_dir,
            node_label=args.node_label,
            point_size=float(args.point_size),
            show_offshell=bool(args.show_offshell),
            focus_node_id=args.focus_node_id,
            focus_task_id=args.focus_task_id,
        )
        produced += n
        print(f"[dilation] {run}: wrote {n} image(s) to {out_dir}")

    for run in shared_runs:
        out_dir = (args.output_root / run.name) if args.output_root is not None else (run / "grid_viz")
        n = _render_shared_read_dir(
            run,
            out_dir,
            node_label=args.node_label,
            point_size=float(args.point_size),
            focus_node_id=args.focus_node_id,
            focus_task_id=args.focus_task_id,
        )
        produced += n
        print(f"[shared-read] {run}: wrote {n} image(s) to {out_dir}")

    print(f"[done] total images written: {produced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
