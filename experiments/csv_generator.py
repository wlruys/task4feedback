#!/usr/bin/env python3
import sys
import os
import pandas as pd
from pathlib import Path
import math


def merge_csvs_in_dir(base_dir: str):
    base_path = Path(base_dir).resolve()
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: {base_path} is not a valid directory.")
        return

    csv_files = list(base_path.rglob("results.csv"))
    if not csv_files:
        print(f"No CSV files found in {base_path}.")
        return

    print(f"Found {len(csv_files)} CSV files. Merging...")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "RL" in df.columns and "BestPolicy" in df.columns:
                df["vsBest"] = df["RL"] / df["BestPolicy"]
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if not dfs:
        print("No valid CSV files to merge.")
        return

    merged_df = pd.concat(dfs, ignore_index=True)

    # ------------------------
    # Custom sorting sequence
    # ------------------------
    graph_order = {"circle": 0, "corners": 1, "bump": 2}

    # Ensure Memory sort numerically (strip “GB” and convert)
    def parse_mem(x):
        try:
            return float(str(x).replace("GB", "").strip())
        except ValueError:
            return 0.0

    merged_df["GraphOrder"] = merged_df["Graph"].map(graph_order)
    merged_df["MemoryVal"] = merged_df["Memory"].apply(parse_mem)

    merged_df = merged_df.sort_values(
        by=["GraphOrder", "Interior", "Boundary", "MemoryVal"],
        ascending=[True, True, True, True],
        ignore_index=True,
    )

    merged_df = merged_df.drop(columns=["GraphOrder", "MemoryVal"], errors="ignore")

    # Format all float columns to .2f
    for col in merged_df.select_dtypes(include=["float", "float64", "float32"]).columns:
        merged_df[col] = merged_df[col].map(lambda x: f"{x:.2f}")

    output_path = base_path / "merged_output.csv"
    merged_df.to_csv(output_path, index=False)

    print(f"Merged and sorted CSV saved at: {output_path}")
    return output_path


def expand_from_flat_csv(csv_path: str):
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Drop vsBest if exists
    if "vsBest" in df.columns:
        df = df.drop(columns=["vsBest"])

    # ------------------------
    # Apply same ordering logic as merge_csvs_in_dir
    # ------------------------
    graph_order = {"circle": 0, "corners": 1, "bump": 2}

    def parse_mem(mem):
        try:
            return float(str(mem).replace("GB", "").strip())
        except Exception:
            return 0.0

    df["GraphOrder"] = df["Graph"].map(graph_order)
    df["MemoryVal"] = df["Memory"].apply(parse_mem)

    # Format all float columns to .2f
    for col in df.select_dtypes(include=["float", "float64", "float32"]).columns:
        df[col] = df[col].map(lambda x: f"{x:.2f}")

    # Sort by same key order
    df = df.sort_values(
        ["GraphOrder", "Interior", "Boundary", "MemoryVal"],
        ascending=[True, True, True, True],
        ignore_index=True,
    )

    # ------------------------
    # Build expanded table with ordered memory columns
    # ------------------------
    rows = []
    for (graph, interior, boundary), group in df.groupby(["Graph", "Interior", "Boundary"], sort=False):
        row_dict = {
            "Graph": graph,
            "Interior": interior,
            "Boundary": boundary,
        }

        # Sort memory values numerically within the group
        group = group.sort_values("MemoryVal")

        for _, r in group.iterrows():
            mem = str(r["Memory"]).strip()
            rl_val = r.get("RL", "")
            best_val = r.get("BestPolicy", "")
            row_dict[mem] = f"{rl_val}({best_val})"
        rows.append(row_dict)

    expanded_df = pd.DataFrame(rows)

    # Reorder columns so memory columns appear in increasing order left-to-right
    fixed_cols = ["Graph", "Interior", "Boundary"]
    memory_cols = sorted(
        [c for c in expanded_df.columns if c not in fixed_cols],
        key=lambda x: parse_mem(x),
    )
    expanded_df = expanded_df[fixed_cols + memory_cols]

    out_path = csv_path.parent / "merged_summary_expanded.csv"
    expanded_df.to_csv(out_path, index=False)
    print(f"Expanded summary saved to: {out_path}")
    return out_path


def csv_to_latex_table(csv_path: str):
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # ------------------------
    # Helpers
    # ------------------------
    graph_order = {"circle": 0, "corners": 1, "bump": 2}
    display_graph = {"circle": "Circle", "corners": "Diagonal", "bump": "Bump"}

    allowed_vals = [0.1, 1.0, 10.0, 100.0]
    EPS = 1e-6

    def parse_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return float("nan")

    def close_to_allowed(v: float) -> bool:
        if math.isnan(v):
            return False
        return any(abs(v - a) <= EPS for a in allowed_vals)

    def snap_to_allowed(v: float) -> float:
        best = min(allowed_vals, key=lambda a: abs(a - v))
        return best if abs(best - v) <= EPS else float("nan")

    def parse_mem(x):
        try:
            return float(str(x).replace("GB", "").strip())
        except Exception:
            return 0.0

    def fmt_ib(v: float) -> str:
        if abs(v - 0.1) <= EPS:
            return "0.1"
        if abs(v - 1.0) <= EPS:
            return "1.0"
        if abs(v - 10.0) <= EPS:
            return "10"
        if abs(v - 100.0) <= EPS:
            return "100"
        return f"{v:g}"

    def two_dec(x):
        try:
            return float(f"{float(str(x)):.2f}")
        except Exception:
            return x

    # ------------------------
    # Superscript Mapping Logic
    # ------------------------
    superscripts = {
        "ParMETIS": r"\textsuperscript{P}",
        "BlockCyclic": r"\textsuperscript{B}",
        "EFT": r"\textsuperscript{E}",
        "RowCyclic": r"\textsuperscript{R}",
    }

    def get_superscript(name: str) -> str:
        if not isinstance(name, str):
            return ""
        for key, sup in superscripts.items():
            if key in name:
                return sup
        return ""

    # ------------------------
    # Preprocess
    # ------------------------
    df["GraphOrder"] = df["Graph"].map(graph_order)
    df["InteriorVal"] = df["Interior"].apply(parse_float).apply(snap_to_allowed)
    df["BoundaryVal"] = df["Boundary"].apply(parse_float).apply(snap_to_allowed)
    df = df[df["InteriorVal"].apply(close_to_allowed)]
    df = df[df["BoundaryVal"].apply(close_to_allowed)]
    df["MemoryVal"] = df["Memory"].apply(parse_mem)

    df = df.sort_values(
        ["GraphOrder", "Graph", "InteriorVal", "BoundaryVal", "MemoryVal"],
        ascending=[True, True, True, True, True],
        ignore_index=True,
    )

    # ------------------------
    # Build LaTeX lines
    # ------------------------
    latex_lines = []

    for graph_key in ["circle", "corners", "bump"]:
        gdf = df[df["Graph"] == graph_key]
        if gdf.empty:
            continue

        unique_pairs = gdf[["InteriorVal", "BoundaryVal"]].drop_duplicates()
        n_rows = len(unique_pairs)

        latex_lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{display_graph[graph_key]}}} ")

        for idx, (ival, bval) in enumerate(unique_pairs.itertuples(index=False, name=None)):
            sub = gdf[(gdf["InteriorVal"] == ival) & (gdf["BoundaryVal"] == bval)]
            sub = sub.sort_values("MemoryVal")

            values = []
            for _, r in sub.iterrows():
                rl_val = two_dec(r["RL"])
                best_val = two_dec(r["BestPolicy"])
                sup = get_superscript(r.get("BestPolicyName", ""))

                # Highlight if RL is 0.05 less than BestPolicy
                cell_prefix = ""
                if isinstance(rl_val, float) and isinstance(best_val, float):
                    if rl_val + 0.05 < best_val:
                        cell_prefix = r"\cellcolor{gray!25} "

                formatted = f"{cell_prefix}{rl_val:.2f} ({best_val:.2f}{sup})"
                values.append(formatted)

            row = f"& {fmt_ib(ival)} & {fmt_ib(bval)} & " + " & ".join(values) + " \\\\"
            latex_lines.append(row)

        latex_lines.append("\\midrule\n")

    if latex_lines:
        while latex_lines and latex_lines[-1].strip() == "":
            latex_lines.pop()
        if latex_lines and latex_lines[-1].strip() == "\\midrule":
            latex_lines[-1] = "\\bottomrule"

    latex_output = "\n".join(latex_lines)

    # ------------------------
    # Save and Display
    # ------------------------
    out_path = csv_path.parent / "table_output.tex"
    with open(out_path, "w") as f:
        f.write(latex_output)

    print(f"LaTeX table saved to: {out_path}\n")
    print(latex_output)
    print("\nLegend:\n" "ParMETIS\\textsuperscript{§}, BlockCyclic\\textsuperscript{‡}, " "EFT\\textsuperscript{*}, and RowCyclic\\textsuperscript{†}\n" "(Gray cells: RL is ≥0.05 below BestPolicy)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_csvs.py <directory_path>")
        sys.exit(1)
    csv_path = merge_csvs_in_dir(sys.argv[1])
    expand_from_flat_csv(Path(sys.argv[1]) / "merged_output.csv")
    if csv_path:
        csv_to_latex_table(csv_path)
