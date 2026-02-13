EVAL_GRAPH_STEPS = 512
NGPUS = 4
TOLERANCE = 0.98

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl

# ------------------------------------------------------------
# Plot styling
# ------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "figure.titlesize": 20,
    }
)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
    }
)

CB_PALETTE = {
    "ParMETIS": "#648FFF",
    "EFT": "#E69F00",
    "Static": "#009E73",
    "RL": "#000000",
    "INF": "#FF0000",
    "THRESH": "#CC79A7",
}

os.makedirs(f"{NGPUS}gpus/figures/makespan", exist_ok=True)

# ------------------------------------------------------------
# Read CSVs
# ------------------------------------------------------------
df = pd.read_csv(f"{NGPUS}gpus/level_sweep_results_{EVAL_GRAPH_STEPS}.csv")
df_rl = pd.read_csv(f"{NGPUS}gpus/results_rl_{EVAL_GRAPH_STEPS}.csv")

df_rl["is_inf"] = df_rl["mapper"].str.endswith("_inf")

group_cols = ["graph", "mem", "interior", "boundary"]

best_rl = df_rl[~df_rl["is_inf"]].loc[df_rl[~df_rl["is_inf"]].groupby(group_cols)["time"].idxmin()].reset_index(drop=True)

best_rl_inf = df_rl[df_rl["is_inf"]].loc[df_rl[df_rl["is_inf"]].groupby(group_cols)["time"].idxmin()].reset_index(drop=True)

best_rl["mapper"] = "rl"
best_rl_inf["mapper"] = "rl_inf"

cols_keep = df.columns
best_rl = best_rl[cols_keep]
best_rl_inf = best_rl_inf[cols_keep]

df = pd.concat([df, best_rl, best_rl_inf], ignore_index=True)
df = df[df["mem"] >= 24e9].copy()

df["is_inf"] = df["mapper"].str.endswith("_inf")
df["base_mapper"] = df["mapper"].str.replace("_inf", "", regex=False)

groups = df.groupby(["graph", "interior", "boundary"])
shown_representative = False


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def normalize_mem(mem, pivot_mem):
    return (mem - pivot_mem) / pivot_mem * 100


def find_mapper_pivot_vs_inf(group, best_inf_plot, mapper, tolerance):
    raw = group[(group["base_mapper"] == mapper) & (~group["is_inf"])][["mem", "time"]].rename(columns={"time": "mapper_time"})

    if raw.empty:
        return None

    aligned = raw.merge(
        best_inf_plot[["mem", "best_inf_time"]],
        on="mem",
        how="inner",
    )

    if aligned.empty:
        return None

    aligned["perf"] = aligned["best_inf_time"] / aligned["mapper_time"]
    eligible = aligned[aligned["perf"] >= tolerance]

    return eligible["mem"].max() if not eligible.empty else None


def find_static_pivot_vs_inf(group, best_inf_plot, tolerance):
    raw = group[(group["base_mapper"].isin(["b4", "b2", "rc"])) & (~group["is_inf"])]

    if raw.empty:
        return None

    raw = raw.sort_values("time").groupby("mem", as_index=False).first()
    raw = raw.rename(columns={"time": "mapper_time"})

    aligned = raw.merge(
        best_inf_plot[["mem", "best_inf_time"]],
        on="mem",
        how="inner",
    )

    aligned["perf"] = aligned["best_inf_time"] / aligned["mapper_time"]
    eligible = aligned[aligned["perf"] >= tolerance]

    return eligible["mem"].max() if not eligible.empty else None


# ------------------------------------------------------------
# Plot loop
# ------------------------------------------------------------
for (graph, interior, boundary), group in groups:
    fig, ax = plt.subplots(figsize=(8, 5))

    # --------------------------------------------------------
    # INF baselines
    # --------------------------------------------------------
    inf_min = group[group["is_inf"]].groupby("mem")["time"].min().rename("inf_min_time")

    best_inf_per_mem = group[(group["is_inf"]) & (group["base_mapper"] != "rl")].groupby("mem")["time"].min().rename("best_inf_time").reset_index()

    g = group.merge(inf_min, on="mem", how="inner")
    g = g.merge(best_inf_per_mem, on="mem", how="left")
    g = g[~g["is_inf"]].copy()

    # --------------------------------------------------------
    # ParMETIS normalization reference
    # --------------------------------------------------------
    parmetis_ref = g[g["base_mapper"] == "parmetis"].groupby("mem")["time"].min().rename("parmetis_time").reset_index()

    g = g.merge(parmetis_ref, on="mem", how="inner")

    pm = g[g["base_mapper"] == "parmetis"].copy()
    eft = g[g["base_mapper"] == "eft"].copy()
    rl = g[g["base_mapper"] == "rl"].copy()

    static = g[g["base_mapper"].isin(["b4", "b2", "rc"])].sort_values("time").groupby("mem", as_index=False).first().copy()

    # --------------------------------------------------------
    # Pivot memory definition (INF vs ParMETIS)
    # --------------------------------------------------------
    best_inf_plot = g[["mem", "best_inf_time", "parmetis_time"]].drop_duplicates().sort_values("mem")

    best_inf_plot["best_inf_norm"] = best_inf_plot["best_inf_time"] / best_inf_plot["parmetis_time"]

    eligible = best_inf_plot[best_inf_plot["best_inf_norm"] > TOLERANCE]
    if eligible.empty:
        continue

    pivot_mem = eligible["mem"].max()

    parmetis_pivot_time = parmetis_ref[parmetis_ref["mem"] == pivot_mem]["parmetis_time"].iloc[0]

    for df_ in [pm, eft, static, rl]:
        df_["normalized_time"] = df_["time"] / parmetis_pivot_time

    # --------------------------------------------------------
    # Plot curves
    # --------------------------------------------------------
    if not pm.empty:
        pm = pm.sort_values("mem")
        ax.plot(
            normalize_mem(pm["mem"], pivot_mem),
            pm["normalized_time"],
            label="ParMETIS",
            linewidth=3,
            color=CB_PALETTE["ParMETIS"],
        )

    if not eft.empty:
        eft = eft.sort_values("mem")
        ax.plot(
            normalize_mem(eft["mem"], pivot_mem),
            eft["normalized_time"],
            label="EFT",
            linewidth=3,
            color=CB_PALETTE["EFT"],
        )

    if not static.empty:
        static = static.sort_values("mem")
        ax.plot(
            normalize_mem(static["mem"], pivot_mem),
            static["normalized_time"],
            label="Static",
            linewidth=3,
            color=CB_PALETTE["Static"],
        )

    if not rl.empty:
        rl = rl.sort_values("mem")
        ax.plot(
            normalize_mem(rl["mem"], pivot_mem),
            rl["normalized_time"],
            label="RL",
            linewidth=3,
            color=CB_PALETTE["RL"],
        )
        ax.set_xlim(left=-20, right=normalize_mem(rl["mem"].max(), pivot_mem))

    ax.plot(
        normalize_mem(best_inf_plot["mem"], pivot_mem),
        best_inf_plot["best_inf_time"] / parmetis_pivot_time,
        linestyle="--",
        linewidth=2,
        # label="$\\infty$",
        label="Unconst. Baseline",
        color=CB_PALETTE["INF"],
    )

    # --------------------------------------------------------
    # Pivot lines vs INF
    # --------------------------------------------------------
    pivots = {
        "ParMETIS": find_mapper_pivot_vs_inf(group, best_inf_plot, "parmetis", TOLERANCE),
        "EFT": find_mapper_pivot_vs_inf(group, best_inf_plot, "eft", TOLERANCE),
        "Static": find_static_pivot_vs_inf(group, best_inf_plot, TOLERANCE),
        "RL": find_mapper_pivot_vs_inf(group, best_inf_plot, "rl", TOLERANCE),
    }

    for name, mem in pivots.items():
        if mem is not None:
            ax.axvline(
                normalize_mem(mem, pivot_mem),
                linestyle=":",
                linewidth=3,
                alpha=0.8,
                color=CB_PALETTE[name],
            )
            if name == "RL":
                print(graph, interior, boundary, f"{mem / pivot_mem:.2f}")

    # --------------------------------------------------------
    # Axes + save
    # --------------------------------------------------------
    ax.set_xlabel("Relative Problem Size (%)")
    ax.set_ylabel("Exec. time (Normalized)")
    ax.set_ylim(bottom=0, top=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fname = f"makespan_{graph}_int{interior}_bnd{boundary}.pdf"
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(f"{NGPUS}gpus/figures/makespan", fname), bbox_inches="tight")

    if not shown_representative:
        plt.show()
        shown_representative = True
    else:
        plt.close(fig)
