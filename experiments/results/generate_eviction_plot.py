EVAL_GRAPH_STEPS = 512
NGPUS = 4
TOLERANCE = 0.98
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.size": 16,  # base font size
        "axes.labelsize": 20,  # x/y label size
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
    "ParMETIS": "#648FFF",  # blue
    "EFT": "#E69F00",  # orange
    "Static": "#009E73",  # green
    "RL": "#000000",  # black
    "INF": "#FF0000",  # vermillion
    "THRESH": "#CC79A7",  # purple
}
os.makedirs("{NGPUS}gpus/figures", exist_ok=True)
table = ""
prev_cfg = None

# ------------------------------------------------------------
# Read CSV
# ------------------------------------------------------------
df = pd.read_csv(f"{NGPUS}gpus/level_sweep_results_{EVAL_GRAPH_STEPS}.csv")

# ------------------------------------------------------------
# Read RL CSV
# ------------------------------------------------------------
df_rl = pd.read_csv(f"{NGPUS}gpus/results_rl_{EVAL_GRAPH_STEPS}.csv")

# Split inf / non-inf for RL
df_rl["is_inf"] = df_rl["mapper"].str.endswith("_inf")

# ------------------------------------------------------------
# For each (graph, mem, interior, boundary),
# pick best non-inf and best inf RL mapper
# ------------------------------------------------------------
group_cols = ["graph", "mem", "interior", "boundary"]

best_rl = df_rl[~df_rl["is_inf"]].loc[df_rl[~df_rl["is_inf"]].groupby(group_cols)["time"].idxmin()].reset_index(drop=True)

best_rl_inf = df_rl[df_rl["is_inf"]].loc[df_rl[df_rl["is_inf"]].groupby(group_cols)["time"].idxmin()].reset_index(drop=True)
# ------------------------------------------------------------
# Normalize mapper names
# ------------------------------------------------------------
best_rl["mapper"] = "rl"
best_rl_inf["mapper"] = "rl_inf"

# ------------------------------------------------------------
# Keep only columns used downstream
# ------------------------------------------------------------
cols_keep = df.columns  # ensures schema match
best_rl = best_rl[cols_keep]
best_rl_inf = best_rl_inf[cols_keep]

# ------------------------------------------------------------
# Append to main dataframe
# ------------------------------------------------------------
df = pd.concat([df, best_rl, best_rl_inf], ignore_index=True)

df = df[df["mem"] >= 24e9].copy()

# Split mapper into base name and inf flag
df["is_inf"] = df["mapper"].str.endswith("_inf")
df["base_mapper"] = df["mapper"].str.replace("_inf", "", regex=False)

# Group by (graph, interior, boundary)
groups = df.groupby(["graph", "interior", "boundary"])
num_plots = len(groups)

# ------------------------------------------------------------
# Plot loop
# ------------------------------------------------------------
shown_representative = False

for (graph, interior, boundary), group in groups:
    fig, ax = plt.subplots(figsize=(8, 5))
    # --------------------------------------------------------
    # 1. For each mem, find smallest time among *_inf mappers
    # --------------------------------------------------------
    inf_min = group[group["is_inf"]].groupby("mem")["time"].min().rename("inf_min_time")

    # --------------------------------------------------------
    # Fastest INF time per memory (excluding RL)
    # --------------------------------------------------------
    best_inf_per_mem = group[(group["is_inf"]) & (group["base_mapper"] != "rl")].groupby("mem")["time"].min().rename("best_inf_time").reset_index()

    g = group.merge(inf_min, on="mem", how="inner")
    g = g.merge(best_inf_per_mem, on="mem", how="left")

    # --------------------------------------------------------
    # 2. Keep only NON-INF runs
    # --------------------------------------------------------
    g = g[~g["is_inf"]].copy()

    # --------------------------------------------------------
    # 2.5 Keep only mem values where RL exists
    # --------------------------------------------------------
    # rl_mems = g[g["base_mapper"] == "rl"]["mem"].unique()
    # g = g[g["mem"].isin(rl_mems)].copy()

    # --------------------------------------------------------
    # 3. Pointwise normalization by non-INF ParMETIS
    # --------------------------------------------------------
    parmetis_ref = g[g["base_mapper"] == "parmetis"].groupby("mem")["time"].min().rename("parmetis_time").reset_index()

    g = g.merge(parmetis_ref, on="mem", how="inner")

    # --------------------------------------------------------
    # 4. Prepare mapper-specific data
    # --------------------------------------------------------
    pm = g[g["base_mapper"] == "parmetis"]
    eft = g[g["base_mapper"] == "eft"]
    rl = g[g["base_mapper"] == "rl"]

    # static = g[g["base_mapper"].isin(["b4", "b2", "rc"])].groupby("mem")["time"].min().reset_index()
    static = g[g["base_mapper"].isin(["b4", "b2", "rc"])].sort_values("time").groupby("mem", as_index=False).first()

    # --------------------------------------------------------
    # 5 Find pivot memory (furthest mem where INF < 0.99)
    # --------------------------------------------------------
    best_inf_plot = g[["mem", "best_inf_time", "parmetis_time"]].drop_duplicates().sort_values("mem")

    best_inf_plot["best_inf_norm"] = best_inf_plot["best_inf_time"] / best_inf_plot["parmetis_time"]

    eligible = best_inf_plot[best_inf_plot["best_inf_norm"] > TOLERANCE]

    if eligible.empty:
        # Skip plot if no valid pivot
        continue

    pivot_mem = eligible["mem"].max()
    # print(f"Pivot mem for {graph}, {interior}, {boundary}: {pivot_mem}")
    # results = ",\"mem\":["

    def normalize_mem(mem):
        return (mem - pivot_mem) / pivot_mem * 100

    # ParMETIS makespan at 0% (pivot memory)
    parmetis_pivot_time = parmetis_ref[parmetis_ref["mem"] == pivot_mem]["parmetis_time"].iloc[0]
    g["makespan"] = g["time"] / parmetis_pivot_time
    parmetis_global_min = parmetis_pivot_time

    # --------------------------------------------------------
    # 6. Plot
    # --------------------------------------------------------
    ZERO_EVICT_EPS = 1e-12

    if not pm.empty:
        pm = pm.sort_values("mem")
        ax.plot(normalize_mem(pm["mem"]), pm["eviction"] / 96e9 / 4 / EVAL_GRAPH_STEPS, label="ParMETIS", linewidth=3, color=CB_PALETTE["ParMETIS"])
        zero_evict_pm = pm[pm["eviction"] <= ZERO_EVICT_EPS]
        if not zero_evict_pm.empty:
            max_zero_mem = zero_evict_pm["mem"].max()
            x_zero = normalize_mem(max_zero_mem)
            pm_max = max_zero_mem
            ax.axvline(
                x_zero,
                linestyle=":",
                linewidth=3,
                alpha=0.9,
                color=CB_PALETTE["ParMETIS"],
                zorder=2,
            )
    if not eft.empty:
        eft = eft.sort_values("mem")
        ax.plot(
            normalize_mem(eft["mem"]),
            eft["eviction"] / 96e9 / 4 / EVAL_GRAPH_STEPS,
            label="EFT",
            linewidth=3,
            color=CB_PALETTE["EFT"],
        )
        zero_evict_eft = eft[eft["eviction"] <= ZERO_EVICT_EPS]
        if not zero_evict_eft.empty:
            max_zero_mem = zero_evict_eft["mem"].max()
            x_zero = normalize_mem(max_zero_mem)

            ax.axvline(
                x_zero,
                linestyle=":",
                linewidth=3,
                alpha=0.9,
                color=CB_PALETTE["EFT"],
                zorder=2,
            )

    if not static.empty:
        static = static.sort_values("mem")
        ax.plot(normalize_mem(static["mem"]), static["eviction"] / 96e9 / 4 / EVAL_GRAPH_STEPS, label="Static", linewidth=3, color=CB_PALETTE["Static"])
        zero_evict_static = static[static["eviction"] <= ZERO_EVICT_EPS]
        if not zero_evict_static.empty:
            max_zero_mem = zero_evict_static["mem"].max()
            x_zero = normalize_mem(max_zero_mem)

            ax.axvline(
                x_zero,
                linestyle=":",
                linewidth=3,
                alpha=0.9,
                color=CB_PALETTE["Static"],
                zorder=2,
            )
    if not rl.empty:
        rl = rl.sort_values("mem")
        ax.plot(
            normalize_mem(rl["mem"]),
            rl["eviction"] / 96e9 / 4 / EVAL_GRAPH_STEPS,
            linestyle="-",
            label="RL",
            linewidth=3,
            color=CB_PALETTE["RL"],
        )
        max_rl_mem = rl["mem"].max()
        min_rl_mem = rl["mem"].min()
        ax.set_xlim(left=-20, right=normalize_mem(max_rl_mem))

        zero_evict_rl = rl[rl["eviction"] <= ZERO_EVICT_EPS]

        if not zero_evict_rl.empty:
            max_zero_mem = zero_evict_rl["mem"].max()
            x_zero = normalize_mem(max_zero_mem)
            print(graph, interior, boundary, f"{max_zero_mem / pm_max:.2f}")
            ax.axvline(
                x_zero,
                linestyle=":",
                linewidth=3,
                alpha=0.9,
                color=CB_PALETTE["RL"],
                zorder=2,
            )

    # ax.plot(
    #     normalize_mem(best_inf_plot["mem"]),
    #     best_inf_plot["best_inf_time"] / parmetis_global_min,
    #     linestyle="--",
    #     linewidth=2,
    #     label="$\\infty$",
    #     color=CB_PALETTE["INF"],
    # )

    # --------------------------------------------------------
    # 6.6 Find max mem where RL is within 1% of INF baseline
    # --------------------------------------------------------

    # Build aligned RL vs INF table using raw times
    rl_raw = group[(group["base_mapper"] == "rl") & (~group["is_inf"])][["mem", "time"]].rename(columns={"time": "rl_time"})

    inf_raw = best_inf_plot[["mem", "best_inf_time"]]

    rl_vs_inf = rl_raw.merge(inf_raw, on="mem", how="inner")

    # RL within tolerance of INF
    rl_vs_inf["perf"] = rl_vs_inf["best_inf_time"] / rl_vs_inf["rl_time"]

    eligible_rl = rl_vs_inf[rl_vs_inf["perf"] >= TOLERANCE]

    rl_pivot_mem = eligible_rl["mem"].max() if not eligible_rl.empty else None

    # print(
    #     f"{graph}, {interior}, {boundary}, {rl_vs_inf[rl_vs_inf['mem']==rl_pivot_mem]['perf']}"
    # )
    # print(rl_vs_inf[rl_vs_inf['mem']==rl_pivot_mem])

    # --------------------------------------------------------
    # 6.7 Draw RL≈INF threshold line
    # --------------------------------------------------------
    # if rl_pivot_mem is not None:
    #     ax.axvline(
    #         normalize_mem(rl_pivot_mem),
    #         linestyle="-.",
    #         linewidth=2,
    #         alpha=0.8,
    #         # label="RL within 1% of INF",
    #         color=CB_PALETTE["THRESH"],
    #     )
    #     rl_x = normalize_mem(rl_pivot_mem)
    #     rl_x_rounded = int(round(rl_x))

    # ax.annotate(
    #     f"{rl_x_rounded}%",
    #     xy=(rl_x, 0),
    #     xycoords=("data", "axes fraction"),
    #     xytext=(4, 205),  # right & slightly up
    #     textcoords="offset points",
    #     ha="left",
    #     va="bottom",
    #     fontsize=9,
    #     color="black",
    # )

    # --------------------------------------------------------
    # 7. Axes formatting
    # --------------------------------------------------------
    # ax.set_title(f"Graph={graph}, Int={interior}, Bnd={boundary}")
    ax.set_xlabel("Relative Problem Size (%)")
    ax.set_ylabel("Eviction Volume (Normalized)")
    ax.set_ylim(bottom=0, top=0.2)
    # if graph == "corners":
    #     ax.set_xlim(right=normalize_mem(rl_pivot_mem) + 10)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fname = f"eviction_{graph}_int{interior}_bnd{boundary}.pdf"
    fpath = os.path.join("{NGPUS}gpus/figures", fname)

    fig.tight_layout(pad=0.2)
    fig.savefig(fpath, bbox_inches="tight")
    # if not shown_representative:
    #     plt.show()  # show exactly one figure
    #     shown_representative = True
    # else:
    #     plt.close(fig)  # silently save the rest

    # if (interior, boundary) == (1, 0.1):
    #     for rel_mem in [-20, -10, 0, 10, 20]:
    #         makespan = []
    #         dm = []
    #         evict=[]
    #         test_mem = pivot_mem * (1 + rel_mem / 100)

    #         # Find closest mem in INF baseline
    #         closest_mem = best_inf_plot.loc[
    #             (best_inf_plot["mem"] - test_mem).abs().idxmin(),
    #             "mem",
    #         ]

    #         # Fetch the corresponding INF run from original group
    #         closest = group[
    #             (group["is_inf"]) & (group["mem"] == closest_mem)
    #         ].iloc[0]

    #         mem_val = closest["mem"]
    #         inf_time = closest["time"]
    #         inf_data_movement = closest["data_movement"]
    #         inf_data_movement = 96e9 * EVAL_GRAPH_STEPS * 4 / 100
    #         gpu_size = 96e9 * 4 * EVAL_GRAPH_STEPS / 100
    #         makespan.append(1.0)
    #         dm.append(closest["data_movement"] / inf_data_movement)
    #         evict.append(0)

    #         rl_row = group[
    #             (group["base_mapper"] == "rl") & (~group["is_inf"]) & (group["mem"] == mem_val)
    #         ]
    #         rl_time = rl_row["time"].min()
    #         makespan.append(rl_time / inf_time)
    #         dm.append(rl_row["data_movement"].min() / inf_data_movement)
    #         evict.append(rl_row["eviction"].min() / gpu_size)

    #         parmetis_row = group[(group["base_mapper"] == "parmetis") & (~group["is_inf"]) & (group["mem"] == mem_val)]
    #         makespan.append(parmetis_row["time"].min() / inf_time)
    #         dm.append(parmetis_row["data_movement"].min() / inf_data_movement)
    #         evict.append(parmetis_row["eviction"].min() / gpu_size)

    #         static_row = group[
    #             (group["base_mapper"].isin(["b4", "b2", "rc"])) & (~group["is_inf"]) & (group["mem"] == mem_val)
    #         ]
    #         makespan.append(static_row["time"].min() / inf_time)
    #         dm.append(static_row["data_movement"].min() / inf_data_movement)
    #         evict.append(static_row["eviction"].min() / gpu_size)

    #         def fmt_list(xs, decimals=2):
    #             if decimals == 2:
    #                 return "[" + ", ".join(f"{x:.2f}" if x >0.0 else "-" for x in xs) + "]"
    #             elif decimals == 1:
    #                 return "[" + ", ".join(f"{x:.1f}" if x >0.0 else "-" for x in xs) + "]"
    #         print(f"{graph}, {interior}, {boundary}, " f"{rel_mem}%, makespan={fmt_list(makespan)}, " f"data_movement={fmt_list(dm, 1)}, eviction={fmt_list(evict)}")


# print(table + " \\\\")
