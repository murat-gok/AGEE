"""
Regenerate fig5_multi_llm_dualmode.{pdf,png} from merged real+synthetic data.

Reads analysis/results/multi_llm_merged_summary.csv (produced by
merge_real_llm.py) and writes the updated figure.

The figure distinguishes 'measured' from 'estimated' markers via edge
color (black for measured, gray for estimated).
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

MERGED_CSV = os.path.join(PROJECT_ROOT, "analysis", "results",
                           "multi_llm_merged_summary.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "paper", "figures",
                         "fig5_multi_llm_dualmode")


def main():
    if not os.path.exists(MERGED_CSV):
        print(f"  [FAIL] Not found: {MERGED_CSV}")
        print(f"  Run merge_real_llm.py first.")
        sys.exit(1)

    df = pd.read_csv(MERGED_CSV)
    print(f"  Loaded {len(df)} rows from {MERGED_CSV}")

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Sort by memory rate ascending for left panel
    df_sorted = df.sort_values("memory_rate").reset_index(drop=True)
    colors = ["#117733", "#88CCEE", "#DDCC77", "#CC6677", "#882255"]
    markers = ["o", "s", "D", "^", "v"]

    # Left panel: memory rate bars with measured/estimated edge color
    edge_colors = ["black" if r["source"] == "measured" else "gray"
                    for _, r in df_sorted.iterrows()]
    edge_widths = [1.5 if r["source"] == "measured" else 0.5
                    for _, r in df_sorted.iterrows()]

    bars = ax1.barh(
        df_sorted["llm"], df_sorted["memory_rate"],
        xerr=[
            df_sorted["memory_rate"] - df_sorted["memory_rate_lo"],
            df_sorted["memory_rate_hi"] - df_sorted["memory_rate"]
        ],
        color=[colors[i] for i in range(len(df_sorted))],
        alpha=0.85,
        error_kw={"linewidth": 0.8}
    )
    for bar, ec, ew in zip(bars, edge_colors, edge_widths):
        bar.set_edgecolor(ec)
        bar.set_linewidth(ew)

    ax1.set_xlabel("Memory mode rate")
    ax1.set_xlim(0, 1)
    ax1.axvline(0.5, color="gray", lw=0.5, linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.2, axis="x", linestyle="--", linewidth=0.4)

    # Right panel: Hits@1 vs AGEE scatter
    for i, (_, r) in enumerate(df_sorted.iterrows()):
        ec = "black" if r["source"] == "measured" else "gray"
        ew = 1.2 if r["source"] == "measured" else 0.6
        ax2.scatter(r["hits_overall"], r["agee_traversal"],
                    c=colors[i], marker=markers[i], s=90,
                    edgecolors=ec, linewidths=ew,
                    label=r["llm"], zorder=3)
        ax2.errorbar(
            r["hits_overall"], r["agee_traversal"],
            xerr=[[r["hits_overall"] - r["hits_overall_lo"]],
                  [r["hits_overall_hi"] - r["hits_overall"]]],
            yerr=[[r["agee_traversal"] - r["agee_lo"]],
                  [r["agee_hi"] - r["agee_traversal"]]],
            fmt="none", ecolor="gray", elinewidth=0.6, capsize=2,
            alpha=0.6, zorder=2
        )

    ax2.set_xlabel("Hits@1 (overall)")
    ax2.set_ylabel(r"AGEE (traversal mode)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.4, 0.75)
    ax2.legend(loc="upper left", fontsize=7, frameon=False,
               handletextpad=0.3, borderpad=0.3)
    ax2.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)

    plt.tight_layout()
    plt.savefig(OUT_PATH + ".pdf", bbox_inches="tight")
    plt.savefig(OUT_PATH + ".png", bbox_inches="tight", dpi=200)
    print(f"  Wrote {OUT_PATH}.pdf and .png")


if __name__ == "__main__":
    main()
