"""
Figure 2 (manuscript): S'-I' scatter on the 681 valid MetaQA trajectories,
colored by agent and marker-shape by mode (LLM traversal/memory; non-LLM
always traversal).

Purpose: visualize the dual-mode finding alongside the agent-level
exploration pattern. The LLM-traversal cluster sits at low S' / high I'
(the focused-traversal corner); the systematic explorers spread along
the high-S' axis. This single chart conveys most of §5.

Inputs:  kgqa_experiment/results/kgqa_trajectories.csv
<<<<<<< HEAD
Outputs: tkde/figures/fig2_si_scatter.pdf
         tkde/figures/fig2_si_scatter.png
=======
Outputs: paper/figures/fig2_si_scatter.pdf
         paper/figures/fig2_si_scatter.png
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Run from project root:
    python analysis/figures/make_fig2_si_scatter.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CSV_PATH = os.path.join(PROJECT_ROOT, "kgqa_experiment", "results",
                        "kgqa_trajectories.csv")
<<<<<<< HEAD
OUT_DIR = os.path.join(PROJECT_ROOT, "tkde", "figures")
=======
OUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
os.makedirs(OUT_DIR, exist_ok=True)


AGENT_COLORS = {
    "llm_react":   "#D62728",  # red
    "bfs":         "#1F77B4",  # blue
    "greedy":      "#2CA02C",  # green
    "random_walk": "#9467BD",  # purple
}
AGENT_LABELS = {
    "llm_react":   "LLM-ReAct",
    "bfs":         "BFS",
    "greedy":      "Greedy",
    "random_walk": "Random-Walk",
}


def main():
    df = pd.read_csv(CSV_PATH)
    df["skip_reason"] = df["skip_reason"].fillna("")
    valid = df[df["skip_reason"] == ""].copy()
    valid["mode"] = "traversal"  # everyone valid is in traversal mode
    print(f"Loaded {len(valid)} valid rows")

<<<<<<< HEAD
    # Figure setup — TKDE single-column figure
=======
    # Figure setup — single-column figure for manuscript
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=150)

    # Plot per agent
    for agent in ["random_walk", "greedy", "bfs", "llm_react"]:  # back-to-front order
        sub = valid[valid["agent"] == agent]
        ax.scatter(sub["coverage"], sub["info_rate"],
                   c=AGENT_COLORS[agent],
                   s=28, alpha=0.55, edgecolors="white", linewidths=0.5,
                   label=AGENT_LABELS[agent], zorder=3)

    # Annotate the dual-mode finding: the LLM cluster is in a distinct region
    llm = valid[valid["agent"] == "llm_react"]
    ax.annotate(
        f"LLM traversal mode\n(N={len(llm)}, mean Hits@1 = 0.75)",
        xy=(llm["coverage"].mean(), llm["info_rate"].mean()),
        xytext=(0.05, 0.62),
        fontsize=8.5, color="#8B0000",
        arrowprops=dict(arrowstyle="->", color="#8B0000", lw=1.0, alpha=0.7),
    )

    # Axes
    ax.set_xlabel(r"$S'$ — Structural coverage", fontsize=10)
    ax.set_ylabel(r"$I'$ — Information gain rate", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)
    ax.tick_params(labelsize=9)

    # Legend
    legend = ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92,
                       edgecolor="gray", facecolor="white",
                       handletextpad=0.4, borderpad=0.5)
    legend.set_title("Agent", prop={"size": 8.5, "weight": "bold"})

    # Subtitle below
    ax.text(0.5, -0.18,
            f"N = {len(valid)} valid trajectories on MetaQA-2hop. The LLM occupies a "
            f"low-$S'$ / high-$I'$ region (focused traversal);\nsystematic explorers "
            "spread along high-$S'$ (exhaustive coverage).",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color="dimgray")

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "fig2_si_scatter.pdf")
    png_path = os.path.join(OUT_DIR, "fig2_si_scatter.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
