"""
Figure 3 (manuscript): Dirichlet weight-perturbation stability.

Sample 1000 weight vectors w = (w_S, w_I, w_E) from a Dirichlet(8, 7, 5)
prior (mode at the manuscript default (0.40, 0.35, 0.25)). For each
sample, recompute the AGEE composite for the 681 valid MetaQA-2hop
trajectories using the stored S', I', E' values and that w. Determine
which agent ranks highest. Report top-1 stability across the 1000 samples.

The figure shows a ternary scatter where each point is a sample weight,
colored by which agent it identifies as the winner.

Inputs:  kgqa_experiment/results/kgqa_trajectories.csv
<<<<<<< HEAD
Outputs: tkde/figures/fig3_dirichlet_stability.{pdf,png}
=======
Outputs: paper/figures/fig3_dirichlet_stability.{pdf,png}
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Run from project root:
    python analysis/figures/make_fig3_dirichlet_stability.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    "llm_react":   "#D62728",
    "bfs":         "#1F77B4",
    "greedy":      "#2CA02C",
    "random_walk": "#9467BD",
}
AGENT_LABELS = {
    "llm_react":   "LLM-ReAct",
    "bfs":         "BFS",
    "greedy":      "Greedy",
    "random_walk": "Random-Walk",
}

EPSILON = 0.01
P = 0.5


def weighted_power_mean(components, weights, p=P, eps=EPSILON):
    """AGEE's weighted power mean — same as in core/metrics.py."""
    c = np.maximum(np.asarray(components), eps)
    return (np.sum(weights * c ** p)) ** (1.0 / p)


def main():
    df = pd.read_csv(CSV_PATH)
    df["skip_reason"] = df["skip_reason"].fillna("")
    valid = df[df["skip_reason"] == ""].copy()
    print(f"Loaded {len(valid)} valid rows")

    # Per-agent component means
    per_agent = valid.groupby("agent")[["coverage", "info_rate", "efficiency"]].mean()
    print("\nMean component values per agent:")
    print(per_agent)

    # Dirichlet sweep
    rng = np.random.default_rng(42)
    N_SAMPLES = 1000
    # Mode-matched concentration: alpha = 8 at default (0.40), 7 at (0.35), 5 at (0.25)
    alpha = np.array([8.0, 7.0, 5.0])
    samples = rng.dirichlet(alpha, size=N_SAMPLES)
    print(f"\nMean sampled weight: {samples.mean(axis=0).round(3)}")

    # For each sample, compute AGEE for each agent and find the winner
    winners = []
    for w in samples:
        agee_per_agent = {
            agent: weighted_power_mean(per_agent.loc[agent].values, w)
            for agent in per_agent.index
        }
        winner = max(agee_per_agent, key=agee_per_agent.get)
        winners.append(winner)

    from collections import Counter
    win_counts = Counter(winners)
    print(f"\nWinner counts across {N_SAMPLES} Dirichlet samples:")
    for agent, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {agent}: {count} ({100*count/N_SAMPLES:.1f}%)")

    top_winner, top_count = win_counts.most_common(1)[0]
    print(f"\nTop-1 stability: {100 * top_count / N_SAMPLES:.1f}% (winner: {top_winner})")

    # ----- Visualization: ternary plot -----
    # Each sample w = (w_S, w_I, w_E) sums to 1, so it's a point in a triangle.
    # Project onto 2D: x = w_I + 0.5 w_E, y = (sqrt(3)/2) w_E
    x = samples[:, 1] + 0.5 * samples[:, 2]
    y = (np.sqrt(3) / 2) * samples[:, 2]
    colors = [AGENT_COLORS[w] for w in winners]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=150)

    # Draw the triangle outline
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", lw=1.2)

    # Plot samples
    ax.scatter(x, y, c=colors, s=18, alpha=0.6, edgecolors="white", linewidths=0.3)

    # Mark the manuscript default w = (0.40, 0.35, 0.25)
    w_def = np.array([0.40, 0.35, 0.25])
    x_def = w_def[1] + 0.5 * w_def[2]
    y_def = (np.sqrt(3) / 2) * w_def[2]
    ax.plot(x_def, y_def, marker="*", color="black",
            markersize=18, markeredgecolor="white", markeredgewidth=1.5,
            zorder=5, label="Default w=(0.40, 0.35, 0.25)")

    # Vertex labels: S' at origin, I' at right, E' at top
    ax.text(-0.04, -0.04, r"$w_{S'}=1$", fontsize=10, ha="right", va="top")
    ax.text(1.04, -0.04, r"$w_{I'}=1$", fontsize=10, ha="left", va="top")
    ax.text(0.5, np.sqrt(3)/2 + 0.03, r"$w_{E'}=1$", fontsize=10, ha="center", va="bottom")

    # Legend with winner counts
    handles = []
    for agent, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / N_SAMPLES
        handles.append(Patch(facecolor=AGENT_COLORS[agent], edgecolor="white",
                             label=f"{AGENT_LABELS[agent]} ({count}, {pct:.1f}%)"))
    handles.append(plt.Line2D([0], [0], marker="*", color="black", markersize=12,
                              linestyle="none", markeredgecolor="white",
                              label="Default weights"))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              title="Top-1 winner", title_fontsize=9, framealpha=0.92)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.10, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.5, -0.10,
            f"N = {N_SAMPLES} Dirichlet samples; top-1 stability {100*top_count/N_SAMPLES:.1f}%.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            style="italic", color="dimgray")

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "fig3_dirichlet_stability.pdf")
    png_path = os.path.join(OUT_DIR, "fig3_dirichlet_stability.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
