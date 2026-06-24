"""
Figure 1 (manuscript): coverage-time curves per agent on a representative
MetaQA-2hop subgraph, with the closed-form ideal AUC shaded.

Purpose: visualize the E' component directly. The ideal curve represents
"discover one new node per step then plateau"; each agent's curve sits
below it. The area-ratio is E'.

Strategy: pick a single representative question (median-sized subgraph,
all four agents produce valid trajectories), re-run each algorithmic agent
deterministically, and plot |D_t|/n vs t.

Note: the LLM trajectory cannot be reproduced without re-running Ollama,
which is impractical here. We use the saved trajectory from the LLM run
if available (the current CSV doesn't have trajectory data; future runs
will, thanks to the trajectory-saving patch). For this figure we plot
BFS, Greedy, and Random-Walk only — the LLM-traversal pattern is conveyed
clearly enough in Figure 2.

Inputs:  kgqa_experiment/data_metaqa/{kb.txt, 2hop_test.txt}
<<<<<<< HEAD
Outputs: tkde/figures/fig1_coverage_curves.{pdf,png}
=======
Outputs: paper/figures/fig1_coverage_curves.{pdf,png}
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Run from project root:
    python analysis/figures/make_fig1_coverage_curves.py
"""
import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "kgqa_experiment")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_kgqa_experiment import (load_knowledge_graph, load_questions,
                                  extract_subgraph,
                                  bfs_agent, greedy_novelty_agent,
                                  random_walk_agent)

<<<<<<< HEAD
OUT_DIR = os.path.join(PROJECT_ROOT, "tkde", "figures")
=======
OUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
os.makedirs(OUT_DIR, exist_ok=True)


def coverage_curve(traj, G):
    """Cumulative |V_t|/n where V_t is the set of unique visited nodes by step t."""
    n = len(G.nodes())
    visited = set()
    curve = []
    for v in traj:
        if v in G:
            visited.add(v)
        curve.append(len(visited) / n)
    return np.array(curve)


def ideal_curve(T, n):
    """Closed-form ideal: visit one new node per step (linear)."""
    curve = np.minimum(np.arange(1, T + 1) / n, 1.0)
    return curve


def main():
    kb_path = os.path.join(PROJECT_ROOT, "kgqa_experiment", "data_metaqa", "kb.txt")
    q_path = os.path.join(PROJECT_ROOT, "kgqa_experiment", "data_metaqa", "2hop_test.txt")
    G_full, edge_relations = load_knowledge_graph(kb_path)
    questions = load_questions(q_path, n=200, seed=42)

    # Pick representative question: medium-sized subgraph (~250-450 nodes)
    representative = None
    for qi, q in enumerate(questions):
        subG, _ = extract_subgraph(G_full, edge_relations, q["topic_entity"], n_hops=3)
        if 200 < len(subG.nodes()) < 450:
            representative = (qi, q, subG)
            break
    assert representative is not None
    qi, q, subG = representative
    print(f"Selected question {qi}: {q['question'][:60]}")
    print(f"  Subgraph: {len(subG.nodes())} nodes, {len(subG.edges())} edges")
    print(f"  Topic entity: {q['topic_entity']}")

    # Run each algorithmic agent
    agents = {
        "BFS":         ("bfs",         bfs_agent,            "#1F77B4"),
        "Greedy":      ("greedy",      greedy_novelty_agent, "#2CA02C"),
        "Random-Walk": ("random_walk", random_walk_agent,    "#9467BD"),
    }

    curves = {}
    for label, (name, fn, color) in agents.items():
        random.seed(42)
        out = fn(q["topic_entity"], subG, max_hops=10)
        traj = out["trajectory"]
        curve = coverage_curve(traj, subG)
        curves[label] = (curve, color, len(set(traj)))

    # Find max trajectory length for x-axis
    T_max = max(len(c[0]) for c in curves.values())
    n = len(subG.nodes())

    # Compute ideal curve using BFS's unique count (a reasonable reference)
    n_u_ref = max(c[2] for c in curves.values())
    ideal = ideal_curve(T_max, n_u_ref, n)

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
    steps = np.arange(1, T_max + 1)

    # Shade the ideal area
    ax.fill_between(steps, 0, ideal, color="#FFE082", alpha=0.45,
                     label="Ideal AUC region", zorder=1)
    ax.plot(steps, ideal, color="#F57F17", linewidth=1.5, linestyle="--",
            label="Ideal coverage", zorder=2)

    # Plot agent curves
    for label, (curve, color, n_u) in curves.items():
        x = np.arange(1, len(curve) + 1)
        ax.plot(x, curve, color=color, linewidth=2.0,
                marker="o", markersize=4, alpha=0.9,
                label=f"{label} (T={len(curve)}, $|V_\\tau|$={n_u})",
                zorder=4)

    ax.set_xlabel("Step $t$", fontsize=10)
    ax.set_ylabel(r"Discovered fraction $|\mathcal{D}_t|/n$", fontsize=10)
    ax.set_xlim(0.5, T_max + 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
              edgecolor="gray", facecolor="white")
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "fig1_coverage_curves.pdf")
    png_path = os.path.join(OUT_DIR, "fig1_coverage_curves.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
