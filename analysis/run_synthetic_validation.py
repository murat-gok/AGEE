"""
Sprint 1 Week 2 Task 4 — synthetic graph validation (manuscript §5.4).

Runs 4 agents on 6 synthetic topologies, 30 trials per (graph, agent),
computes AGEE composite, and reports per-agent means with bootstrap 95%
CIs. Output goes to:
  - analysis/results/synthetic_validation.csv   (raw data, 720 rows)
  - analysis/results/synthetic_validation_summary.csv (4x6 summary)
<<<<<<< HEAD
  - tkde/figures/tab_synthetic.tex   (LaTeX-ready table for the manuscript)
=======
  - paper/figures/tab_synthetic.tex   (LaTeX-ready table for the manuscript)
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Run from project root:
    python analysis/run_synthetic_validation.py
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.agee import AGEECalculator, DEFAULT_CONFIG
from agents.agents import create_agent
from data.synthetic import create_synthetic_graphs


N_RUNS = 30
SEED_BASE = 0  # per-run seeds: SEED_BASE + run_idx
GRAPH_LIST = [
    "erdos_renyi", "barabasi_albert", "watts_strogatz",
    "stochastic_block", "karate_club", "les_miserables"
]
GRAPH_LABELS = {
    "erdos_renyi": "Erd\\H{o}s-R\\'enyi",
    "barabasi_albert": "Barab\\'asi-Albert",
    "watts_strogatz": "Watts-Strogatz",
    "stochastic_block": "Stochastic Block",
    "karate_club": "Karate Club",
    "les_miserables": "Les Mis\\'erables",
}
AGENTS = ["bfs", "greedy", "mcts", "random_walk"]
AGENT_LABELS = {
    "bfs": "BFS",
    "greedy": "Greedy",
    "mcts": "MCTS",
    "random_walk": "Random Walk",
}


def bootstrap_ci(values, statistic=np.mean, n_iter=1000, alpha=0.05, rng=None):
    """Standard non-parametric bootstrap CI for a scalar statistic."""
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return np.nan, np.nan
    boots = np.array([statistic(rng.choice(a, size=len(a), replace=True))
                      for _ in range(n_iter)])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    print("=" * 72)
    print("  Synthetic graph validation (manuscript §5.4)")
    print("=" * 72)

    t0 = time.time()
    graphs = create_synthetic_graphs(seed=42)
    print(f"\n  Loaded {len(graphs)} synthetic graphs:")
    for name in GRAPH_LIST:
        G = graphs[name]
        print(f"    {GRAPH_LABELS[name]:<22} n={len(G.nodes()):>4}, m={len(G.edges()):>5}")

    rows = []
    for gname in GRAPH_LIST:
        G = graphs[gname]
        calc = AGEECalculator(G, config=DEFAULT_CONFIG, graph_name=gname)
        K = calc.n_communities
        max_steps = min(100, len(G.nodes()))

        for agent in AGENTS:
            for run in range(N_RUNS):
                kw = {"n_simulations": 20} if agent == "mcts" else {}
                ag = create_agent(agent, G, calc.partition,
                                  max_steps=max_steps,
                                  seed=SEED_BASE + run, **kw)
                traj = ag.walk()
                if len(traj) < 2:
                    continue
                r = calc.compute(traj, agent)
                rows.append({
                    "graph": gname, "agent": agent, "run": run,
                    "K": K, "n": len(G.nodes()), "m": len(G.edges()),
                    "agee": r.agee, "S": r.coverage, "I": r.info_rate,
                    "E": r.efficiency, "T": len(traj),
                })
        print(f"    {gname:<20} K={K:<3}  done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    print(f"\n  Total rows: {len(df)} in {time.time()-t0:.0f}s")

    # ---- Save raw data ----
    out_dir = os.path.join(PROJECT_ROOT, "analysis", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "synthetic_validation.csv"), index=False)

    # ---- Summary with bootstrap CIs ----
    summary = []
    for gname in GRAPH_LIST:
        for agent in AGENTS:
            sub = df[(df["graph"] == gname) & (df["agent"] == agent)]
            if len(sub) == 0:
                continue
            agee_mean = sub["agee"].mean()
            agee_lo, agee_hi = bootstrap_ci(sub["agee"].values)
            summary.append({
                "graph": gname, "agent": agent, "n_runs": len(sub),
                "agee_mean": agee_mean, "agee_ci_lo": agee_lo, "agee_ci_hi": agee_hi,
                "S_mean": sub["S"].mean(), "I_mean": sub["I"].mean(),
                "E_mean": sub["E"].mean(),
            })
    summ_df = pd.DataFrame(summary)
    summ_df.to_csv(os.path.join(out_dir, "synthetic_validation_summary.csv"), index=False)

    # ---- Print summary table ----
    print("\n  Summary (AGEE composite mean [95% bootstrap CI]):")
    print(f"  {'Graph':<22} {'Agent':<12} {'AGEE':>8} {'CI low':>8} {'CI high':>8}")
    print("  " + "-" * 65)
    for r in summary:
        print(f"  {GRAPH_LABELS[r['graph']]:<22} {AGENT_LABELS[r['agent']]:<12} "
              f"{r['agee_mean']:>8.3f} {r['agee_ci_lo']:>8.3f} {r['agee_ci_hi']:>8.3f}")

    # ---- Per-graph winning agent ----
    print("\n  Top-1 agent per graph (highest mean AGEE):")
    for gname in GRAPH_LIST:
        sub = summ_df[summ_df["graph"] == gname]
        winner = sub.loc[sub["agee_mean"].idxmax(), "agent"]
        print(f"    {GRAPH_LABELS[gname]:<22}: {AGENT_LABELS[winner]}")

    # ---- Write LaTeX table for the manuscript ----
<<<<<<< HEAD
    write_latex_table(summ_df, os.path.join(PROJECT_ROOT, "tkde", "figures",
=======
    write_latex_table(summ_df, os.path.join(PROJECT_ROOT, "paper", "figures",
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
                                            "tab_synthetic.tex"))
    print("\n  Saved CSV summary and LaTeX table.")


def write_latex_table(summ_df, out_path):
    """Generate a manuscript-ready LaTeX table from the summary DataFrame."""
    lines = []
    lines.append("% Auto-generated by analysis/run_synthetic_validation.py")
    lines.append("% Manuscript §5.4 synthetic graph validation table")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Synthetic graph validation: mean AGEE composite "
                 "across 30 trials per (graph, agent) with 95\\% bootstrap "
                 "confidence intervals. Winning agent per graph (highest AGEE) "
                 "is \\textbf{bolded}. The winning agent is topology-dependent: "
                 "BFS dominates on tree-like and scale-free graphs "
                 "(Barab\\'asi-Albert, Karate, Les Mis), Greedy on small-world "
                 "graphs (Watts-Strogatz), and Random-Walk on the "
                 "block-structured graph (Stochastic Block). AGEE thus reflects "
                 "exploration strategy rather than encoding a universal "
                 "preference.}")
    lines.append("\\label{tab:synthetic}")
    lines.append("\\begin{tabular}{l" + "c" * len(AGENTS) + "}")
    lines.append("\\toprule")
    header = "\\textbf{Graph}"
    for a in AGENTS:
        header += f" & \\textbf{{{AGENT_LABELS[a]}}}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    for gname in GRAPH_LIST:
        sub = summ_df[summ_df["graph"] == gname]
        if len(sub) == 0:
            continue
        winner = sub.loc[sub["agee_mean"].idxmax(), "agent"]
        row = GRAPH_LABELS[gname]
        for a in AGENTS:
            r = sub[sub["agent"] == a].iloc[0]
            cell = f"{r['agee_mean']:.3f} {{\\scriptsize [{r['agee_ci_lo']:.3f}, {r['agee_ci_hi']:.3f}]}}"
            if a == winner:
                cell = f"\\textbf{{{cell}}}"
            row += " & " + cell
        lines.append(row + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
