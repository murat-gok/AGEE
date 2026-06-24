"""
Louvain vs Leiden robustness analysis on MetaQA-2hop.

Re-computes AGEE for BFS, Greedy, and RandomWalk agents on the first
N_QUESTIONS MetaQA-2hop subgraphs under both Louvain and Leiden community
partitions, with random_state=42. Saves per-pair results to
`analysis/results/louvain_vs_leiden.csv` and prints a summary table that
matches §7.1 of the manuscript.

Run from the project root:
    python analysis/run_partition_robustness.py [--n_questions 80]

Wall-clock budget: ~2 minutes for the default 80 questions on a single CPU.
Set --n_questions 200 to cover all 200 questions; expect ~5 minutes.
"""

import argparse
import os
import sys
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (PROJECT_ROOT, SCRIPT_DIR, os.path.join(PROJECT_ROOT, "kgqa_experiment")):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.agee import AGEECalculator, DEFAULT_CONFIG
from run_kgqa_experiment import (load_knowledge_graph, load_questions,
                                  extract_subgraph,
                                  bfs_agent, greedy_novelty_agent,
                                  random_walk_agent)


def main(n_questions: int = 80, max_hops: int = 10):
    t0 = time.time()

    kb_path = os.path.join(PROJECT_ROOT, "kgqa_experiment", "data_metaqa", "kb.txt")
    q_path = os.path.join(PROJECT_ROOT, "kgqa_experiment", "data_metaqa", "2hop_test.txt")
    G_full, edge_relations = load_knowledge_graph(kb_path)
    questions = load_questions(q_path, n=200, seed=42)[:n_questions]
    print(f"  Loaded data in {time.time()-t0:.1f}s; analyzing {len(questions)} questions", flush=True)

    agents = {"bfs": bfs_agent, "greedy": greedy_novelty_agent,
              "random_walk": random_walk_agent}

    results = []
    for qi, q in enumerate(questions):
        topic = q["topic_entity"]
        subG, _ = extract_subgraph(G_full, edge_relations, topic, n_hops=3)
        if len(subG.nodes()) < 2:
            continue

        node_list = list(subG.nodes())
        node_to_id = {n: i for i, n in enumerate(node_list)}
        G_int = nx.Graph()
        G_int.add_nodes_from(range(len(node_list)))
        for u, v in subG.edges():
            G_int.add_edge(node_to_id[u], node_to_id[v])

        calc_louv = AGEECalculator(G_int, config={**DEFAULT_CONFIG,
            "graph": {"community_method": "louvain", "resolution": 1.0,
                      "random_state": 42}})
        calc_leid = AGEECalculator(G_int, config={**DEFAULT_CONFIG,
            "graph": {"community_method": "leiden", "resolution": 1.0,
                      "random_state": 42}})

        for an, fn in agents.items():
            random.seed(42)
            out = fn(topic, subG, max_hops=max_hops)
            traj = out["trajectory"]
            if len(traj) < 2:
                continue
            traj_int = [node_to_id[n] for n in traj if n in node_to_id]
            if len(traj_int) < 2:
                continue

            r_lou = calc_louv.compute(traj_int, an)
            r_lei = calc_leid.compute(traj_int, an)

            results.append({
                "qi": qi, "agent": an, "T": len(traj),
                "K_louv": calc_louv.n_communities,
                "K_leid": calc_leid.n_communities,
                "agee_louv": r_lou.agee, "agee_leid": r_lei.agee,
                "S_louv": r_lou.coverage, "S_leid": r_lei.coverage,
                "I_louv": r_lou.info_rate, "I_leid": r_lei.info_rate,
                "E_louv": r_lou.efficiency, "E_leid": r_lei.efficiency,
            })

        if qi % 20 == 0 and qi > 0:
            print(f"  [{qi:>3}/{len(questions)}] {time.time()-t0:.0f}s elapsed",
                  flush=True)

    print(f"\n  TOTAL: {len(results)} rows in {time.time()-t0:.1f}s")

    dfr = pd.DataFrame(results)
    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "louvain_vs_leiden.csv")
    dfr.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    # ----- Summary that matches manuscript §7.1 -----
    print("\n" + "=" * 72)
    print("  Louvain vs Leiden robustness analysis (matches manuscript §7.1)")
    print("=" * 72)

    print(f"\n  K distribution: Louvain mean={dfr['K_louv'].mean():.1f}, "
          f"Leiden mean={dfr['K_leid'].mean():.1f}, "
          f"K-equal={(dfr['K_louv']==dfr['K_leid']).mean():.1%}")

    print(f"\n  Per-component agreement (Pearson r and mean |Δ|):")
    print(f"  {'Component':<12} {'r':>8} {'mean |Δ|':>10}")
    print("  " + "-" * 36)
    for col, label in [("agee", "AGEE"), ("S", "S' (cover)"),
                       ("I", "I' (info)"), ("E", "E' (eff)")]:
        r_pe, _ = stats.pearsonr(dfr[f"{col}_louv"], dfr[f"{col}_leid"])
        delta = (dfr[f"{col}_louv"] - dfr[f"{col}_leid"]).abs()
        print(f"  {label:<12} {r_pe:>8.4f} {delta.mean():>10.4f}")

    print(f"\n  Per-agent AGEE under each partition:")
    print(f"  {'Agent':<14} {'N':>5} {'AGEE Louv':>11} {'AGEE Leid':>11} {'|Δ|':>8}")
    print("  " + "-" * 50)
    for an in ["bfs", "greedy", "random_walk"]:
        sub = dfr[dfr["agent"] == an]
        al, ae = sub["agee_louv"].mean(), sub["agee_leid"].mean()
        print(f"  {an:<14} {len(sub):>5} {al:>11.4f} {ae:>11.4f} {abs(al-ae):>8.4f}")

    mean_louv = dfr.groupby("agent")["agee_louv"].mean().sort_values(ascending=False)
    mean_leid = dfr.groupby("agent")["agee_leid"].mean().sort_values(ascending=False)
    print(f"\n  Agent ranking:")
    print(f"    Louvain: {mean_louv.index.tolist()}")
    print(f"    Leiden : {mean_leid.index.tolist()}")
    print(f"    Identical? {list(mean_louv.index) == list(mean_leid.index)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_questions", type=int, default=80,
                    help="Number of MetaQA questions to analyse (default 80; "
                         "set to 200 for full coverage).")
    ap.add_argument("--max_hops", type=int, default=10)
    args = ap.parse_args()
    main(n_questions=args.n_questions, max_hops=args.max_hops)
