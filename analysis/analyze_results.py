"""
<<<<<<< HEAD
Tables 3, 4, and §5.2 supporting analysis for the AGEE TKDE manuscript.
=======
Tables 3, 4, and §5.2 supporting analysis for the AGEE manuscript.
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Reads the complete N=200 trajectory dataset and produces:
  - Table 3: per-agent summary with mode-conditional Hits@1
  - Table 4: point-biserial correlations with bootstrap CIs and BH-FDR p-values
  - §5.2 statistics: confusion matrix, chi-square, S' distribution, mode discriminant

Usage:
    python analyze_results.py
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest

# Resolve project root regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# analysis/ is a child of the project root, just like kgqa_experiment/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (PROJECT_ROOT, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

CSV_PATH = os.path.join(PROJECT_ROOT, "kgqa_experiment", "results",
                        "kgqa_trajectories.csv")


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


def bootstrap_corr_ci(x, y, n_iter=1000, alpha=0.05, rng=None):
    """Bootstrap CI for point-biserial correlation."""
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x); y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    boots = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        if len(set(x[idx])) > 1 and np.std(y[idx]) > 0:
            r, _ = stats.pointbiserialr(x[idx], y[idx])
            boots.append(r)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return lo, hi


def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def main():
    print("=" * 72)
<<<<<<< HEAD
    print("  AGEE TKDE — Tables 3, 4, and §5.2 statistics")
=======
    print("  AGEE — Tables 3, 4, and §5.2 statistics")
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
    print("=" * 72)

    df = pd.read_csv(CSV_PATH)
    df["skip_reason"] = df["skip_reason"].fillna("")
    df["valid"] = (df["skip_reason"] == "").astype(int)

    print(f"\n  Loaded: {len(df)} rows, {df['question_id'].nunique()} questions, "
          f"{df['agent'].nunique()} agents")
    print(f"  Valid trajectories (skip_reason == ''): {df['valid'].sum()}")

    valid = df[df["valid"] == 1].copy()

    # ----- TABLE 3: per-agent summary -----
    print("\n" + "=" * 72)
    print("  TABLE 3 — Agent performance on MetaQA-2hop (N=200 questions)")
    print("=" * 72)
    cols = ["Agent", "N_attempts", "N_traversed", "Hits@1 (all)",
            "Hits@1 (traversal)", "AGEE", "S'", "I'", "E'"]
    print(f"  {cols[0]:<12} {cols[1]:>9} {cols[2]:>11} "
          f"{cols[3]:>13} {cols[4]:>18} {cols[5]:>7} {cols[6]:>6} {cols[7]:>6} {cols[8]:>6}")
    print("  " + "-" * 88)

    for agent in ["llm_react", "bfs", "greedy", "random_walk"]:
        all_rows = df[df["agent"] == agent]
        val_rows = valid[valid["agent"] == agent]
        if len(all_rows) == 0:
            continue
        h_all = all_rows["hit"].mean()
        h_val = val_rows["hit"].mean() if len(val_rows) else np.nan
        agee_m = val_rows["agee"].mean()
        s_m = val_rows["coverage"].mean()
        i_m = val_rows["info_rate"].mean()
        e_m = val_rows["efficiency"].mean()
        print(f"  {agent:<12} {len(all_rows):>9} {len(val_rows):>11} "
              f"{h_all:>13.3f} {h_val:>18.3f} "
              f"{agee_m:>7.3f} {s_m:>6.3f} {i_m:>6.3f} {e_m:>6.3f}")

    # ----- TABLE 4: correlations with FDR correction -----
    print("\n" + "=" * 72)
    print(f"  TABLE 4 — Point-biserial correlation vs Hits@1 (N={len(valid)} valid)")
    print("=" * 72)
    print(f"  {'Metric':<14} {'r_pb':>8} {'95% CI':>16} {'p (raw)':>10} "
          f"{'p (BH-FDR)':>12} {'sig':>4}")
    print("  " + "-" * 70)

    metrics = ["agee", "coverage", "info_rate", "efficiency", "usr"]
    raw_p = []
    rs = []
    cis = []
    for m in metrics:
        sub = valid.dropna(subset=[m, "hit"])
        r, p = stats.pointbiserialr(sub["hit"], sub[m])
        lo, hi = bootstrap_corr_ci(sub["hit"].values, sub[m].values, n_iter=1000)
        rs.append(r); raw_p.append(p); cis.append((lo, hi))

    bh = bh_fdr(np.array(raw_p))
    for m, r, p, p_adj, (lo, hi) in zip(metrics, rs, raw_p, bh, cis):
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        ci_str = f"[{lo:+.2f}, {hi:+.2f}]"
        print(f"  {m:<14} {r:>+8.4f} {ci_str:>16} {p:>10.3g} {p_adj:>12.3g} {sig:>4}")

    # ----- §5.2: Memory mode vs Traversal mode -----
    print("\n" + "=" * 72)
    print("  §5.2 — Memory mode vs Traversal mode (LLM ReAct)")
    print("=" * 72)

    llm = df[df["agent"] == "llm_react"].copy()
    llm["traversed"] = (llm["skip_reason"] == "").astype(int)

    print("\n  Confusion table:")
    ct = pd.crosstab(llm["traversed"].map({1: "Traversal", 0: "Memory"}),
                     llm["hit"].map({1: "Correct", 0: "Wrong"}),
                     margins=True)
    print(ct.to_string())

    mem = llm[llm["traversed"] == 0]
    trav = llm[llm["traversed"] == 1]
    mem_ci = binomtest(int(mem["hit"].sum()), len(mem)).proportion_ci()
    trav_ci = binomtest(int(trav["hit"].sum()), len(trav)).proportion_ci()

    print(f"\n  Conditional Hits@1 with 95% CI:")
    print(f"    Memory mode    (N={len(mem):>3}): "
          f"{mem['hit'].mean():.4f}  [{mem_ci.low:.3f}, {mem_ci.high:.3f}]")
    print(f"    Traversal mode (N={len(trav):>3}): "
          f"{trav['hit'].mean():.4f}  [{trav_ci.low:.3f}, {trav_ci.high:.3f}]")

    chi2, p_chi, dof, _ = stats.chi2_contingency(
        [[int(mem["hit"].sum()), len(mem) - int(mem["hit"].sum())],
         [int(trav["hit"].sum()), len(trav) - int(trav["hit"].sum())]]
    )
    print(f"\n  Chi-square test (mode × correctness):")
    print(f"    chi2 = {chi2:.3f}, dof = {dof}, p = {p_chi:.3g}")

    print(f"\n  KS test on subgraph size (mode predicts question difficulty?):")
    ks_stat, ks_p = stats.ks_2samp(trav["n_nodes_graph"], mem["n_nodes_graph"])
    print(f"    KS = {ks_stat:.4f}, p = {ks_p:.4f}")
    print(f"    (high p = no significant difficulty difference between modes)")

    # BFS performance on the same question splits
    bfs = df[df["agent"] == "bfs"]
    mem_qs = set(mem["question_id"])
    trav_qs = set(trav["question_id"])
    bfs_mem = bfs[bfs["question_id"].isin(mem_qs)]["hit"].mean()
    bfs_trav = bfs[bfs["question_id"].isin(trav_qs)]["hit"].mean()
    print(f"\n  BFS Hits@1 on the same question splits (sanity check):")
    print(f"    On 'memory' questions:    {bfs_mem:.4f}  (N={len(mem_qs)})")
    print(f"    On 'traversal' questions: {bfs_trav:.4f}  (N={len(trav_qs)})")

    print(f"\n  S' (coverage) distribution by agent (valid rows only):")
    for ag in ["llm_react", "bfs", "greedy", "random_walk"]:
        sub = valid[valid["agent"] == ag]
        print(f"    {ag:<14} mean={sub['coverage'].mean():.3f}, "
              f"std={sub['coverage'].std():.3f}, N={len(sub)}")

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
