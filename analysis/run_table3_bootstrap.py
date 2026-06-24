"""
Sprint 1 Week 3 Task 2 — bootstrap CIs for Table 3 entries.

Extends analyze_results.py to produce a Table 3 with explicit 95%
bootstrap CIs on every reported number (Hits@1, AGEE, S', I', E').

Output:
  analysis/results/table3_with_ci.csv
<<<<<<< HEAD
  tkde/figures/tab_headline.tex  (auto-generated LaTeX table)
=======
  paper/figures/tab_headline.tex  (auto-generated LaTeX table)
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Run from project root:
    python analysis/run_table3_bootstrap.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import binomtest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

CSV_PATH = os.path.join(PROJECT_ROOT, "kgqa_experiment", "results",
                        "kgqa_trajectories.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "results")
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT = 1000
RNG_SEED = 42

AGENTS = ["llm_react", "bfs", "greedy", "random_walk"]
AGENT_LABELS = {
    "llm_react":   "LLM-ReAct",
    "bfs":         "BFS",
    "greedy":      "Greedy",
    "random_walk": "Random-Walk",
}


def bootstrap_mean_ci(values, n_boot=N_BOOT, alpha=0.05, rng=None):
    """95% percentile bootstrap CI on the mean of `values`."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return np.nan, np.nan
    boots = np.array([rng.choice(a, size=len(a), replace=True).mean()
                      for _ in range(n_boot)])
    return np.percentile(boots, 100 * alpha / 2), np.percentile(boots, 100 * (1 - alpha / 2))


def main():
    print("=" * 72)
    print("  Sprint 1 Week 3 — Table 3 with bootstrap CIs")
    print("=" * 72)

    df = pd.read_csv(CSV_PATH)
    df["skip_reason"] = df["skip_reason"].fillna("")

    rows = []
    for agent in AGENTS:
        all_rows = df[df["agent"] == agent]
        valid_rows = all_rows[all_rows["skip_reason"] == ""]
        if len(all_rows) == 0:
            continue

        n_att = len(all_rows)
        n_val = len(valid_rows)

        # Hits@1 (all) — Clopper-Pearson for proportion
        n_correct_all = int(all_rows["hit"].sum())
        ci_all = binomtest(n_correct_all, n_att).proportion_ci(confidence_level=0.95)

        # Hits@1 (traversal) — Clopper-Pearson on valid only
        n_correct_val = int(valid_rows["hit"].sum())
        ci_val = binomtest(n_correct_val, n_val).proportion_ci(confidence_level=0.95) if n_val else None

        # Bootstrap CIs for AGEE composites and components on valid rows
        rng = np.random.default_rng(RNG_SEED)
        agee_lo, agee_hi = bootstrap_mean_ci(valid_rows["agee"].values, rng=rng)
        S_lo, S_hi = bootstrap_mean_ci(valid_rows["coverage"].values, rng=rng)
        I_lo, I_hi = bootstrap_mean_ci(valid_rows["info_rate"].values, rng=rng)
        E_lo, E_hi = bootstrap_mean_ci(valid_rows["efficiency"].values, rng=rng)

        rows.append({
            "agent": agent,
            "n_att": n_att, "n_val": n_val,
            "hits_all": all_rows["hit"].mean(),
            "hits_all_lo": ci_all.low, "hits_all_hi": ci_all.high,
            "hits_val": valid_rows["hit"].mean() if n_val else np.nan,
            "hits_val_lo": ci_val.low if ci_val else np.nan,
            "hits_val_hi": ci_val.high if ci_val else np.nan,
            "agee": valid_rows["agee"].mean(), "agee_lo": agee_lo, "agee_hi": agee_hi,
            "S": valid_rows["coverage"].mean(), "S_lo": S_lo, "S_hi": S_hi,
            "I": valid_rows["info_rate"].mean(), "I_lo": I_lo, "I_hi": I_hi,
            "E": valid_rows["efficiency"].mean(), "E_lo": E_lo, "E_hi": E_hi,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(OUT_DIR, "table3_with_ci.csv"), index=False)

    print("\n  Table 3 with 95% CIs (Hits: Clopper-Pearson; AGEE/components: bootstrap):")
    print(f"  {'Agent':<14} {'N_val':>5} {'Hits@1 (all)':>26} {'Hits@1 (trav)':>26}")
    print("  " + "-" * 80)
    for r in rows:
        h_all = f"{r['hits_all']:.3f} [{r['hits_all_lo']:.3f}, {r['hits_all_hi']:.3f}]"
        h_val = (f"{r['hits_val']:.3f} [{r['hits_val_lo']:.3f}, {r['hits_val_hi']:.3f}]"
                 if not np.isnan(r['hits_val']) else "—")
        print(f"  {AGENT_LABELS[r['agent']]:<14} {r['n_val']:>5} {h_all:>26} {h_val:>26}")

    print(f"\n  {'Agent':<14} {'AGEE':>20} {'S':>16} {'I':>16} {'E':>16}")
    print("  " + "-" * 90)
    for r in rows:
        agee = f"{r['agee']:.3f} [{r['agee_lo']:.2f},{r['agee_hi']:.2f}]"
        S = f"{r['S']:.3f} [{r['S_lo']:.2f},{r['S_hi']:.2f}]"
        I = f"{r['I']:.3f} [{r['I_lo']:.2f},{r['I_hi']:.2f}]"
        E = f"{r['E']:.3f} [{r['E_lo']:.2f},{r['E_hi']:.2f}]"
        print(f"  {AGENT_LABELS[r['agent']]:<14} {agee:>20} {S:>16} {I:>16} {E:>16}")

<<<<<<< HEAD
    write_latex_headline_table(rows, os.path.join(PROJECT_ROOT, "tkde",
=======
    write_latex_headline_table(rows, os.path.join(PROJECT_ROOT, "paper",
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
                                                   "figures", "tab_headline.tex"))
    print("\n  Saved CSV and tab_headline.tex.")


def write_latex_headline_table(rows, out_path):
    """Auto-generate Table 3 with all CIs visible."""
    lines = []
    lines.append("% Auto-generated by analysis/run_table3_bootstrap.py")
    lines.append("% Manuscript Table 3 — headline, with 95% bootstrap/Clopper-Pearson CIs.")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Agent performance on MetaQA-2hop ($N{=}200$ "
                 "questions, 800 attempts). \\emph{Hits@1 (traversal)} "
                 "restricts to attempts with trajectory length $\\geq 2$; for "
                 "non-LLM agents the two figures coincide. AGEE and components "
                 "are computed on valid trajectories. All values are means "
                 "with 95\\% CIs: Hits@1 via Clopper-Pearson; AGEE and "
                 "components via 1{,}000-iteration percentile bootstrap.}")
    lines.append("\\label{tab:headline}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lrrcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Agent} & $N_\\mathrm{att}$ & $N_\\mathrm{val}$ "
                 "& \\textbf{Hits@1 (all)} & \\textbf{Hits@1 (traversal)} "
                 "& \\textbf{AGEE} & $S'$ & $I'$ & $E'$ \\\\")
    lines.append("\\midrule")

    # bold the winning AGEE cell
    agees = [r['agee'] for r in rows]
    max_agee = max(agees)

    for r in rows:
        h_all = f"{r['hits_all']:.3f} {{\\scriptsize [{r['hits_all_lo']:.2f},{r['hits_all_hi']:.2f}]}}"
        if not np.isnan(r['hits_val']):
            h_val = f"{r['hits_val']:.3f} {{\\scriptsize [{r['hits_val_lo']:.2f},{r['hits_val_hi']:.2f}]}}"
            # bold the winning Hits@1 (traversal)
            if r['agent'] == 'llm_react':
                h_val = f"\\textbf{{{h_val}}}"
        else:
            h_val = "---"

        agee_cell = f"{r['agee']:.3f} {{\\scriptsize [{r['agee_lo']:.2f},{r['agee_hi']:.2f}]}}"
        if r['agee'] == max_agee:
            agee_cell = f"\\textbf{{{agee_cell}}}"

        S_cell = f"{r['S']:.3f} {{\\scriptsize [{r['S_lo']:.2f},{r['S_hi']:.2f}]}}"
        I_cell = f"{r['I']:.3f} {{\\scriptsize [{r['I_lo']:.2f},{r['I_hi']:.2f}]}}"
        E_cell = f"{r['E']:.3f} {{\\scriptsize [{r['E_lo']:.2f},{r['E_hi']:.2f}]}}"

        line = (f"{AGENT_LABELS[r['agent']]} & {r['n_att']} & {r['n_val']} "
                f"& {h_all} & {h_val} & {agee_cell} & {S_cell} & {I_cell} & {E_cell} \\\\")
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
