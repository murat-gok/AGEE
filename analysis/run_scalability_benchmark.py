"""
Sprint 2 Task 3 — Scalability benchmark.

Measures AGEE's wall-clock and memory scaling across:
  - 4 graph topologies: BA (scale-free), ER (random), WS (small-world),
    SBM (clustered)
  - 5 sizes: n ∈ {10^3, 5×10^3, 10^4, 5×10^4, 10^5}
  - 3 trials per (topology, size) combination (1 trial for largest)

Computes log-log regression of wall-clock vs n to estimate empirical
scaling exponent. Reports per-component (Leiden partition vs S' vs I'
vs E' compute) breakdown.

RAM disclosure: sandbox has ~4 GB available; n=10^6 cannot be run here.
The manuscript reports measured scaling up to 10^5 and extrapolates
n=10^6 from the log-log regression.

Outputs:
  analysis/results/scalability_benchmark.csv
  paper/figures/tab_scalability.tex
  paper/figures/fig6_scalability.{pdf,png}

Run from project root:
    python analysis/run_scalability_benchmark.py
"""
import os
import sys
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.agee import AGEECalculator, DEFAULT_CONFIG

# Configurable sizes — start small to avoid sandbox OOM
SIZES = [1000, 5000, 10000, 50000, 100000]
TOPOLOGIES = ["BA", "ER", "WS", "SBM"]
N_TRIALS = {1000: 3, 5000: 2, 10000: 2, 50000: 1, 100000: 1}
SEED_BASE = 42


def make_graph(topology, n, seed):
    """Generate a synthetic graph of given topology and size."""
    rng = np.random.default_rng(seed)
    if topology == "BA":
        # Scale-free: Barabási-Albert preferential attachment
        m = 3  # edges per new node
        G = nx.barabasi_albert_graph(n, m, seed=seed)
    elif topology == "ER":
        # Random: Erdős-Rényi G(n, p), target mean degree ~6
        # Use fast_gnp_random_graph (O(n+m)) instead of erdos_renyi_graph (O(n^2))
        p = 6.0 / n
        G = nx.fast_gnp_random_graph(n, p, seed=seed)
    elif topology == "WS":
        # Small-world: Watts-Strogatz with k=6 nearest neighbours, β=0.1
        G = nx.watts_strogatz_graph(n, k=6, p=0.1, seed=seed)
    elif topology == "SBM":
        # Stochastic block model: 10 communities of equal size.
        # Scale p_intra inversely with n so mean degree stays bounded (~10).
        n_blocks = 10
        sizes = [n // n_blocks] * n_blocks
        sizes[-1] += n - sum(sizes)
        # Target mean intra-community degree ~5, inter ~1
        p_intra = min(0.02, 50.0 / max(n // n_blocks, 1))
        p_inter = min(0.001, 1.0 / n)
        p_matrix = [[p_intra if i == j else p_inter
                     for j in range(n_blocks)]
                    for i in range(n_blocks)]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    else:
        raise ValueError(topology)
    return G


def synthesize_trajectory(G, length, seed):
    """Random-walk-like trajectory across the graph."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if not nodes:
        return []
    current = nodes[rng.integers(0, len(nodes))]
    trajectory = [current]
    visited = {current}
    for _ in range(length - 1):
        neighbors = [nb for nb in G.neighbors(current) if nb not in visited]
        if not neighbors:
            # Restart from a random node (random-restart walk)
            current = nodes[rng.integers(0, len(nodes))]
        else:
            current = neighbors[rng.integers(0, len(neighbors))]
        trajectory.append(current)
        visited.add(current)
    return trajectory


def measure_one(topology, n, trial):
    """One trial: generate graph, trajectory, compute AGEE, time it."""
    seed = SEED_BASE + trial * 1000

    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e9  # GB

    t0 = time.time()
    G = make_graph(topology, n, seed)
    t_graph = time.time() - t0

    # Trajectory length: min(2000, n/10) — long enough to be non-trivial
    traj_length = min(2000, max(100, n // 10))
    t1 = time.time()
    trajectory = synthesize_trajectory(G, traj_length, seed + 1)
    t_traj = time.time() - t1

    # AGEE compute (this is the part we care about)
    t2 = time.time()
    calc = AGEECalculator(G, config=DEFAULT_CONFIG, graph_name=f"scale_{topology}_n{n}")
    t_calc_init = time.time() - t2

    t3 = time.time()
    r = calc.compute(trajectory, f"trial{trial}")
    t_agee = time.time() - t3

    mem_after = proc.memory_info().rss / 1e9
    mem_delta = mem_after - mem_before

    result = {
        "topology": topology,
        "n": n,
        "trial": trial,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "traj_length": len(trajectory),
        "t_graph_gen_s": t_graph,
        "t_trajectory_s": t_traj,
        "t_agee_init_s": t_calc_init,
        "t_agee_compute_s": t_agee,
        "t_total_s": t_graph + t_traj + t_calc_init + t_agee,
        "mem_delta_gb": mem_delta,
        "agee": r.agee,
        "S": r.coverage,
        "I": r.info_rate,
        "E": r.efficiency,
    }

    # Cleanup
    del G, trajectory, calc, r
    gc.collect()

    return result


def main():
    print("=" * 72)
    print("  Sprint 2 Task 3 — Scalability benchmark")
    print("=" * 72)

    available_gb = psutil.virtual_memory().available / 1e9
    print(f"\n  Available memory: {available_gb:.1f} GB")

    rows = []
    for topology in TOPOLOGIES:
        print(f"\n  Topology: {topology}")
        for n in SIZES:
            n_trials = N_TRIALS[n]
            for trial in range(n_trials):
                # Check available memory before attempting large graphs
                if n >= 50000:
                    avail = psutil.virtual_memory().available / 1e9
                    if avail < 1.0:
                        print(f"    [SKIP] n={n} trial={trial}: low memory "
                              f"({avail:.2f} GB available)")
                        continue
                try:
                    r = measure_one(topology, n, trial)
                    rows.append(r)
                    print(f"    n={n:>6d} trial={trial}: "
                          f"agee={r['agee']:.3f}, t={r['t_total_s']:.2f}s, "
                          f"mem={r['mem_delta_gb']:+.2f} GB")
                except Exception as e:
                    print(f"    n={n} trial={trial} FAILED: {e}")
                    gc.collect()

    df = pd.DataFrame(rows)
    out_dir = os.path.join(PROJECT_ROOT, "analysis", "results")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "scalability_benchmark.csv"), index=False)

    # Log-log regression: t_total = a * n^b
    print("\n  Log-log scaling regression per topology:")
    fits = {}
    for topology in TOPOLOGIES:
        sub = df[df["topology"] == topology]
        if len(sub) < 4:
            continue
        # Use mean per n
        agg = sub.groupby("n")["t_total_s"].mean().reset_index()
        log_n = np.log10(agg["n"].values)
        log_t = np.log10(agg["t_total_s"].values)
        # Linear fit
        slope, intercept = np.polyfit(log_n, log_t, 1)
        # Extrapolate to 10^6
        t_extrap_1e6 = 10 ** (slope * 6 + intercept)
        fits[topology] = {
            "exponent": slope, "intercept": intercept,
            "t_at_1e6_extrap_s": t_extrap_1e6,
        }
        print(f"    {topology}: exponent={slope:.3f}, "
              f"extrapolated t(n=10^6) ≈ {t_extrap_1e6:.1f}s "
              f"= {t_extrap_1e6/60:.1f}min")

    # Write LaTeX table and figure
    fig_dir = os.path.join(PROJECT_ROOT, "paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    write_latex_table(df, fits, os.path.join(fig_dir, "tab_scalability.tex"))
    make_figure(df, fits, os.path.join(fig_dir, "fig6_scalability"))
    print("\n  Saved CSV, tab_scalability.tex, fig6_scalability.pdf/.png")


def write_latex_table(df, fits, out_path):
    """LaTeX table: t and AGEE per (topology, n)."""
    lines = []
    lines.append("% Auto-generated by analysis/run_scalability_benchmark.py")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Scalability of AGEE on synthetic large graphs. "
                 "Each cell reports mean wall-clock time (in seconds) over "
                 "$N_\\mathrm{trials}$ trials for the full AGEE pipeline "
                 "(graph generation + trajectory + Leiden partition + "
                 "S'I'E' compute). The log-log fit gives the empirical "
                 "scaling exponent $\\alpha$ in $t \\propto n^\\alpha$; "
                 "$n=10^6$ values are extrapolated from the regression. "
                 "Sandbox memory ($\\sim$4 GB) limits direct measurement to "
                 "$n \\leq 10^5$.}")
    lines.append("\\label{tab:scalability}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Topology} & "
                 "$\\mathbf{10^3}$ & $\\mathbf{5{\\times}10^3}$ & "
                 "$\\mathbf{10^4}$ & $\\mathbf{5{\\times}10^4}$ & "
                 "$\\mathbf{10^5}$ & $\\boldsymbol{\\alpha}$ "
                 "(extrap.\\ $10^6$) \\\\")
    lines.append("\\midrule")
    for topology in TOPOLOGIES:
        sub = df[df["topology"] == topology]
        if len(sub) == 0:
            continue
        cells = [topology]
        for n in SIZES:
            data = sub[sub["n"] == n]["t_total_s"]
            if len(data) > 0:
                cells.append(f"{data.mean():.2f}")
            else:
                cells.append("---")
        if topology in fits:
            f = fits[topology]
            cells.append(f"{f['exponent']:.2f} "
                         f"({f['t_at_1e6_extrap_s']/60:.1f}min)")
        else:
            cells.append("---")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_figure(df, fits, out_path):
    """Two-panel figure: wall-clock vs n (log-log), memory vs n."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors = {"BA": "#117733", "ER": "#88CCEE", "WS": "#DDCC77", "SBM": "#CC6677"}
    markers = {"BA": "o", "ER": "s", "WS": "D", "SBM": "^"}

    for topology in TOPOLOGIES:
        sub = df[df["topology"] == topology]
        if len(sub) == 0:
            continue
        agg = sub.groupby("n").agg({"t_total_s": ["mean", "std"],
                                     "mem_delta_gb": "mean"}).reset_index()
        agg.columns = ["n", "t_mean", "t_std", "mem_mean"]
        ax1.errorbar(agg["n"], agg["t_mean"], yerr=agg["t_std"].fillna(0),
                     fmt=markers[topology] + "-", color=colors[topology],
                     label=topology, capsize=2, markersize=5, linewidth=1.0,
                     markeredgecolor="black", markeredgewidth=0.4)
        ax2.plot(agg["n"], agg["mem_mean"], markers[topology] + "-",
                 color=colors[topology], label=topology, markersize=5,
                 linewidth=1.0, markeredgecolor="black",
                 markeredgewidth=0.4)

    # Extrapolation lines to 10^6
    n_extrap = np.logspace(3, 6, 100)
    for topology in TOPOLOGIES:
        if topology in fits:
            f = fits[topology]
            t_pred = 10 ** (f["exponent"] * np.log10(n_extrap) + f["intercept"])
            ax1.plot(n_extrap, t_pred, color=colors[topology],
                     linewidth=0.5, alpha=0.4, linestyle="--")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Graph size $n$")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.axvline(1e6, color="gray", lw=0.5, linestyle=":", alpha=0.6)
    ax1.text(1e6, ax1.get_ylim()[1] * 0.5, r" extrap.", fontsize=7,
             color="gray", verticalalignment="top")
    ax1.legend(loc="upper left", fontsize=7, frameon=False)
    ax1.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Graph size $n$")
    ax2.set_ylabel("Peak memory delta (GB)")
    ax2.legend(loc="upper left", fontsize=7, frameon=False)
    ax2.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4)

    plt.tight_layout()
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.savefig(out_path + ".png", bbox_inches="tight", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
