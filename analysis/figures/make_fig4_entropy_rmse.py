"""
Figure 4 (manuscript): Empirical RMSE of Shannon entropy estimators as a
function of trajectory length T, on a synthetic Stochastic Block Model
graph with known ground-truth community structure.

Why: justifies the James-Stein choice over plug-in MLE, Miller-Madow,
and an alternative shrinkage scheme. The expected pattern:
  - MLE has noticeable downward bias at small T
  - Miller-Madow corrects most of MLE's first-order bias
  - James-Stein (Hausser-Strimmer 2009) is robust across T
  - All converge to the ground-truth H at large T

Setup:
  SBM with K=5 blocks, n=400 nodes, p_in=0.15, p_out=0.01.
  Ground-truth community distribution is uniform (80/80/80/80/80),
  so true entropy H = log K.
  For each T in [10, 20, ..., 400], simulate random samples from
  the uniform community distribution and estimate H by each method.
  Repeat 1000 times per T and compute RMSE vs ground-truth.

Inputs:  None (synthetic)
<<<<<<< HEAD
Outputs: tkde/figures/fig4_entropy_rmse.{pdf,png}
=======
Outputs: paper/figures/fig4_entropy_rmse.{pdf,png}
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378

Run from project root:
    python analysis/figures/make_fig4_entropy_rmse.py
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from core.graph_utils import james_stein_shrinkage

<<<<<<< HEAD
OUT_DIR = os.path.join(PROJECT_ROOT, "tkde", "figures")
=======
OUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
os.makedirs(OUT_DIR, exist_ok=True)


def shannon(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def estimate_mle(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return shannon(p)


def estimate_miller_madow(counts):
    """MLE + (K-1)/(2N) bias correction."""
    n = counts.sum()
    if n == 0:
        return 0.0
    K_nonzero = (counts > 0).sum()
    return estimate_mle(counts) + (K_nonzero - 1) / (2 * n)


def estimate_james_stein(counts, K):
    p_sh, _ = james_stein_shrinkage(counts, K)
    return shannon(p_sh)


def main():
    # ---- Ground truth: uniform community over K=5 ----
    K = 5
    p_true = np.full(K, 1.0 / K)
    H_true = shannon(p_true)
    print(f"Ground truth: K={K}, H={H_true:.4f} (= log K)")

    # ---- Experimental setup ----
    T_grid = np.array([5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 200, 300, 400])
    n_repeats = 1000
    rng = np.random.default_rng(42)

    estimators = {
        "MLE (plug-in)":      estimate_mle,
        "Miller-Madow":       estimate_miller_madow,
        "James-Stein":        lambda c: estimate_james_stein(c, K),
    }

    # ---- Run ----
    rmse = {name: np.zeros(len(T_grid)) for name in estimators}
    bias = {name: np.zeros(len(T_grid)) for name in estimators}

    for i, T in enumerate(T_grid):
        errors = {name: np.zeros(n_repeats) for name in estimators}
        for r in range(n_repeats):
            samples = rng.multinomial(T, p_true)
            for name, fn in estimators.items():
                H_hat = fn(samples)
                errors[name][r] = H_hat - H_true
        for name in estimators:
            rmse[name][i] = np.sqrt(np.mean(errors[name] ** 2))
            bias[name][i] = np.mean(errors[name])

    print("\nT     |       MLE  Miller-Madow  James-Stein")
    for i, T in enumerate(T_grid):
        print(f"{T:>4}  |  {rmse['MLE (plug-in)'][i]:.4f}      "
              f"{rmse['Miller-Madow'][i]:.4f}      {rmse['James-Stein'][i]:.4f}")

    # ---- Figure ----
    colors = {
        "MLE (plug-in)":  "#888888",
        "Miller-Madow":   "#1F77B4",
        "James-Stein":    "#D62728",
    }
    markers = {
        "MLE (plug-in)":  "o",
        "Miller-Madow":   "s",
        "James-Stein":    "D",
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=150)
    for name in estimators:
        ax.plot(T_grid, rmse[name], color=colors[name],
                marker=markers[name], markersize=5, linewidth=1.5,
                label=name, alpha=0.92)
    ax.set_xscale("log")
    ax.set_xlabel(r"Trajectory length $T$")
    ax.set_ylabel(r"RMSE of $\hat H$ relative to $H = \log K$")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92,
              edgecolor="gray", title="Estimator", title_fontsize=9)
    ax.set_ylim(bottom=0)

    ax.text(0.5, -0.20,
            f"Synthetic uniform community distribution, K = {K}; "
            f"{n_repeats} Monte Carlo repetitions per $T$.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color="dimgray")

    plt.tight_layout()
    pdf_path = os.path.join(OUT_DIR, "fig4_entropy_rmse.pdf")
    png_path = os.path.join(OUT_DIR, "fig4_entropy_rmse.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
