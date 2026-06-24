"""
Sprint 2 Task 3 supplement — Direct n=10^6 scalability measurement.

This script targets a 32 GB-class machine to replace the n=10^6
extrapolated values in Table tab:scalability with measured values.

Configuration:
  - 4 topologies (BA, ER, WS, SBM)
  - Single n=10^6 trial per topology (BA/ER/WS expected ~5-7min each,
    SBM expected ~110min due to denser parametrisation)
  - Trajectory length capped at 2000 (same as Sprint 2 task 3)
  - Memory monitoring throughout

POWER OUTAGE RESILIENCE:
  - Topology-level checkpointing: each completed topology is appended
    to CSV immediately. If power is lost mid-run, restart picks up
    from the last completed topology.
  - Cheap topologies (BA, ER, WS, ~5-7 min each) run FIRST, so even
    a mid-SBM interruption preserves three results.
  - For maximum safety, split the SBM run into a separate invocation:
        python analysis/run_scalability_n1e6.py --topologies BA,ER,WS
        python analysis/run_scalability_n1e6.py --topologies SBM
  - WITHIN a topology (e.g. SBM's 110-min Leiden), there is no
    fine-grained checkpoint — power loss during SBM Leiden will lose
    that single topology's progress. Mitigation: UPS, or accept the
    risk for SBM specifically.

Outputs:
  analysis/results/scalability_n1e6.csv
    -> Will be merged with scalability_benchmark.csv by merge_scalability.py

Run:
    python analysis/run_scalability_n1e6.py
    python analysis/run_scalability_n1e6.py --topologies BA,ER,WS
    python analysis/run_scalability_n1e6.py --topologies SBM
"""
import os
import sys
import time
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from core.agee import AGEECalculator, DEFAULT_CONFIG

N = 1_000_000
# Order: cheap topologies first so an SBM-mid power outage still
# preserves the BA/ER/WS results.
DEFAULT_TOPOLOGIES = ["BA", "ER", "WS", "SBM"]
SEED_BASE = 42
OUT_PATH = os.path.join(PROJECT_ROOT, "analysis", "results",
                         "scalability_n1e6.csv")


def make_graph(topology, n, seed):
    """Same generators as run_scalability_benchmark.py."""
    if topology == "BA":
        return nx.barabasi_albert_graph(n, 3, seed=seed)
    if topology == "ER":
        p = 6.0 / n
        return nx.fast_gnp_random_graph(n, p, seed=seed)
    if topology == "WS":
        return nx.watts_strogatz_graph(n, k=6, p=0.1, seed=seed)
    if topology == "SBM":
        n_blocks = 10
        sizes = [n // n_blocks] * n_blocks
        sizes[-1] += n - sum(sizes)
        p_intra = min(0.02, 50.0 / max(n // n_blocks, 1))
        p_inter = min(0.001, 1.0 / n)
        p_matrix = [[p_intra if i == j else p_inter
                     for j in range(n_blocks)]
                    for i in range(n_blocks)]
        return nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    raise ValueError(topology)


def synthesize_trajectory(G, length, seed):
    """Random-walk-restart trajectory."""
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
            current = nodes[rng.integers(0, len(nodes))]
        else:
            current = neighbors[rng.integers(0, len(neighbors))]
        trajectory.append(current)
        visited.add(current)
    return trajectory


def write_row(row):
    """Append a single row to CSV (creates header if file missing)."""
    df = pd.DataFrame([row])
    write_header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Direct n=10^6 scalability measurement"
    )
    parser.add_argument(
        "--topologies", type=str, default=",".join(DEFAULT_TOPOLOGIES),
        help="Comma-separated topologies to run (default: BA,ER,WS,SBM)"
    )
    args = parser.parse_args()

    requested = [t.strip() for t in args.topologies.split(",") if t.strip()]
    invalid = [t for t in requested if t not in DEFAULT_TOPOLOGIES]
    if invalid:
        print(f"  [FAIL] Invalid topologies: {invalid}")
        print(f"  Valid options: {DEFAULT_TOPOLOGIES}")
        sys.exit(1)
    topologies = requested

    print("=" * 72)
    print("  Sprint 2 Task 3 supplement — Direct n=10^6 measurement")
    print("=" * 72)
    print(f"\n  Target: {len(topologies)} topologies × 1 trial = "
          f"{len(topologies)} runs at n={N:,}")
    print(f"  Topologies: {topologies}")
    print(f"  Available memory: "
          f"{psutil.virtual_memory().available/1e9:.1f} GB")
    print(f"  Output: {OUT_PATH}")

    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        done = set(existing["topology"])
        print(f"\n  Resume mode: {len(done)} topologies already complete: "
              f"{sorted(done)}")
    else:
        done = set()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    for topology in topologies:
        if topology in done:
            print(f"\n  [SKIP] {topology}: already in CSV")
            continue

        print(f"\n  >>> {topology} at n={N:,} <<<")
        avail_before = psutil.virtual_memory().available / 1e9
        print(f"      Available memory before: {avail_before:.2f} GB")
        if avail_before < 8.0:
            print(f"      [WARNING] Low memory; consider closing other apps")

        proc = psutil.Process()
        mem_before = proc.memory_info().rss / 1e9
        peak_mem_during = mem_before
        t_total_start = time.time()

        # Step 1: generate graph
        print(f"      [1/3] Generating graph...")
        t0 = time.time()
        try:
            G = make_graph(topology, N, SEED_BASE)
        except MemoryError:
            print(f"      [FAIL] MemoryError generating {topology}")
            continue
        t_graph = time.time() - t0
        n_e = G.number_of_edges()
        mem_after_graph = proc.memory_info().rss / 1e9
        peak_mem_during = max(peak_mem_during, mem_after_graph)
        print(f"            done in {t_graph:.1f}s, |E|={n_e:,}, "
              f"mem={mem_after_graph:.2f} GB")

        # Step 2: synthesize trajectory
        print(f"      [2/3] Synthesising trajectory...")
        traj_len = min(2000, max(100, N // 10))
        t0 = time.time()
        trajectory = synthesize_trajectory(G, traj_len, SEED_BASE + 1)
        t_traj = time.time() - t0
        print(f"            done in {t_traj:.1f}s, |traj|={len(trajectory)}")

        # Step 3: compute AGEE
        print(f"      [3/3] Computing AGEE (Leiden + S'I'E')...")
        t0 = time.time()
        try:
            calc = AGEECalculator(G, config=DEFAULT_CONFIG,
                                   graph_name=f"scale_{topology}_n1e6")
            t_init = time.time() - t0
            mem_after_init = proc.memory_info().rss / 1e9
            peak_mem_during = max(peak_mem_during, mem_after_init)
            print(f"            Leiden init: {t_init:.1f}s, "
                  f"mem={mem_after_init:.2f} GB")

            t0 = time.time()
            r = calc.compute(trajectory, "trial0")
            t_agee = time.time() - t0
            peak_mem_during = max(peak_mem_during,
                                   proc.memory_info().rss / 1e9)
            print(f"            S'I'E': {t_agee:.2f}s, agee={r.agee:.3f}")
        except MemoryError:
            print(f"      [FAIL] MemoryError computing AGEE on {topology}")
            del G, trajectory
            gc.collect()
            continue
        except Exception as e:
            print(f"      [FAIL] {type(e).__name__}: {e}")
            del G
            gc.collect()
            continue

        t_total = time.time() - t_total_start
        mem_delta = peak_mem_during - mem_before

        row = {
            "topology": topology,
            "n": N,
            "trial": 0,
            "n_nodes": G.number_of_nodes(),
            "n_edges": n_e,
            "traj_length": len(trajectory),
            "t_graph_gen_s": t_graph,
            "t_trajectory_s": t_traj,
            "t_agee_init_s": t_init,
            "t_agee_compute_s": t_agee,
            "t_total_s": t_total,
            "mem_delta_gb": mem_delta,
            "agee": r.agee,
            "S": r.coverage,
            "I": r.info_rate,
            "E": r.efficiency,
        }
        write_row(row)
        print(f"      [DONE] {topology}: total={t_total:.1f}s "
              f"({t_total/60:.1f}min), peak mem delta={mem_delta:+.2f} GB")
        print(f"             Row written to CSV")

        # Cleanup
        del G, trajectory, calc, r
        gc.collect()

    print("\n" + "=" * 72)
    print("  All topologies complete. Inspect output:")
    print(f"    {OUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
