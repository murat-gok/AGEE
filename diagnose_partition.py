"""
Sprint 1 Week 2 — partition diagnostic.

Tests whether python-louvain and leidenalg are producing genuinely
different partitions on your machine. Run from the project root:

    python diagnose_partition.py

Expected output (Linux/Mac, my container):
    python-louvain: 0.16
    leidenalg:      0.11.0
    Karate Louvain: K=4, signature=(14, 11, 5, 4)
    Karate Leiden : K=4, signature=(12, 11, 6, 5)
    Identical partition? False

If your output shows 'Identical partition? True', it means the two
algorithms are returning the same partition on your Windows install,
which is what caused the suspicious r=1.0000 result in the
partition-robustness analysis.
"""
import sys
import warnings
import collections

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import community as cl
import leidenalg
import igraph as ig
import networkx as nx


def signature(partition):
    """Sorted community size tuple, ignoring community-label permutations."""
    sizes = sorted(collections.Counter(partition.values()).values(), reverse=True)
    return tuple(sizes)


def main():
    print("=" * 60)
    print("  Sprint 1 Week 2 — partition diagnostic")
    print("=" * 60)

    print("\nPackage versions:")
    print("  python-louvain:", getattr(cl, "__version__", "unknown"))
    print("  leidenalg:     ", leidenalg.version)
    print("  python-igraph: ", ig.__version__)

    print("\n--- Test 1: Karate Club graph ---")
    G = nx.karate_club_graph()
    p_louv = cl.best_partition(G, random_state=42, resolution=1.0)
    G_ig = ig.Graph.from_networkx(G)
    part_leid = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=42
    )
    p_leid = {v: part_leid.membership[i] for i, v in enumerate(G.nodes())}

    print("  Louvain: K =", len(set(p_louv.values())),
          " signature =", signature(p_louv))
    print("  Leiden : K =", len(set(p_leid.values())),
          " signature =", signature(p_leid))
    print("  Identical partition?", p_louv == p_leid)

    print("\n--- Test 2: Barabasi-Albert (n=200) ---")
    G2 = nx.barabasi_albert_graph(200, 3, seed=42)
    p_louv2 = cl.best_partition(G2, random_state=42, resolution=1.0)
    G2_ig = ig.Graph.from_networkx(G2)
    part_leid2 = leidenalg.find_partition(
        G2_ig, leidenalg.ModularityVertexPartition, seed=42
    )
    p_leid2 = {v: part_leid2.membership[i] for i, v in enumerate(G2.nodes())}

    print("  Louvain: K =", len(set(p_louv2.values())),
          " signature =", signature(p_louv2)[:5], "...")
    print("  Leiden : K =", len(set(p_leid2.values())),
          " signature =", signature(p_leid2)[:5], "...")
    print("  Identical partition?", p_louv2 == p_leid2)

    print("\n--- Test 3: Louvain re-run reproducibility (seed=42) ---")
    p_louv_a = cl.best_partition(G, random_state=42, resolution=1.0)
    p_louv_b = cl.best_partition(G, random_state=42, resolution=1.0)
    print("  Two Louvain runs with seed=42 give same partition?",
          p_louv_a == p_louv_b)

    print("\n--- Test 4: Leiden re-run reproducibility (seed=42) ---")
    part_a = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=42
    )
    part_b = leidenalg.find_partition(
        G_ig, leidenalg.ModularityVertexPartition, seed=42
    )
    p_a = {v: part_a.membership[i] for i, v in enumerate(G.nodes())}
    p_b = {v: part_b.membership[i] for i, v in enumerate(G.nodes())}
    print("  Two Leiden runs with seed=42 give same partition?",
          p_a == p_b)

    print("\n" + "=" * 60)
    print("  Diagnostic complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
