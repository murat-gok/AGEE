"""
Graph utilities for AGEE v3.

W4 FIX: Uses Shannon entropy with James-Stein shrinkage (Hausser & Strimmer 2009),
which is the statistically valid combination. Rényi-2 with HS09 shrinkage was
a misapplication — HS09 optimality is proven for Shannon plug-in MSE only.
"""

import numpy as np
import networkx as nx
from collections import Counter
from typing import Dict, Set, Optional
import warnings


# ---------------------------------------------------------------------------
# James-Stein shrinkage for Shannon entropy (proper HS09)
# ---------------------------------------------------------------------------

def james_stein_shrinkage(counts: np.ndarray, n_categories: int) -> np.ndarray:
    """
    James-Stein shrinkage estimator for a categorical distribution.
    Shrinks MLE towards uniform. Guarantees all p_i > 0.
    
    Reference: Hausser & Strimmer, JMLR 2009 — optimal for Shannon plug-in.
    """
    n_samples = counts.sum()
    if n_samples == 0:
        return np.ones(n_categories) / n_categories

    p_mle = counts / n_samples
    target = 1.0 / n_categories

    sum_sq = np.sum(p_mle ** 2)
    lambda_den = (n_samples - 1) * np.sum((p_mle - target) ** 2)

    if lambda_den < 1e-12:
        lam = 1.0
    else:
        lam = np.clip((1.0 - sum_sq) / lambda_den, 0.0, 1.0)

    p_shrink = lam * target + (1.0 - lam) * p_mle
    return p_shrink, lam  # return lambda for reproducibility reporting


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum p_i log(p_i). Handles zeros."""
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log(p))


def shannon_max(K: int) -> float:
    """Maximum Shannon entropy for K categories = log(K)."""
    if K <= 1:
        return 0.0
    return np.log(K)


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities(G: nx.Graph, method: str = "louvain",
                       resolution: float = 1.0,
                       random_state: int = 42) -> Dict[int, int]:
    """Detect communities, return node -> community_id mapping.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    method : str
        'louvain' (default), 'leiden', or 'label_propagation'.
    resolution : float
        Resolution parameter for modularity-based methods.
    random_state : int
        Seed for stochastic partitioning. Default 42 (project-wide seed
        from seeds.yaml). Pinning this is essential for reproducibility:
        on small graphs (e.g. Karate Club), Louvain may produce
        K-invariant but lambda-different partitions across runs, which
        changes the reported shrinkage intensity.
    """
    if len(G.nodes()) == 0:
        return {}

    if method == "louvain":
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(
                G, resolution=resolution, random_state=random_state
            )
        except ImportError:
            warnings.warn("python-louvain not installed, using label_propagation")
            return detect_communities(G, method="label_propagation",
                                      random_state=random_state)
    elif method == "label_propagation":
        communities = nx.community.label_propagation_communities(G)
        partition = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
    elif method == "leiden":
        try:
            import leidenalg
            import igraph as ig
        except ImportError as e:
            raise ImportError(
                "Leiden community detection requires the 'leidenalg' and "
                "'python-igraph' packages. Install with:\n"
                "    pip install leidenalg python-igraph\n"
                "Alternatively, request method='louvain' explicitly to use "
                "Louvain. Silent fallback has been removed because it caused "
                "experiments labelled 'Leiden' to silently use Louvain."
            ) from e
        G_ig = ig.Graph.from_networkx(G)
        part = leidenalg.find_partition(
            G_ig, leidenalg.ModularityVertexPartition,
            seed=random_state
        )
        partition = {v: part.membership[i] for i, v in enumerate(G.nodes())}
    else:
        raise ValueError(f"Unknown method: {method}")

    unique_labels = sorted(set(partition.values()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    partition = {node: label_map[cid] for node, cid in partition.items()}
    return partition


def community_distribution(visited_nodes: Set[int], partition: Dict[int, int],
                           n_communities: int) -> tuple:
    """
    Compute community distribution with James-Stein shrinkage.
    Returns (p_shrink, shrinkage_intensity_lambda).
    """
    counts = np.zeros(n_communities)
    for v in visited_nodes:
        if v in partition:
            counts[partition[v]] += 1
    return james_stein_shrinkage(counts, n_communities)


# ---------------------------------------------------------------------------
# Shortest path utilities
# ---------------------------------------------------------------------------

def shortest_path_distance(G: nx.Graph, source: int, target: int) -> float:
    try:
        return nx.shortest_path_length(G, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float('inf')
