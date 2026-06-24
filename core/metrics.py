"""
AGEE v3 component metrics.

KEY CHANGES FROM v2:
  W1 FIX: Redundancy redefined as edge-novelty ratio (not USR).
          Breaks the circularity where R = USR was used as a component
          then AGEE was compared against USR as a baseline.
  W4 FIX: Coverage uses Shannon entropy with proper HS09 shrinkage
          (not Renyi-2, which lacks HS09 optimality guarantees).
  W8 FIX: Edge-novelty does not automatically give BFS R=1, because
          BFS traverses tree edges only — it misses cross-edges entirely.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional

from core.graph_utils import (
    shannon_entropy, shannon_max, community_distribution
)


# ===================================================================
# COMPONENT 1 — STRUCTURAL COVERAGE (S')
# W4 FIX: Shannon + HS09 (properly applied)
# ===================================================================

def coverage_v3(trajectory: List[int], G: nx.Graph,
                partition: Dict[int, int], n_communities: int) -> Tuple[float, float]:
    """
    Structural coverage using Shannon entropy with James-Stein shrinkage.

    Returns (coverage_score, shrinkage_lambda) for reproducibility.
    """
    visited = set(trajectory)
    n_total = len(G.nodes())
    if n_total == 0:
        return 0.0, 0.0

    node_cov = len(visited) / n_total

    if n_communities <= 1:
        return node_cov, 0.0

    alpha_K = 1.0 - 1.0 / n_communities

    # Shannon entropy with HS09 shrinkage (statistically valid combination)
    p_shrink, lam = community_distribution(visited, partition, n_communities)
    H = shannon_entropy(p_shrink)
    H_max = shannon_max(n_communities)
    comm_score = H / H_max if H_max > 0 else 0.0
    comm_score = np.clip(comm_score, 0.0, 1.0)

    score = alpha_K * comm_score + (1.0 - alpha_K) * node_cov
    return score, lam


# ===================================================================
# COMPONENT 2 — INFORMATION GAIN RATE (I')
# ===================================================================

def info_rate_v3(trajectory: List[int], G: nx.Graph,
                 beta: float = 1.0) -> float:
    """
    Local discovery fraction with diminishing-returns weighting.

    I' = (1/T) * sum w_t * g_t
    g_t = |N(v_t) \\ D_{t-1}| / |N(v_t)|
    w_t = (1 - |D_t| / |V|)^beta
    """
    if len(trajectory) == 0:
        return 0.0

    T = len(trajectory)
    n_total = len(G.nodes())
    if n_total == 0:
        return 0.0

    discovered = set()
    weighted_sum = 0.0

    for v in trajectory:
        neighbors = set(G.neighbors(v))
        degree = max(1, len(neighbors))
        new_neighbors = neighbors - discovered
        g_t = len(new_neighbors) / degree

        coverage_frac = len(discovered) / n_total
        w_t = (1.0 - min(coverage_frac, 1.0)) ** beta

        weighted_sum += w_t * g_t
        discovered.add(v)
        discovered.update(neighbors)

    return weighted_sum / T


# ===================================================================
# COMPONENT 3 — EXPLORATION EFFICIENCY (E')
# ===================================================================

def efficiency_v3(trajectory: List[int], G: nx.Graph,
                  partition: Optional[Dict[int, int]] = None,
                  n_communities: int = 1) -> float:
    """
    Coverage-speed AUC: how quickly cumulative coverage is achieved.

    E' = 0.5 * (AUC_node / AUC_ideal_node) + 0.5 * (AUC_comm / AUC_ideal_comm)

    AUC_ideal is defined in closed form:
      AUC_ideal_node = (1/T) * [sum_{t=0}^{n_u-1} (t+1)/n + (T - n_u) * n_u/n]
    where n_u = |unique nodes visited|, n = |V|.
    """
    T = len(trajectory)
    if T < 2:
        return 0.0

    n_total = len(G.nodes())
    if n_total == 0:
        return 0.0

    # --- Node coverage AUC ---
    visited = set()
    cumulative = []
    for v in trajectory:
        visited.add(v)
        cumulative.append(len(visited) / n_total)

    actual_auc = np.trapezoid(cumulative, dx=1.0 / T)

    n_unique = len(set(trajectory))
    if n_unique <= 1:
        coverage_auc = 0.0
    else:
        # Ideal: linear ramp to n_unique/n_total, then plateau
        ideal = []
        for t in range(T):
            if t < n_unique:
                ideal.append((t + 1) / n_total)
            else:
                ideal.append(n_unique / n_total)
        ideal_auc = np.trapezoid(ideal, dx=1.0 / T)
        coverage_auc = actual_auc / max(ideal_auc, 1e-12)
        coverage_auc = min(1.0, coverage_auc)

    # --- Community coverage AUC ---
    if partition is None or n_communities <= 1:
        community_auc = coverage_auc
    else:
        discovered_comms = set()
        cum_comm = []
        for v in trajectory:
            c = partition.get(v, 0)
            discovered_comms.add(c)
            cum_comm.append(len(discovered_comms) / n_communities)

        actual_comm = np.trapezoid(cum_comm, dx=1.0 / T)
        n_comm_found = len(set(partition.get(v, 0) for v in trajectory))
        ideal_comm = []
        for t in range(T):
            if t < n_comm_found:
                ideal_comm.append((t + 1) / n_communities)
            else:
                ideal_comm.append(n_comm_found / n_communities)
        ideal_comm_auc = np.trapezoid(ideal_comm, dx=1.0 / T)
        community_auc = actual_comm / max(ideal_comm_auc, 1e-12)
        community_auc = min(1.0, community_auc)

    return 0.5 * coverage_auc + 0.5 * community_auc


# ===================================================================
# COMPONENT 4 — REDUNDANCY (R)
# W1 FIX: Edge Novelty Ratio — NOT USR
# W8 FIX: BFS no longer gets R=1 automatically
# ===================================================================

def edge_novelty_ratio(trajectory: List[int], G: nx.Graph) -> float:
    """
    W1 FIX: Edge novelty ratio — fraction of steps that traverse
    a previously-untraversed edge.

    R_edge = |{t : (v_t, v_{t+1}) is a new edge}| / (T-1)

    This is fundamentally different from USR (unique nodes / steps):
      - USR measures node-level novelty
      - Edge novelty measures structural exploration of connections
      - BFS does NOT get R=1: it traverses only tree edges, missing all
        cross-edges and back-edges, so R_edge < 1 on any non-tree graph
      - An agent that visits the same nodes but via different edges gets
        credit for exploring new connections

    Domain: [0, 1]
    """
    if len(trajectory) < 2:
        return 0.0

    traversed_edges = set()
    novel_steps = 0

    for i in range(len(trajectory) - 1):
        u, v = trajectory[i], trajectory[i + 1]
        # Undirected: normalize edge representation
        edge = (min(u, v), max(u, v))

        if G.has_edge(u, v) and edge not in traversed_edges:
            novel_steps += 1
            traversed_edges.add(edge)

    return novel_steps / (len(trajectory) - 1)


def compute_usr(trajectory: List[int], n_total_nodes: int = 0) -> float:
    """
    USR (Unique Step Ratio) — kept as a BASELINE metric only.
    NOT used as an AGEE component (that would be circular per W1).
    """
    if len(trajectory) == 0:
        return 0.0
    T = len(trajectory)
    unique = len(set(trajectory))
    max_unique = min(T, n_total_nodes) if n_total_nodes > 0 else T
    return unique / max(max_unique, 1)


# ===================================================================
# AGGREGATION
# ===================================================================

def aggregate_power_mean(S: float, I: float, E: float,
                         w: Tuple[float, float, float] = (0.40, 0.35, 0.25),
                         p: float = 0.5,
                         epsilon: float = 0.01) -> float:
    """
    Power mean aggregation — 3 components, no redundancy penalty.

    W1 FIX: R (= USR) dropped entirely to eliminate circularity.
    W8 FIX: No BFS confound since R is gone.

    AGEE = M_p(S', I', E') = (w1*S^p + w2*I^p + w3*E^p)^(1/p)

    The epsilon-floor is a numerical safeguard. Boundedness and
    zero-collapse follow from it as design properties, not theorems.
    """
    S_f = max(S, epsilon)
    I_f = max(I, epsilon)
    E_f = max(E, epsilon)

    return (w[0] * S_f ** p + w[1] * I_f ** p + w[2] * E_f ** p) ** (1.0 / p)
