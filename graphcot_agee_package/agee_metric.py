"""
AGEE metric -- faithful re-implementation of the manuscript's Section 5.

    AGEE(tau, G) = M_p( S'(tau,G), I'(tau,G), E'(tau,G); w )

with the power-mean aggregation, James-Stein-shrunk structural coverage,
diminishing-returns information-gain rate, and AUC-vs-ideal exploration
efficiency exactly as defined in Equations (2)-(9) and Algorithm 1.

IMPORTANT (comparability): community detection must match your ToG/PoG run.
The paper's "default Leiden" == leidenalg modularity partition with a fixed
seed; that is the default here. Run ``validate_agee.py`` on one of your
existing ToG/PoG trajectory files to confirm this module reproduces the
published AGEE values before trusting the Graph-CoT numbers.
"""
import math
import warnings
import numpy as np
import config


# ---------------------------------------------------------------------------
# Leiden partition  ->  {node: community_id}
# ---------------------------------------------------------------------------
def leiden_partition(G):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}, 0
    if G.number_of_edges() == 0:
        # every node is its own community
        return {v: i for i, v in enumerate(nodes)}, len(nodes)

    # Preferred: leidenalg (Traag 2019 -- the paper's citation).
    try:
        import igraph as ig
        import leidenalg
        idx = {v: i for i, v in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        g = ig.Graph(n=len(nodes), edges=edges)
        if config.LEIDEN_OBJECTIVE == "modularity":
            part = leidenalg.find_partition(
                g, leidenalg.ModularityVertexPartition, seed=config.LEIDEN_SEED)
        else:
            part = leidenalg.find_partition(
                g, leidenalg.CPMVertexPartition, seed=config.LEIDEN_SEED)
        labels = part.membership
        return {nodes[i]: labels[i] for i in range(len(nodes))}, len(set(labels))
    except ImportError:
        pass

    # Fallback 1: python-igraph's built-in Leiden.
    try:
        import igraph as ig
        idx = {v: i for i, v in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        g = ig.Graph(n=len(nodes), edges=edges)
        part = g.community_leiden(objective_function="modularity")
        labels = part.membership
        return {nodes[i]: labels[i] for i in range(len(nodes))}, len(set(labels))
    except ImportError:
        pass

    # Fallback 2: NOT Leiden -- warn loudly. Numbers may not match ToG/PoG.
    warnings.warn(
        "leidenalg/igraph unavailable; falling back to networkx greedy "
        "modularity. This is NOT Leiden and AGEE values may not be comparable "
        "to your ToG/PoG runs. Install leidenalg+python-igraph.")
    from networkx.algorithms.community import greedy_modularity_communities
    comms = greedy_modularity_communities(G)
    lab = {}
    for cid, cset in enumerate(comms):
        for v in cset:
            lab[v] = cid
    return lab, len(comms)


# ---------------------------------------------------------------------------
# Component 1: structural coverage S'
# ---------------------------------------------------------------------------
def structural_coverage(visited_unique, comm_of, K, n):
    nu = len(visited_unique)
    if K <= 1:
        # alpha_K = 0 -> entropy term vanishes; S' = |V_tau|/n
        return nu / n, {"K": K, "lambda_star": float("nan"), "H_norm": float("nan")}

    # community-visit MLE distribution over K communities
    counts = np.zeros(K)
    for v in visited_unique:
        c = comm_of.get(v)
        if c is not None:
            counts[c] += 1
    total = counts.sum()
    if total == 0:
        return 0.0, {"K": K, "lambda_star": float("nan"), "H_norm": 0.0}
    p_mle = counts / total

    # James-Stein shrinkage toward uniform (HS09)
    num = 1.0 - np.sum(p_mle ** 2)
    den = (nu - 1) * np.sum((p_mle - 1.0 / K) ** 2) if nu > 1 else 0.0
    lam = 1.0 if den == 0 else float(np.clip(num / den, 0.0, 1.0))
    p_sh = lam * (1.0 / K) + (1.0 - lam) * p_mle

    nz = p_sh[p_sh > 0]
    H = float(-np.sum(nz * np.log(nz)))
    H_norm = H / math.log(K)

    alpha_K = 1.0 - 1.0 / K
    s_prime = alpha_K * H_norm + (1.0 - alpha_K) * (nu / n)
    return float(np.clip(s_prime, 0.0, 1.0)), {
        "K": K, "lambda_star": lam, "H_norm": H_norm}


# ---------------------------------------------------------------------------
# Component 2: information gain rate I'
# ---------------------------------------------------------------------------
def info_gain_rate(visited_seq, G, n, beta):
    T = len(visited_seq)
    if T == 0:
        return 0.0
    discovered = set()
    if config.START_NODE_IN_DISCOVERED and visited_seq:
        discovered.add(visited_seq[0])
    total = 0.0
    for v in visited_seq:
        Nv = set(G.neighbors(v)) if v in G else set()
        # g_t with D_{t-1}
        g_t = (len(Nv - discovered) / len(Nv)) if Nv else 0.0
        if config.W_USES_PREUPDATE_DISCOVERED:
            w_t = (1.0 - len(discovered) / n) ** beta
            total += w_t * g_t
            discovered |= {v} | Nv
        else:
            discovered |= {v} | Nv
            w_t = (1.0 - len(discovered) / n) ** beta
            total += w_t * g_t
    return float(total / T)


# ---------------------------------------------------------------------------
# Component 3: exploration efficiency E'
# ---------------------------------------------------------------------------
def _auc_actual(cover_curve):
    return float(np.mean(cover_curve)) if len(cover_curve) else 0.0


def _auc_ideal(units_total, denom, T):
    # discover one new unit/step for `units_total` steps, then plateau
    u = units_total
    s = (u * (u + 1) / 2.0) + max(0, (T - u)) * u
    return float(s / (T * denom)) if (T > 0 and denom > 0) else 0.0


def exploration_efficiency(visited_seq, comm_of, K, n):
    T = len(visited_seq)
    if T == 0:
        return 0.0
    # node coverage curve
    seen_nodes, node_curve = set(), []
    for v in visited_seq:
        seen_nodes.add(v)
        node_curve.append(len(seen_nodes) / n)
    nu = len(seen_nodes)
    auc_node = _auc_actual(node_curve)
    auc_node_ideal = _auc_ideal(nu, n, T)

    # community coverage curve
    seen_comm, comm_curve = set(), []
    for v in visited_seq:
        c = comm_of.get(v)
        if c is not None:
            seen_comm.add(c)
        comm_curve.append(len(seen_comm) / K if K > 0 else 0.0)
    Ku = len(seen_comm)
    auc_comm = _auc_actual(comm_curve)
    auc_comm_ideal = _auc_ideal(Ku, K, T)

    r_node = (auc_node / auc_node_ideal) if auc_node_ideal > 0 else 0.0
    r_comm = (auc_comm / auc_comm_ideal) if auc_comm_ideal > 0 else 0.0
    e_prime = 0.5 * min(r_node, 1.0) + 0.5 * min(r_comm, 1.0)
    return float(np.clip(e_prime, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregation: weighted power mean
# ---------------------------------------------------------------------------
def power_mean(xs, w, p, eps):
    acc = 0.0
    for x, wi in zip(xs, w):
        acc += wi * (max(x, eps) ** p)
    return float(acc ** (1.0 / p))


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def compute_agee(visited_seq, G):
    """visited_seq: ordered list of entity ids the agent focused on (the
    trajectory tau). G: undirected simple graph (graph_env.build_undirected_graph).
    Returns a dict of components + AGEE and diagnostics."""
    # keep only nodes present in G, preserve order
    visited_seq = [v for v in visited_seq if v in G]
    n = G.number_of_nodes()
    T = len(visited_seq)
    visited_unique = list(dict.fromkeys(visited_seq))
    out = {"n": n, "T": T, "n_visited": len(visited_unique)}
    if n == 0 or T == 0:
        out.update(S=float("nan"), I=float("nan"), E=float("nan"),
                   AGEE=float("nan"), K=0, lambda_star=float("nan"))
        return out

    comm_of, K = leiden_partition(G)
    S, sdiag = structural_coverage(visited_unique, comm_of, K, n)
    I = info_gain_rate(visited_seq, G, n, config.AGEE_BETA)
    E = exploration_efficiency(visited_seq, comm_of, K, n)
    AGEE = power_mean([S, I, E], config.AGEE_WEIGHTS, config.AGEE_P, config.AGEE_EPS)

    out.update(S=S, I=I, E=E, AGEE=AGEE, K=K,
               lambda_star=sdiag.get("lambda_star"))
    return out


if __name__ == "__main__":
    # tiny smoke test (uses networkx fallback if no leidenalg installed)
    import networkx as nx
    G = nx.karate_club_graph()
    G = nx.relabel_nodes(G, {i: f"e{i}" for i in G.nodes()})
    traj = ["e0", "e1", "e2", "e3", "e7", "e13", "e19", "e33"]
    print(compute_agee(traj, G))
