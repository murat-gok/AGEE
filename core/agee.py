"""
AGEECalculator v3 — incorporates all reviewer fixes.

Reports shrinkage intensity lambda per graph (W9 reproducibility).
Uses edge-novelty ratio (W1), Shannon entropy (W4).
Also computes USR as a separate baseline metric (not a component).
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os, yaml

from core.graph_utils import detect_communities
from core.metrics import (
    coverage_v3, info_rate_v3, efficiency_v3,
    edge_novelty_ratio, compute_usr, aggregate_power_mean
)


@dataclass
class AGEEResult:
    agee: float = 0.0
    coverage: float = 0.0
    info_rate: float = 0.0
    efficiency: float = 0.0
    redundancy: float = 0.0       # edge novelty ratio (W1 fix)
    usr: float = 0.0              # USR reported separately as baseline
    shrinkage_lambda: float = 0.0 # HS09 lambda (W9 reproducibility)
    trajectory_length: int = 0
    n_unique_nodes: int = 0
    n_unique_edges: int = 0
    n_unique_communities: int = 0
    agent_name: str = ""
    graph_name: str = ""


DEFAULT_CONFIG = {
    "agee": {
        "power_p": 0.5,
        "epsilon_floor": 0.01,
        "weights": {"coverage": 0.40, "info_rate": 0.35, "efficiency": 0.25},
        "redundancy_delta": 0.15,
    },
    "info_rate": {"beta": 1.0},
    "graph": {"community_method": "leiden", "resolution": 1.0,
              "random_state": 42},
}


class AGEECalculator:
    def __init__(self, G: nx.Graph, config: Optional[dict] = None,
                 graph_name: str = "unnamed"):
        self.G = G
        self.graph_name = graph_name
        self.config = config or DEFAULT_CONFIG

        for key, val in DEFAULT_CONFIG.items():
            if key not in self.config:
                self.config[key] = val
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    if k2 not in self.config[key]:
                        self.config[key][k2] = v2

        g_cfg = self.config["graph"]
        self.partition = detect_communities(
            G, method=g_cfg["community_method"],
            resolution=g_cfg.get("resolution", 1.0),
            random_state=g_cfg.get("random_state", 42),
        )
        self.n_communities = len(set(self.partition.values())) if self.partition else 1

    def compute(self, trajectory: List[int],
                agent_name: str = "unknown") -> AGEEResult:
        result = AGEEResult()
        result.agent_name = agent_name
        result.graph_name = self.graph_name
        result.trajectory_length = len(trajectory)

        if len(trajectory) == 0:
            return result

        visited = set(trajectory)
        result.n_unique_nodes = len(visited)
        visited_comms = set(self.partition.get(v, 0) for v in visited)
        result.n_unique_communities = len(visited_comms)

        cfg = self.config

        # Coverage (W4: Shannon + HS09)
        result.coverage, result.shrinkage_lambda = coverage_v3(
            trajectory, self.G, self.partition, self.n_communities
        )

        # InfoRate
        result.info_rate = info_rate_v3(
            trajectory, self.G,
            beta=cfg["info_rate"].get("beta", 1.0)
        )

        # Efficiency
        result.efficiency = efficiency_v3(
            trajectory, self.G,
            partition=self.partition,
            n_communities=self.n_communities
        )

        # Redundancy metrics — reported as diagnostics, NOT used in AGEE score
        # W1 FIX: edge_novelty and USR are diagnostic covariates, not components
        result.redundancy = edge_novelty_ratio(trajectory, self.G)
        result.usr = compute_usr(trajectory, n_total_nodes=len(self.G.nodes()))

        # Unique edges traversed
        edges = set()
        for i in range(len(trajectory) - 1):
            u, v = trajectory[i], trajectory[i + 1]
            if self.G.has_edge(u, v):
                edges.add((min(u, v), max(u, v)))
        result.n_unique_edges = len(edges)

        # Aggregation — 3 components only (W1: no R in the score)
        agee_cfg = cfg["agee"]
        w = (agee_cfg["weights"]["coverage"],
             agee_cfg["weights"]["info_rate"],
             agee_cfg["weights"]["efficiency"])
        result.agee = aggregate_power_mean(
            result.coverage, result.info_rate, result.efficiency,
            w=w,
            p=agee_cfg.get("power_p", 0.5),
            epsilon=agee_cfg.get("epsilon_floor", 0.01),
        )

        return result

    def compute_multi(self, trajectories, agent_name="unknown"):
        results = [self.compute(traj, agent_name) for traj in trajectories]
        if not results:
            return results, {}

        def stats(vals):
            a = np.array(vals)
            m, s = np.mean(a), (np.std(a, ddof=1) if len(a) > 1 else 0.0)
            ci = 1.96 * s / np.sqrt(len(a)) if len(a) > 1 else 0.0
            return {"mean": m, "std": s, "ci_low": m - ci, "ci_high": m + ci}

        summary = {
            "agee": stats([r.agee for r in results]),
            "coverage": stats([r.coverage for r in results]),
            "info_rate": stats([r.info_rate for r in results]),
            "efficiency": stats([r.efficiency for r in results]),
            "redundancy": stats([r.redundancy for r in results]),
            "usr": stats([r.usr for r in results]),
            "n_runs": len(results),
        }
        return results, summary
