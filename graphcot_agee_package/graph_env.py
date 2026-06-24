"""
Graph environment.

Two views of the same RoG subgraph are produced:

1. A Graph-CoT-style structure ``graph[node_type][nid] = {features, neighbors}``
   plus graph-function helpers (check_neighbours / check_nodes / check_degree)
   and a lightweight string Retriever. This is what the *agent* interacts with
   and is faithful to PeterGriffinJin/Graph-CoT's ``tools/graph_funcs.py``.

2. An undirected, unweighted simple graph (networkx) over the same entities.
   This is the graph ``G`` that *AGEE* is computed on -- matching the core
   AGEE definition (Sec. 5: "undirected graph G=(V,E)"). Relation labels and
   edge direction are intentionally collapsed here.  [MUST-MATCH] this is the
   same collapse your ToG/PoG AGEE computation uses.
"""
import re
import networkx as nx

NODE_TYPE = "entity"


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# View 1: Graph-CoT interaction structure
# ---------------------------------------------------------------------------
def build_graphcot_graph(triples, add_inverse=True):
    """Return (graph, relations_by_node) in Graph-CoT's expected layout.

    Nodes are entity surface strings. Neighbours are keyed by relation; if
    add_inverse, incoming edges are exposed as '~relation' so the agent can
    traverse either direction (standard in KG-QA)."""
    nodes = {}

    def ensure(e):
        if e not in nodes:
            nodes[e] = {"features": {"name": e}, "neighbors": {}}

    for h, r, t in triples:
        ensure(h)
        ensure(t)
        nodes[h]["neighbors"].setdefault(r, [])
        if t not in nodes[h]["neighbors"][r]:
            nodes[h]["neighbors"][r].append(t)
        if add_inverse:
            inv = "~" + r
            nodes[t]["neighbors"].setdefault(inv, [])
            if h not in nodes[t]["neighbors"][inv]:
                nodes[t]["neighbors"][inv].append(h)

    graph = {NODE_TYPE: nodes}
    relations_by_node = {nid: sorted(nodes[nid]["neighbors"].keys()) for nid in nodes}
    return graph, relations_by_node


class GraphFuncs:
    """Faithful re-implementation of Graph-CoT tools/graph_funcs.py."""

    def __init__(self, graph):
        self.index = {}
        for ntype in graph:
            for nid, payload in graph[ntype].items():
                self.index[nid] = payload

    def check_neighbours(self, node, neighbor_type=None):
        if node not in self.index:
            raise KeyError(node)
        nb = self.index[node]["neighbors"]
        if neighbor_type:
            return list(nb.get(neighbor_type, []))
        return nb

    def check_nodes(self, node, feature=None):
        if node not in self.index:
            raise KeyError(node)
        feats = self.index[node]["features"]
        return feats.get(feature, "") if feature else feats

    def check_degree(self, node, neighbor_type):
        if node not in self.index:
            raise KeyError(node)
        return len(self.index[node]["neighbors"].get(neighbor_type, []))

    def relations(self, node):
        if node not in self.index:
            raise KeyError(node)
        return sorted(self.index[node]["neighbors"].keys())


class Retriever:
    """Lightweight string retriever (no dense index -> no GPU/faiss needed).

    Returns the best-matching entity id for a free-text query: exact
    normalised match first, then highest token-overlap / substring score."""

    def __init__(self, node_ids):
        self.node_ids = list(node_ids)
        self._normed = {nid: norm(nid) for nid in self.node_ids}
        self._tokens = {nid: set(self._normed[nid].split()) for nid in self.node_ids}

    def search_single(self, query, k=1):
        q = norm(query)
        qtok = set(q.split())
        # exact
        for nid in self.node_ids:
            if self._normed[nid] == q:
                return nid
        # substring containment (either direction)
        cands = []
        for nid in self.node_ids:
            nn = self._normed[nid]
            score = 0.0
            if q and (q in nn or nn in q):
                score += 2.0
            if qtok:
                score += len(qtok & self._tokens[nid]) / len(qtok)
            if score > 0:
                cands.append((score, nid))
        if not cands:
            return None
        cands.sort(reverse=True)
        return cands[0][1]


# ---------------------------------------------------------------------------
# View 2: undirected simple graph for AGEE
# ---------------------------------------------------------------------------
def build_undirected_graph(triples):
    """Collapse triples to an undirected, unweighted simple graph (no self-loops)."""
    G = nx.Graph()
    for h, r, t in triples:
        G.add_node(h)
        G.add_node(t)
        if h != t:
            G.add_edge(h, t)
    return G
