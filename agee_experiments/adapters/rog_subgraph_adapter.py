#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — RoG Subgraph Adaptörü (Virtuoso Baypas)
=============================================================================
Amaç:
    ToG ve PoG, tam Freebase'i bir Virtuoso SPARQL sunucusunda bekler
    (~100 GB RAM). 16 GB RAM'lik makinede bu MÜMKÜN DEĞİL.

    Bu adaptör, RoG'un HuggingFace üzerindeki ön-işlenmiş SORU-BAZLI
    subgraph'larını (rmanluo/RoG-webqsp, rmanluo/RoG-cwq) belleğe yükler ve
    ToG/PoG'un ihtiyaç duyduğu KG-erişim fonksiyonlarını yerel olarak sağlar.

    Her soru için subgraph ~1300-1950 düğüm -> 16 GB RAM'e rahat sığar.

Veri formatı (RoG):
    Her örnek şu alanları içerir:
      - id, question, answer (liste)
      - q_entity (topic entity adları, liste)
      - a_entity (cevap entity adları, liste)
      - graph: [[head, relation, tail], ...]  (üçlüler, ADLARLA)

Kullanım:
    from rog_subgraph_adapter import SubgraphKG
    kg = SubgraphKG.from_rog_example(example)
    rels = kg.get_relations("Mel Gibson")           # ilişki listesi
    tails = kg.get_tail_entities("Mel Gibson", "film.actor.film")
    G = kg.to_networkx()                             # AGEE için NetworkX grafı

Not:
    RoG subgraph'ları YÖNSÜZ projeksiyon için kullanılacak (AGEE undirected
    tanımlı). Yön bilgisi ToG/PoG gezinmesi için korunur; AGEE hesabında
    to_networkx(undirected=True) ile yönsüzleştirilir.
=============================================================================
"""
from __future__ import annotations
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Iterable, Optional

try:
    import networkx as nx
except ImportError:
    nx = None


class SubgraphKG:
    """Tek bir sorunun RoG subgraph'ını saran, KG-erişim API'si sunan sınıf."""

    def __init__(self, triples: Iterable[Tuple[str, str, str]]):
        # Yönlü komşuluk: head -> relation -> {tail, ...}
        self._out: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        # Ters yön: tail -> relation -> {head, ...}  (gerektiğinde geri gitme)
        self._in: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._entities: Set[str] = set()
        self._relations: Set[str] = set()
        self._triples: List[Tuple[str, str, str]] = []

        for h, r, t in triples:
            h, r, t = str(h), str(r), str(t)
            self._out[h][r].add(t)
            self._in[t][r].add(h)
            self._entities.add(h); self._entities.add(t)
            self._relations.add(r)
            self._triples.append((h, r, t))

    # ----- Yapıcılar ---------------------------------------------------------
    @classmethod
    def from_rog_example(cls, example: dict) -> "SubgraphKG":
        """RoG HuggingFace örneğinden (graph alanı) KG kurar."""
        graph = example.get("graph") or example.get("subgraph") or []
        triples = []
        for tr in graph:
            if isinstance(tr, (list, tuple)) and len(tr) == 3:
                triples.append((tr[0], tr[1], tr[2]))
        return cls(triples)

    @classmethod
    def from_triple_file(cls, path: str, sep: str = "\t") -> "SubgraphKG":
        """'head <sep> relation <sep> tail' formatlı dosyadan KG kurar
        (MetaQA kb.txt ile uyumlu)."""
        triples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(sep)
                if len(parts) == 3:
                    triples.append(tuple(parts))
        return cls(triples)

    # ----- ToG/PoG'un beklediği temel erişim fonksiyonları -------------------
    def get_relations(self, entity: str, direction: str = "both") -> List[str]:
        """Bir entity'nin (giden/gelen/her iki) ilişkilerini döndürür.
        ToG'daki relation_search adımının yerel karşılığı."""
        rels: Set[str] = set()
        if direction in ("out", "both"):
            rels |= set(self._out.get(entity, {}).keys())
        if direction in ("in", "both"):
            rels |= set(self._in.get(entity, {}).keys())
        return sorted(rels)

    def get_tail_entities(self, entity: str, relation: str,
                          direction: str = "out") -> List[str]:
        """(entity, relation) için komşu entity'leri döndürür.
        ToG'daki entity_search adımının yerel karşılığı."""
        res: Set[str] = set()
        if direction in ("out", "both"):
            res |= self._out.get(entity, {}).get(relation, set())
        if direction in ("in", "both"):
            res |= self._in.get(entity, {}).get(relation, set())
        return sorted(res)

    def get_neighbors(self, entity: str) -> List[str]:
        """Yönden bağımsız tüm komşular (AGEE local discovery için kullanışlı)."""
        nb: Set[str] = set()
        for r, tails in self._out.get(entity, {}).items():
            nb |= tails
        for r, heads in self._in.get(entity, {}).items():
            nb |= heads
        nb.discard(entity)
        return sorted(nb)

    def has_entity(self, entity: str) -> bool:
        return entity in self._entities

    # ----- AGEE için NetworkX köprüsü ----------------------------------------
    def to_networkx(self, undirected: bool = True):
        """AGEE hesabı için graf nesnesi üretir.
        undirected=True -> AGEE'nin yönsüz tanımıyla uyumlu projeksiyon."""
        if nx is None:
            raise ImportError("networkx gerekli: pip install networkx")
        G = nx.Graph() if undirected else nx.DiGraph()
        G.add_nodes_from(self._entities)
        for h, r, t in self._triples:
            if h == t:
                continue  # self-loop'ları AGEE topluluk tespitinde atla
            G.add_edge(h, t, relation=r)
        return G

    # ----- Bilgi ------------------------------------------------------------
    @property
    def num_entities(self) -> int: return len(self._entities)
    @property
    def num_relations(self) -> int: return len(self._relations)
    @property
    def num_triples(self) -> int: return len(self._triples)

    def stats(self) -> dict:
        return {"entities": self.num_entities,
                "relations": self.num_relations,
                "triples": self.num_triples}


# =============================================================================
# Hızlı self-test
# =============================================================================
if __name__ == "__main__":
    print("RoG Subgraph Adaptörü — self-test")
    demo = {
        "question": "who directed the movie X",
        "q_entity": ["Movie X"],
        "a_entity": ["Director Y"],
        "graph": [
            ["Movie X", "directed_by", "Director Y"],
            ["Movie X", "starred_actors", "Actor A"],
            ["Director Y", "directed", "Movie Z"],
            ["Actor A", "acted_in", "Movie Z"],
        ],
    }
    kg = SubgraphKG.from_rog_example(demo)
    print("  stats:", kg.stats())
    print("  relations(Movie X):", kg.get_relations("Movie X"))
    print("  tails(Movie X, directed_by):", kg.get_tail_entities("Movie X", "directed_by"))
    print("  neighbors(Movie X):", kg.get_neighbors("Movie X"))
    if nx is not None:
        G = kg.to_networkx()
        print(f"  networkx: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    print("  [OK] self-test geçti")
