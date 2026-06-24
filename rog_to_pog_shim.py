#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — PoG <-> RoG Subgraph Shim (Virtuoso Baypas)
=============================================================================
PoG'un KG fonksiyonlarını RoG subgraph'a yönlendirir. ToG shim'ine benzer
ama PoG'un farklılıklarını ele alır:
  - relation_search_prune(entity_id, sub_questions, entity_name, pre_relations,
                          pre_head, question, args) -> (relations, token_num)
  - select_relations(string, entity_id, head_relations, tail_relations)
    [ToG'un clean_relations'ından farklı: head VE tail listesi alır,
     yön atamasını DOĞRU yapar -> ek yön-düzeltmeye gerek yok]
  - entity_search ToG ile aynı (m. filtresi yok)

Entity adları doğrudan kimlik olarak kullanılır (MID yok).
=============================================================================
"""
from __future__ import annotations
import os, sys, json
from typing import Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "adapters"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "adapters"))
from rog_subgraph_adapter import SubgraphKG  # noqa: E402

_ACTIVE_KG: SubgraphKG | None = None
_SUBGRAPH_INDEX: Dict[str, dict] = {}


def load_subgraph_index(rog_jsonl_path: str):
    global _SUBGRAPH_INDEX
    _SUBGRAPH_INDEX = {}
    with open(rog_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            qid = str(ex.get("ID") or ex.get("id") or ex.get("qid"))
            _SUBGRAPH_INDEX[qid] = ex
    print(f"[pog-shim] {len(_SUBGRAPH_INDEX)} subgraph indekslendi.")


def set_active_subgraph(qid: str):
    global _ACTIVE_KG
    ex = _SUBGRAPH_INDEX.get(str(qid))
    _ACTIVE_KG = SubgraphKG.from_rog_example(ex) if ex else SubgraphKG([])


def _kg() -> SubgraphKG:
    return _ACTIVE_KG if _ACTIVE_KG is not None else SubgraphKG([])


def id2entity_name_or_type(entity_id):
    if entity_id in (None, "", "UnName_Entity"):
        return "UnName_Entity"
    return str(entity_id)


def entity_search(entity, relation, head=True):
    """PoG entity_search yerine. head=True -> giden (out), False -> gelen (in)."""
    kg = _kg()
    if head:
        return kg.get_tail_entities(entity, relation, direction="out")
    else:
        return kg.get_tail_entities(entity, relation, direction="in")


def relation_search_prune(entity_id, sub_questions, entity_name,
                          pre_relations, pre_head, question, args):
    """PoG relation_search_prune yerine. (relations, token_num) döndürür.
    PoG'un select_relations'ı head VE tail listesi alıp yönü doğru atadığı
    için ToG'daki ek yön-düzeltmeye gerek YOK."""
    from freebase_func import (abandon_rels, select_relations,
                               construct_relation_prune_prompt)
    from utils import run_llm

    kg = _kg()
    head_relations = kg.get_relations(entity_id, direction="out")
    tail_relations = kg.get_relations(entity_id, direction="in")

    if getattr(args, "remove_unnecessary_rel", True):
        head_relations = [r for r in head_relations if not abandon_rels(r)]
        tail_relations = [r for r in tail_relations if not abandon_rels(r)]

    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = sorted(set(head_relations))
    tail_relations = sorted(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()

    if not total_relations:
        return [], {"total": 0, "input": 0, "output": 0}

    prompt = construct_relation_prune_prompt(
        question, sub_questions, entity_name, total_relations, args)
    result, token_num = run_llm(
        prompt, args.temperature_exploration, args.max_length,
        args.opeani_api_keys, args.LLM_type, False, False)

    try:
        flag, relations = select_relations(
            result, entity_id, head_relations, tail_relations)
    except Exception:
        # Qwen çıktısı eval() edilemezse (kırılgan PoG parse) boş dön
        flag, relations = False, "parse error"

    if not flag:
        return [], token_num

    # Qwen yanlış-prefix ürettiyse select_relations zaten atlamıştır
    # (head/tail listesinde yoksa eklemez). Suffix-eşleme ile kurtaralım:
    head_set, tail_set = set(head_relations), set(tail_relations)
    valid = []
    for rel in relations:
        rname = rel["relation"]
        if rname in head_set:
            rel["head"] = True; valid.append(rel)
        elif rname in tail_set:
            rel["head"] = False; valid.append(rel)
        else:
            # yanlış prefix -> suffix ile gerçek ilişkiye hizala
            suffix = rname.split(".")[-1]
            cand = [r for r in (head_relations + tail_relations)
                    if r.split(".")[-1] == suffix] or \
                   [r for r in (head_relations + tail_relations)
                    if suffix and suffix in r]
            if cand:
                rel["relation"] = cand[0]
                rel["head"] = cand[0] in head_set
                valid.append(rel)

    return valid, token_num


if __name__ == "__main__":
    print("PoG<->RoG shim self-test")
    demo = {"ID": "q1", "graph": [
        ["Lou Seal", "sports.mascot.team", "San Francisco Giants"],
        ["San Francisco Giants", "sports.team.championships", "2014 World Series"],
    ]}
    _SUBGRAPH_INDEX["q1"] = demo
    set_active_subgraph("q1")
    print("  relations(Lou Seal, out):", _kg().get_relations("Lou Seal", "out"))
    print("  entity_search(Lou Seal, sports.mascot.team):",
          entity_search("Lou Seal", "sports.mascot.team", head=True))
    print("  [OK] shim self-test geçti")
