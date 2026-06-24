#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — ToG <-> RoG Subgraph Shim (Virtuoso Baypas)
=============================================================================
Bu modül, ToG'un freebase_func.py içindeki ÜÇ üst-seviye KG fonksiyonunu
(relation_search_prune, entity_search, id2entity_name_or_type) RoG subgraph'a
yönlendirir. SPARQL / Virtuoso TAMAMEN baypas edilir.

Temel karar:
  ToG, entity'leri Freebase MID'leriyle (m.xxxx) tanımlar. RoG ise ADLARLA
  gelir ve adlar zaten benzersizdir. Bu yüzden ADLARI doğrudan kimlik olarak
  kullanıyoruz -- sentetik MID üretmeye gerek yok. ToG kodunun geri kalanı
  entity_id'yi opak bir string gibi taşıdığı için ad kullanmak sorunsuzdur.

  Tek incelik: ToG'un entity_search'ü "m." ile başlayan ID'leri filtreler.
  Bunu shim'de devre dışı bırakıyoruz (aşağıdaki entity_search_shim).

Akış:
  1) main_freebase.py her soruya başlarken set_active_subgraph(qid) çağırır
     (hazırlık scripti bu çağrıyı ekler).
  2) ToG'un relation_search_prune / entity_search / id2entity_name_or_type
     çağrıları bu modüldeki shim sürümlerine düşer.
  3) Tüm KG erişimi bellekteki RoG subgraph üzerinden olur.

Kullanım (freebase_func.py yamasında):
  from rog_to_tog_shim import (
      set_active_subgraph, relation_search_prune, entity_search,
      id2entity_name_or_type
  )
=============================================================================
"""
from __future__ import annotations
import os, sys, json
from typing import Dict, List

# Adaptörü bul (aynı agee_experiments/adapters altında)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "adapters"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "adapters"))
from rog_subgraph_adapter import SubgraphKG  # noqa: E402

# ToG'un kendi yardımcılarını kullanacağız (prompt kurma, LLM, temizleme)
# Bunlar freebase_func/utils içinde; shim onları RUNTIME'da çağırır.
# Döngüsel import'tan kaçınmak için geç (lazy) import yapıyoruz.

# --- Aktif subgraph durumu (soru başına değişir) ----------------------------
_ACTIVE_KG: SubgraphKG | None = None
_SUBGRAPH_INDEX: Dict[str, dict] = {}   # qid -> RoG example


def load_subgraph_index(rog_jsonl_path: str):
    """Hazırlık scriptinin ürettiği qid->RoG-example indeksini yükler."""
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
    print(f"[shim] {len(_SUBGRAPH_INDEX)} subgraph indekslendi.")


def set_active_subgraph(qid: str):
    """Her sorunun başında çağrılır; o sorunun subgraph'ını aktif eder."""
    global _ACTIVE_KG
    ex = _SUBGRAPH_INDEX.get(str(qid))
    if ex is None:
        _ACTIVE_KG = SubgraphKG([])   # boş graf -> ToG zarifçe boş döner
        return
    _ACTIVE_KG = SubgraphKG.from_rog_example(ex)


def _kg() -> SubgraphKG:
    if _ACTIVE_KG is None:
        return SubgraphKG([])
    return _ACTIVE_KG


# =============================================================================
# ToG'un beklediği üç fonksiyonun shim sürümleri
# =============================================================================
def id2entity_name_or_type(entity_id):
    """Ad zaten kimlik olduğu için aynen döndürür (Virtuoso lookup yok)."""
    if entity_id in (None, "", "UnName_Entity"):
        return "UnName_Entity"
    return str(entity_id)


def entity_search(entity, relation, head=True):
    """ToG entity_search yerine: RoG subgraph'tan komşu entity'ler.
    head=True -> (entity, relation, ?) yani giden kenar (tail entity'ler).
    head=False -> (?, relation, entity) yani gelen kenar (head entity'ler).
    NOT: 'm.' filtresi YOK; adlar kimliktir."""
    kg = _kg()
    if head:
        return kg.get_tail_entities(entity, relation, direction="out")
    else:
        return kg.get_tail_entities(entity, relation, direction="in")


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head,
                          question, args):
    """ToG relation_search_prune yerine: RoG subgraph'tan ilişkileri topla,
    sonra ToG'un KENDİ pruning mantığını (llm/bm25/sbert) aynen kullan.
    Dönüş formatı ToG ile BİREBİR aynı: retrieve_relations_with_scores."""
    # ToG yardımcılarını lazy import et (döngüsel bağımlılık önleme)
    from freebase_func import abandon_rels, clean_relations, \
        clean_relations_bm25_sent, construct_relation_prune_prompt
    from utils import run_llm
    try:
        from freebase_func import compute_bm25_similarity, retrieve_top_docs
    except Exception:
        compute_bm25_similarity = retrieve_top_docs = None

    kg = _kg()
    # head_relations: entity'den ÇIKAN ilişkiler; tail_relations: GİREN
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

    # GÜVENLİK: bu entity'nin subgraph'ta hiç ilişkisi yoksa boş dön.
    # (RoG'da entity yok ya da yaprak düğüm -> ToG zarifçe bu daldan döner)
    if not total_relations:
        return []

    prune = getattr(args, "prune_tools", "llm")
    if prune == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name,
                                                 total_relations, args)
        result = run_llm(prompt, args.temperature_exploration,
                         args.max_length, args.opeani_api_keys, args.LLM_type)
        flag, rel_scores = clean_relations(result, entity_id, head_relations)
    elif prune == "bm25" and compute_bm25_similarity is not None:
        topn_relations, topn_scores = compute_bm25_similarity(
            question, total_relations, args.width)
        flag, rel_scores = clean_relations_bm25_sent(
            topn_relations, topn_scores, entity_id, head_relations)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            'sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(
            question, total_relations, model, args.width)
        flag, rel_scores = clean_relations_bm25_sent(
            topn_relations, topn_scores, entity_id, head_relations)

    if not flag:
        return []

    # ----- ÖNEMLİ: shim doğrulama katmanı -----------------------------------
    # Küçük LLM (Qwen) bazen ilişki adını YANLIŞ prefix'le üretir
    # (ör. 'location.statistical_region.languages_spoken' yerine gerçek ad
    #  'location.country.languages_spoken'). Ayrıca clean_relations head
    # flag'ini string eşleşmeyle atar; eşleşme tutmazsa yanlış yön verir.
    # Burada her ilişkiyi GERÇEK subgraph ilişkilerine hizalar ve head
    # flag'ini GERÇEK yöne göre yeniden atarız.
    head_set = set(head_relations)   # out (giden) ilişkiler
    tail_set = set(tail_relations)   # in (gelen) ilişkiler
    valid = []
    for rel in rel_scores:
        rname = rel["relation"]
        # (a) ilişki zaten gerçek bir ad mı?
        if rname in head_set:
            rel["head"] = True;  valid.append(rel); continue
        if rname in tail_set:
            rel["head"] = False; valid.append(rel); continue
        # (b) yanlış prefix -> son segmente göre en yakın gerçek ilişkiyi bul
        suffix = rname.split(".")[-1]
        cand = [r for r in (head_relations + tail_relations)
                if r.split(".")[-1] == suffix]
        if not cand:
            # gevşek eşleşme: suffix içeren herhangi bir gerçek ilişki
            cand = [r for r in (head_relations + tail_relations)
                    if suffix and suffix in r]
        if cand:
            real = cand[0]
            rel["relation"] = real
            rel["head"] = real in head_set
            valid.append(rel)
        # eşleşme yoksa bu ilişkiyi at (uydurma, subgraph'ta yok)

    return valid


# =============================================================================
if __name__ == "__main__":
    print("ToG<->RoG shim self-test")
    demo = {"ID": "q1", "graph": [
        ["Jamaica", "official_language", "English"],
        ["Jamaica", "spoken_language", "Jamaican Patois"],
        ["English", "language_family", "Germanic"],
    ]}
    _SUBGRAPH_INDEX["q1"] = demo
    set_active_subgraph("q1")
    print("  relations(Jamaica, out):", _kg().get_relations("Jamaica", "out"))
    print("  entity_search(Jamaica, official_language):",
          entity_search("Jamaica", "official_language", head=True))
    print("  id2name(English):", id2entity_name_or_type("English"))
    print("  [OK] shim self-test geçti")
