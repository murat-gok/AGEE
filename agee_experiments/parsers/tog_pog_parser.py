#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — ToG / PoG Trajectory Parser
=============================================================================
Amaç:
    ToG ve PoG, her soru için derinliğe göre yuvalanmış üçlü yapısı üretir:
        reasoning_chains / cluster_chain_of_entities =
            [ depth_0: [ chain: [(ent, rel, ent), ...], ... ],
              depth_1: [ ... ], ... ]
    Bu parser, bu yapıyı düzleştirip AGEE'ye beslenecek SIRALI
    ziyaret-edilen-düğüm dizisini (trajectory) çıkarır.

Çıktı formatı (her soru için):
    {
      "qid": ...,
      "question": ...,
      "answer": [...],          # gold cevaplar (Hits@1 için)
      "prediction": "...",       # ajanın cevabı (results alanı)
      "hits1": 0/1,              # eşleşme
      "trajectory": ["Ent1", "Ent2", ...],   # SIRALI, tekrarsızlaştırılmamış
      "visited_set": [...],      # benzersiz düğümler
      "edges": [(h, r, t), ...]  # ziyaret edilen kenarlar (interpretability için)
    }

DİKKAT — şema doğrulaması:
    ToG'un save_2_jsonl çıktısı yüksek olasılıkla şu 3 anahtarı taşır:
        "question", "results", "reasoning_chains"
    PoG, ToG forku olduğu için benzer şema + olası memory/sub-objective
    alanları taşıyabilir. Bu parser ESNEKtir: birden çok olası anahtar adını
    dener. Yine de İLK ÇALIŞTIRMADA bir satırı açıp şemayı doğrulayın:
        python tog_pog_parser.py --inspect ToG_webqsp.jsonl
=============================================================================
"""
from __future__ import annotations
import json
import argparse
import re
from typing import List, Dict, Any, Tuple, Optional

# reasoning_chains'in bulunabileceği olası anahtar adları (öncelik sırasıyla)
_CHAIN_KEYS = ["reasoning_chains", "cluster_chain_of_entities",
               "chains", "reasoning_chain", "paths"]
# cevap/sonuç alanı için olası anahtarlar
_PRED_KEYS = ["results", "result", "prediction", "answer_pred", "response"]
# gold cevap için olası anahtarlar
_GOLD_KEYS = ["answer", "answers", "gold", "ground_truth", "a_entity"]
# soru için
_Q_KEYS = ["question", "query", "input"]
# soru id için
_QID_KEYS = ["id", "qid", "ID", "question_id"]

_UNNAMED = {"UnName_Entity", "Unknown", "", None}


def _first_key(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def flatten_chains(chains: Any) -> List[Tuple[str, str, str]]:
    """Yuvalanmış reasoning_chains yapısını DERINLIK SIRASINDA düz üçlü
    listesine indirger. Yapı esnekliği için birkaç biçimi tolere eder."""
    triples: List[Tuple[str, str, str]] = []
    if not chains:
        return triples

    def _emit(item):
        # Bir üçlü mü? (ent, rel, ent)
        if (isinstance(item, (list, tuple)) and len(item) == 3
                and all(isinstance(x, str) for x in item)):
            triples.append((item[0], item[1], item[2]))
            return True
        return False

    def _walk(node):
        if _emit(node):
            return
        if isinstance(node, (list, tuple)):
            for sub in node:
                _walk(sub)
        # str / dict gibi yapraklar yok sayılır

    _walk(chains)
    return triples


def trajectory_from_triples(triples: List[Tuple[str, str, str]]
                            ) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """Üçlü listesinden SIRALI düğüm gezinme dizisi üretir.
    Her üçlü için head sonra tail eklenir; ardışık tekrarlar bastırılır
    (gerçek ziyaret sırasını korur, AGEE local-discovery bunu bekler)."""
    seq: List[str] = []
    edges: List[Tuple[str, str, str]] = []
    for h, r, t in triples:
        for ent in (h, t):
            if ent in _UNNAMED:
                continue
            if not seq or seq[-1] != ent:
                seq.append(ent)
        if h not in _UNNAMED and t not in _UNNAMED:
            edges.append((h, r, t))
    return seq, edges


def normalize_answer(s: str) -> str:
    """Hits@1 karşılaştırması için basit normalizasyon (KGQA standardı)."""
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_hits1(prediction: Any, gold: Any) -> int:
    """Tahmin metni gold cevaplardan herhangi birini içeriyorsa 1."""
    if prediction is None or gold is None:
        return 0
    pred_norm = normalize_answer(prediction if isinstance(prediction, str)
                                 else " ".join(map(str, prediction)))
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    for g in golds:
        gn = normalize_answer(g)
        if gn and gn in pred_norm:
            return 1
    return 0


def parse_line(obj: dict) -> Dict[str, Any]:
    """Tek bir JSONL satırını AGEE-hazır kayda dönüştürür."""
    chains = _first_key(obj, _CHAIN_KEYS, default=[])
    triples = flatten_chains(chains)
    seq, edges = trajectory_from_triples(triples)

    pred = _first_key(obj, _PRED_KEYS)
    gold = _first_key(obj, _GOLD_KEYS)
    rec = {
        "qid": _first_key(obj, _QID_KEYS),
        "question": _first_key(obj, _Q_KEYS),
        "answer": gold,
        "prediction": pred,
        "hits1": compute_hits1(pred, gold),
        "trajectory": seq,
        "visited_set": sorted(set(seq)),
        "edges": edges,
        "traj_len": len(seq),
        "n_unique": len(set(seq)),
    }
    return rec


def parse_file(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [uyarı] satır {ln} JSON parse edilemedi, atlandı")
                continue
            records.append(parse_line(obj))
    return records


def inspect_file(path: str, n: int = 1):
    """İlk n satırın HAM şemasını gösterir — şema doğrulaması için."""
    print(f"=== {path} — ilk {n} satır şema incelemesi ===")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(f"\n--- Satır {i} anahtarları: {list(obj.keys())}")
            for k, v in obj.items():
                t = type(v).__name__
                preview = str(v)[:120]
                print(f"    {k} ({t}): {preview}")
            # parse denemesi
            rec = parse_line(obj)
            print(f"\n  -> trajectory uzunluğu: {rec['traj_len']}, "
                  f"benzersiz: {rec['n_unique']}, hits1: {rec['hits1']}")
            print(f"  -> ilk 8 düğüm: {rec['trajectory'][:8]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ToG/PoG trajectory parser")
    ap.add_argument("input", help="ToG_*.jsonl veya PoG_*.jsonl")
    ap.add_argument("--inspect", action="store_true",
                    help="Sadece şemayı incele (ilk çalıştırmada kullanın)")
    ap.add_argument("--out", default=None, help="AGEE-hazır JSON çıktı yolu")
    args = ap.parse_args()

    if args.inspect:
        inspect_file(args.input, n=2)
    else:
        recs = parse_file(args.input)
        hits = sum(r["hits1"] for r in recs)
        print(f"Toplam {len(recs)} kayıt | Hits@1 = {hits}/{len(recs)} "
              f"= {hits/max(len(recs),1):.3f}")
        out = args.out or (args.input.rsplit(".", 1)[0] + "_agee_ready.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"AGEE-hazır kayıtlar yazıldı -> {out}")
