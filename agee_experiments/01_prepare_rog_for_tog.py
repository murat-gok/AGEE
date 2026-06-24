#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — RoG -> ToG Veri Hazırlama
=============================================================================
Bu script iki şey üretir:
  1) ToG'un beklediği veri dosyası (../data/WebQSP.json veya ../data/cwq.json)
     - Her örnek: {ID, RawQuestion/question, topic_entity, answer}
     - topic_entity formatı: {entity_adı: entity_adı}  (ad=kimlik)
  2) Shim için subgraph indeks dosyası (rog_subgraphs_<dataset>.jsonl)
     - Her satır: {ID, graph}  -> shim bunu set_active_subgraph'ta kullanır

ToG'un soru-alan adı:
  - webqsp -> 'RawQuestion'   (prepare_dataset böyle bekliyor)
  - cwq    -> 'question'

Kullanım:
  python 01_prepare_rog_for_tog.py --dataset webqsp --split test --limit 200
  python 01_prepare_rog_for_tog.py --dataset cwq --split test --limit 200

Çıktılar:
  ToG/data/WebQSP.json          (ToG'un okuyacağı)
  ToG/data/cwq.json
  agee_experiments/rog_subgraphs_webqsp.jsonl   (shim indeksi)
  agee_experiments/rog_subgraphs_cwq.jsonl
=============================================================================
"""
from __future__ import annotations
import os, json, argparse

_DATASET_MAP = {"webqsp": "rmanluo/RoG-webqsp", "cwq": "rmanluo/RoG-cwq"}
_QSTRING = {"webqsp": "RawQuestion", "cwq": "question"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["webqsp", "cwq"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="ilk N soru (pilot)")
    ap.add_argument("--tog-data-dir", default="ToG/data",
                    help="ToG'un veri klasörü (WebQSP.json buraya yazılır)")
    ap.add_argument("--shim-index-dir", default=".",
                    help="shim subgraph indeksinin yazılacağı yer")
    args = ap.parse_args()

    from datasets import load_dataset
    print(f"{_DATASET_MAP[args.dataset]} ({args.split}) yükleniyor...")
    ds = load_dataset(_DATASET_MAP[args.dataset], split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds)} soru işlenecek")

    qstr = _QSTRING[args.dataset]
    tog_data = []
    os.makedirs(args.tog_data_dir, exist_ok=True)
    os.makedirs(args.shim_index_dir, exist_ok=True)

    shim_path = os.path.join(args.shim_index_dir,
                             f"rog_subgraphs_{args.dataset}.jsonl")
    n_empty = 0
    with open(shim_path, "w", encoding="utf-8") as shim_f:
        for i, ex in enumerate(ds):
            qid = str(ex.get("id") or ex.get("qid") or i)
            question = ex.get("question", "")
            graph = ex.get("graph") or []
            q_entities = ex.get("q_entity") or []
            answers = ex.get("answer") or ex.get("a_entity") or []

            if not graph:
                n_empty += 1

            # topic_entity: ad=kimlik formatı
            if isinstance(q_entities, str):
                q_entities = [q_entities]
            topic_entity = {str(e): str(e) for e in q_entities if e}
            # topic entity yoksa, graf içindeki ilk düğümü kullan (fallback)
            if not topic_entity and graph:
                first = graph[0][0]
                topic_entity = {str(first): str(first)}

            tog_item = {
                "ID": qid,
                qstr: question,
                "question": question,        # her iki anahtar da bulunsun
                "topic_entity": topic_entity,
                "answer": answers,
            }
            tog_data.append(tog_item)

            # shim indeksi: sadece ID + graph
            shim_f.write(json.dumps(
                {"ID": qid, "graph": graph}, ensure_ascii=False) + "\n")

    # ToG veri dosyası adı (prepare_dataset ile eşleşmeli)
    fname = {"webqsp": "WebQSP.json", "cwq": "cwq.json"}[args.dataset]
    tog_path = os.path.join(args.tog_data_dir, fname)
    with open(tog_path, "w", encoding="utf-8") as f:
        json.dump(tog_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] ToG veri dosyası -> {tog_path} ({len(tog_data)} soru)")
    print(f"[OK] Shim subgraph indeksi -> {shim_path}")
    if n_empty:
        print(f"[uyarı] {n_empty} sorunun grafı boş (ToG bunlarda boş döner)")
    print(f"\nToG soru-alan adı bu dataset için: '{qstr}'")
    print("Sonraki adım: freebase_func.py yamasını uygulayıp ToG'u çalıştırın.")


if __name__ == "__main__":
    main()
