#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — ToG Crash-Safe Resumable Runner
=============================================================================
ToG'u gece boyu güvenle koşturmak için. ToG'un main_freebase.py döngüsünü
DEĞİŞTİRMEZ; onun yerine ToG modüllerini import edip ana döngüyü BURADA,
her soruyu try/except + checkpoint ile çalıştırır.

Avantajlar:
  - Bir soruda hata olursa (IndexError vb.) o soru ATLANIR, koşu DEVAM EDER.
  - İşlenen her soru checkpoint'e yazılır; kesinti olursa KALDIĞI YERDEN devam.
  - Hatalı sorular ayrı loglanır (sonra incelemek için).
  - Boş trajectory'li sorular da kaydedilir (AGEE'de "keşif yok" olarak işlenir).

ÇALIŞTIRMA (ToG/ToG/ klasöründen):
  cd /mnt/c/Users/user/metricAGEE/agee_experiments/ToG/ToG
  python /mnt/c/Users/user/metricAGEE/agee_experiments/scripts/03_run_tog_safe.py \
      --dataset webqsp --width 3 --depth 3 --LLM_type qwen2.5:7b \
      --prune_tools llm

  (Argümanlar ToG'un main_freebase.py'siyle aynı.)

ÇIKTILAR (ToG/ToG/ içinde):
  ToG_webqsp.jsonl            # ToG'un normal çıktısı (save_2_jsonl ile aynı)
  checkpoint_webqsp.txt       # işlenen ID'ler (resume için)
  failed_webqsp.jsonl         # hata veren sorular + traceback
=============================================================================
"""
from __future__ import annotations
import os, sys, argparse, json, traceback, random
from tqdm import tqdm

# ToG modüllerinin bu klasörde olduğunu varsayıyoruz (ToG/ToG/)
sys.path.insert(0, os.getcwd())


def build_args():
    """ToG'un main_freebase.py'sindeki argparse'ı birebir taklit eder."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="webqsp")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--temperature_exploration", type=float, default=0.4)
    p.add_argument("--temperature_reasoning", type=float, default=0)
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--remove_unnecessary_rel", type=bool, default=True)
    p.add_argument("--LLM_type", type=str, default="gpt-3.5-turbo")
    p.add_argument("--opeani_api_keys", type=str, default="")
    p.add_argument("--num_retain_entity", type=int, default=5)
    p.add_argument("--prune_tools", type=str, default="llm")
    # resume kontrolü
    p.add_argument("--no-resume", action="store_true",
                   help="checkpoint'i yok say, baştan başla")
    return p.parse_args()


def load_checkpoint(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)
    return done


_ACTIVE_QID = ""   # half_stop gibi dolaylı save_2_jsonl çağrıları için


def _save_with_id(qid, question, answer, cluster_chain_of_entities, dataset):
    """ToG'un save_2_jsonl'i ID yazmıyor. Bu sürüm ID + answer'ı da yazar,
    böylece parser/AGEE RoG ile sağlam eşleşir.
    Şema: {ID, question, results, reasoning_chains}"""
    import json as _json
    rec = {"ID": qid, "question": question, "results": answer,
           "reasoning_chains": cluster_chain_of_entities}
    with open(f"ToG_{dataset}.jsonl", "a", encoding="utf-8") as f:
        f.write(_json.dumps(rec, ensure_ascii=False) + "\n")


def make_save_override(dataset):
    """ToG'un save_2_jsonl imzasıyla uyumlu, ama ID'yi global'den ekleyen
    override. half_stop gibi dolaylı çağrıları da yakalar."""
    def _override(question, answer, cluster_chain_of_entities, file_name=None):
        _save_with_id(_ACTIVE_QID, question, answer,
                      cluster_chain_of_entities, dataset)
    return _override


def process_one_question(data, question_string, args,
                         tog):
    """ToG'un tek bir soru için ana döngüsünü çalıştırır.
    main_freebase.py'deki döngü gövdesinin birebir kopyası (sarmalanmış).
    `tog` = ToG modüllerinin fonksiyonlarını taşıyan namespace."""
    global _ACTIVE_QID
    question = data[question_string]
    qid = str(data.get("ID", ""))
    _ACTIVE_QID = qid
    tog.set_active_subgraph(qid)
    topic_entity = data['topic_entity']
    cluster_chain_of_entities = []

    if len(topic_entity) == 0:
        results = tog.generate_without_explored_paths(question, args)
        tog.save_2_jsonl(question, results, [], file_name=args.dataset)
        return

    pre_relations = []
    pre_heads = [-1] * len(topic_entity)
    flag_printed = False

    for depth in range(1, args.depth + 1):
        current_entity_relations_list = []
        i = 0
        for entity in topic_entity:
            if entity != "[FINISH_ID]":
                # GÜVENLİK: pre_heads uzunluğu topic_entity ile uyumsuzsa -1 kullan
                ph = pre_heads[i] if i < len(pre_heads) else -1
                rel_scores = tog.relation_search_prune(
                    entity, topic_entity[entity], pre_relations, ph,
                    question, args)
                current_entity_relations_list.extend(rel_scores)
            i += 1

        total_candidates, total_scores, total_relations = [], [], []
        total_entities_id, total_topic_entities, total_head = [], [], []

        for entity in current_entity_relations_list:
            if entity['head']:
                ec_id = tog.entity_search(entity['entity'], entity['relation'], True)
            else:
                ec_id = tog.entity_search(entity['entity'], entity['relation'], False)
            if args.prune_tools == "llm":
                if len(ec_id) >= 20:
                    ec_id = random.sample(ec_id, args.num_retain_entity)
            if len(ec_id) == 0:
                continue
            scores, ec, ec_id = tog.entity_score(
                question, ec_id, entity['score'], entity['relation'], args)
            (total_candidates, total_scores, total_relations,
             total_entities_id, total_topic_entities, total_head) = \
                tog.update_history(ec, entity, scores, ec_id,
                                   total_candidates, total_scores,
                                   total_relations, total_entities_id,
                                   total_topic_entities, total_head)

        if len(total_candidates) == 0:
            tog.half_stop(question, cluster_chain_of_entities, depth, args)
            flag_printed = True
            break

        flag, chain, entities_id, pre_relations, pre_heads = tog.entity_prune(
            total_entities_id, total_relations, total_candidates,
            total_topic_entities, total_head, total_scores, args)
        cluster_chain_of_entities.append(chain)

        if flag:
            stop, results = tog.reasoning(question, cluster_chain_of_entities, args)
            if stop:
                tog.save_2_jsonl(question, results,
                                 cluster_chain_of_entities, file_name=args.dataset)
                flag_printed = True
                break
            else:
                flag_finish, entities_id = tog.if_finish_list(entities_id)
                if flag_finish:
                    tog.half_stop(question, cluster_chain_of_entities, depth, args)
                    flag_printed = True
                else:
                    topic_entity = {e: tog.id2entity_name_or_type(e)
                                    for e in entities_id}
                    continue
        else:
            tog.half_stop(question, cluster_chain_of_entities, depth, args)
            flag_printed = True

    if not flag_printed:
        results = tog.generate_without_explored_paths(question, args)
        tog.save_2_jsonl(question, results, [], file_name=args.dataset)


def main():
    args = build_args()

    # ToG modüllerini import et (yamalı freebase_func + utils)
    import importlib
    tog = importlib.import_module("freebase_func")  # shim override'lı
    utils = importlib.import_module("utils")
    # Gerekli fonksiyonları tek namespace'te topla
    for name in ["entity_search", "entity_score", "update_history",
                 "entity_prune", "reasoning", "relation_search_prune",
                 "id2entity_name_or_type", "set_active_subgraph",
                 "load_subgraph_index", "half_stop",
                 "generate_without_explored_paths"]:
        if hasattr(tog, name):
            setattr(tog, name, getattr(tog, name))
        elif hasattr(utils, name):
            setattr(tog, name, getattr(utils, name))
    for name in ["save_2_jsonl", "prepare_dataset", "if_finish_list",
                 "generate_without_explored_paths"]:
        if not hasattr(tog, name) and hasattr(utils, name):
            setattr(tog, name, getattr(utils, name))

    # Subgraph indeksini yükle
    tog.load_subgraph_index(f"rog_subgraphs_{args.dataset}.jsonl")

    # save_2_jsonl'i ID-aware sürümle override et (HEM doğrudan HEM half_stop
    # gibi dolaylı çağrılar için her iki modülde de değiştir):
    _override = make_save_override(args.dataset)
    for mod in (tog, utils):
        try:
            setattr(mod, "save_2_jsonl", _override)
        except Exception:
            pass

    datas, question_string = utils.prepare_dataset(args.dataset)
    print(f"Toplam {len(datas)} soru. Crash-safe runner başlıyor.")

    ckpt_path = f"checkpoint_{args.dataset}.txt"
    failed_path = f"failed_{args.dataset}.jsonl"
    done = set() if args.no_resume else load_checkpoint(ckpt_path)
    if done:
        print(f"  Resume: {len(done)} soru zaten işlenmiş, atlanacak.")

    n_ok, n_fail, n_skip = 0, 0, 0
    ckpt_f = open(ckpt_path, "a", encoding="utf-8")

    for data in tqdm(datas):
        qid = str(data.get("ID", ""))
        if qid in done:
            n_skip += 1
            continue
        try:
            process_one_question(data, question_string, args, tog)
            ckpt_f.write(qid + "\n"); ckpt_f.flush()
            n_ok += 1
        except Exception as e:                      # noqa
            n_fail += 1
            tb = traceback.format_exc()
            with open(failed_path, "a", encoding="utf-8") as ff:
                ff.write(json.dumps({
                    "ID": qid,
                    "question": data.get(question_string, ""),
                    "error": str(e),
                    "traceback": tb,
                }, ensure_ascii=False) + "\n")
            # Hatalı soruyu da boş trajectory ile kaydet (AGEE eksiksiz olsun)
            try:
                globals()["_ACTIVE_QID"] = qid
                _save_with_id(qid, data.get(question_string, ""), "[ERROR]",
                              [], args.dataset)
            except Exception:
                pass
            ckpt_f.write(qid + "\n"); ckpt_f.flush()  # tekrar denememek için
            print(f"\n  [atlandı] {qid}: {e}")

    ckpt_f.close()
    print(f"\n=== BİTTİ ===")
    print(f"  Başarılı: {n_ok} | Hatalı(atlandı): {n_fail} | "
          f"Resume-atlandı: {n_skip}")
    print(f"  Çıktı: ToG_{args.dataset}.jsonl")
    if n_fail:
        print(f"  Hatalı sorular: {failed_path} (inceleyebilirsiniz)")


if __name__ == "__main__":
    main()
