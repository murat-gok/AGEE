#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — PoG Otomatik Yama Uygulayıcı
=============================================================================
PoG'u Virtuoso yerine RoG subgraph + lokal Ollama ile çalıştırır.
PoG'un KENDİ try/except'i (her soruyu saran) olduğu için crash-safe
runner'a GEREK YOK — PoG'un kendi main_freebase.py'si çalıştırılır.

Yamalar (her birinin .bak yedeği alınır):
  1) PoG/PoG/freebase_func.py
       - relation_search_prune, entity_search, id2entity_name_or_type
         -> PoG shim sürümleriyle override
       - dosya sonuna shim import + subgraph yükleme fonksiyonları
  2) PoG/PoG/utils.py
       - run_llm -> PoG Ollama köprüsü (token_num döndüren)
  3) PoG/PoG/main_freebase.py
       - SentenceTransformer hardcoded yolu -> None (prune_tools llm kullanır)
       - subgraph indeksi yükleme + her soruda set_active_subgraph
       - save_2_jsonl -> ID-aware override (global qid ile)

Kullanım (agee_experiments/ klasöründen):
  python scripts/02_patch_pog.py --pog-root PoG --dataset webqsp
Geri alma:
  python scripts/02_patch_pog.py --pog-root PoG --restore
=============================================================================
"""
from __future__ import annotations
import os, shutil, argparse, sys

_MARK = "# === AGEE_POG_SHIM_PATCH ==="


def _backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak); print(f"  yedek: {bak}")
    else:
        print(f"  yedek zaten var: {bak}")


def _restore(path):
    bak = path + ".bak"
    if os.path.exists(bak):
        shutil.copy2(bak, path); print(f"  geri yüklendi: {path}")
    else:
        print(f"  [uyarı] yedek yok: {bak}")


def patch_freebase_func(pog_pkg, shim_dir):
    path = os.path.join(pog_pkg, "freebase_func.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if _MARK in src:
        print("  freebase_func.py zaten yamalı, atlanıyor."); return
    _backup(path)
    patch = f'''

{_MARK}
import sys as _sys, os as _os
_sys.path.insert(0, r"{shim_dir}")
from rog_to_pog_shim import (
    set_active_subgraph as _set_active_subgraph,
    load_subgraph_index as _load_subgraph_index,
    relation_search_prune as _pog_relation_search_prune,
    entity_search as _pog_entity_search,
    id2entity_name_or_type as _pog_id2entity_name_or_type,
)
relation_search_prune = _pog_relation_search_prune
entity_search = _pog_entity_search
id2entity_name_or_type = _pog_id2entity_name_or_type
set_active_subgraph = _set_active_subgraph
load_subgraph_index = _load_subgraph_index
# === AGEE_POG_SHIM_PATCH SONU ===
'''
    with open(path, "a", encoding="utf-8") as f:
        f.write(patch)
    print("  [OK] freebase_func.py yamalandı.")


def patch_utils(pog_pkg, shim_dir):
    path = os.path.join(pog_pkg, "utils.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if _MARK in src:
        print("  utils.py zaten yamalı, atlanıyor."); return
    _backup(path)
    patch = f'''

{_MARK}
import sys as _sys2, os as _os2
_sys2.path.insert(0, r"{shim_dir}")
from local_llm_bridge_pog import run_llm as _pog_run_llm
run_llm = _pog_run_llm
# === AGEE_POG_SHIM_PATCH SONU ===
'''
    with open(path, "a", encoding="utf-8") as f:
        f.write(patch)
    print("  [OK] utils.py yamalandı (run_llm -> PoG Ollama köprüsü).")


def patch_main(pog_pkg, dataset):
    path = os.path.join(pog_pkg, "main_freebase.py")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    src = "".join(lines)
    if _MARK in src:
        print("  main_freebase.py zaten yamalı, atlanıyor."); return
    _backup(path)

    out = []
    done_st = done_load = done_set = done_save = False
    for line in lines:
        # (a) SentenceTransformer hardcoded yolunu None yap (prune_tools llm)
        if (not done_st) and "SentenceTransformer(" in line and "model" in line:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(f"{indent}{_MARK}\n")
            out.append(f"{indent}model = None  # prune_tools=llm; SBERT baypas\n")
            done_st = True
            continue
        # (b) prepare_dataset'ten sonra subgraph indeksi yükle
        if (not done_load) and "prepare_dataset(args.dataset)" in line:
            out.append(line)
            out.append(
                f'    {_MARK}\n'
                f'    from freebase_func import load_subgraph_index\n'
                f'    load_subgraph_index("rog_subgraphs_{dataset}.jsonl")\n'
            )
            done_load = True
            continue
        # (c) her soruda set_active_subgraph + global qid (save override için)
        if (not done_set) and "topic_entity = data['topic_entity']" in line:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(
                f'{indent}from freebase_func import set_active_subgraph\n'
                f'{indent}import builtins as _b; _b._POG_ACTIVE_QID = str(data.get("ID",""))\n'
                f'{indent}set_active_subgraph(str(data.get("ID","")))\n'
            )
            out.append(line)
            done_set = True
            continue
        out.append(line)

    if not (done_st and done_load and done_set):
        print(f"  [uyarı] beklenen satırlar bulunamadı "
              f"(st={done_st}, load={done_load}, set={done_set}). Elle kontrol.")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"  [OK] main_freebase.py yamalandı (st={done_st}, load={done_load}, set={done_set}).")


def patch_save_in_utils(pog_pkg):
    """save_2_jsonl'i ID ekleyen sürümle değiştir (utils.py içinde)."""
    path = os.path.join(pog_pkg, "utils.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "_POG_SAVE_OVERRIDE" in src:
        print("  save_2_jsonl zaten override'lı."); return
    override = '''

# _POG_SAVE_OVERRIDE
import builtins as _bsave, json as _jsave, time as _tsave
def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities,
                 call_num, all_t, start_time, file_name):
    tt = _tsave.time() - start_time
    qid = getattr(_bsave, "_POG_ACTIVE_QID", "")
    rec = {"ID": qid, question_string: question, "results": answer,
           "reasoning_chains": cluster_chain_of_entities,
           "call_num": call_num, "total_token": all_t.get("total", 0),
           "input_token": all_t.get("input", 0),
           "output_token": all_t.get("output", 0), "time": tt}
    with open(f"PoG_{file_name}.jsonl", "a", encoding="utf-8") as outfile:
        outfile.write(_jsave.dumps(rec, ensure_ascii=False) + "\\n")
# _POG_SAVE_OVERRIDE SONU
'''
    with open(path, "a", encoding="utf-8") as f:
        f.write(override)
    print("  [OK] save_2_jsonl ID-aware override eklendi (utils.py).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pog-root", default="PoG")
    ap.add_argument("--dataset", default="webqsp")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    pog_pkg = os.path.join(args.pog_root, "PoG")
    if not os.path.isdir(pog_pkg):
        pog_pkg = args.pog_root
    if not os.path.exists(os.path.join(pog_pkg, "freebase_func.py")):
        print(f"[HATA] freebase_func.py bulunamadı: {pog_pkg}"); sys.exit(1)

    here = os.path.dirname(os.path.abspath(__file__))
    shim_dir = os.path.abspath(os.path.join(here, "..", "adapters"))

    if args.restore:
        print("PoG yamaları geri alınıyor...")
        for fn in ["freebase_func.py", "utils.py", "main_freebase.py"]:
            _restore(os.path.join(pog_pkg, fn))
        print("Bitti."); return

    print(f"PoG paketi: {pog_pkg}")
    print(f"Shim klasörü: {shim_dir}\n")
    print("1) freebase_func.py"); patch_freebase_func(pog_pkg, shim_dir)
    print("2) utils.py (run_llm)"); patch_utils(pog_pkg, shim_dir)
    print("3) utils.py (save_2_jsonl)"); patch_save_in_utils(pog_pkg)
    print("4) main_freebase.py"); patch_main(pog_pkg, args.dataset)
    print("\n[TAMAM] PoG yamalandı. RoG subgraph + Ollama ile çalışır.")
    print("Çalıştırmadan önce:")
    print("  - local_llm_bridge_pog.py içinde ACTIVE_BACKBONE'u ayarlayın")
    print("  - 01_prepare ile PoG/PoG/ içine veri + subgraph indeksi üretin")


if __name__ == "__main__":
    main()
