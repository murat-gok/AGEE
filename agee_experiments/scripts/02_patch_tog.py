#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — ToG Otomatik Yama Uygulayıcı
=============================================================================
ToG'u Virtuoso yerine RoG subgraph + lokal Ollama ile çalışacak hale getirir.
Üç dosyayı yamalar (her birinin .bak yedeğini alır):

  1) ToG/ToG/freebase_func.py
       - relation_search_prune, entity_search, id2entity_name_or_type
         -> shim sürümleriyle değiştirilir (dosya sonuna shim import + override)
  2) ToG/ToG/utils.py
       - run_llm gövdesi -> lokal Ollama köprüsü (local_llm_bridge)
  3) ToG/ToG/main_freebase.py
       - her sorunun başında set_active_subgraph(ID) çağrısı eklenir
       - shim subgraph indeksi yüklenir

Strateji: Orijinal fonksiyonları SİLMİYORUZ. Bunun yerine dosya sonuna
"from shim import ... as ..." ekleyip Python'un isim çözümlemesiyle
override ediyoruz. Böylece geri dönüş kolay (sadece .bak'ı geri koy).

Kullanım (agee_experiments/ klasöründen):
  python scripts/02_patch_tog.py --tog-root ToG

Geri alma:
  python scripts/02_patch_tog.py --tog-root ToG --restore
=============================================================================
"""
from __future__ import annotations
import os, shutil, argparse, sys

# Yamaların başına yazılacak işaret (idempotent kontrol için)
_MARK = "# === AGEE_SHIM_PATCH ==="


def _backup(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"  yedek: {bak}")
    else:
        print(f"  yedek zaten var: {bak}")


def _restore(path):
    bak = path + ".bak"
    if os.path.exists(bak):
        shutil.copy2(bak, path)
        print(f"  geri yüklendi: {path}")
    else:
        print(f"  [uyarı] yedek yok: {bak}")


def patch_freebase_func(tog_pkg, shim_dir):
    path = os.path.join(tog_pkg, "freebase_func.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if _MARK in src:
        print("  freebase_func.py zaten yamalı, atlanıyor.")
        return
    _backup(path)
    patch = f'''

{_MARK}
# ToG'un KG fonksiyonlarını RoG subgraph shim'iyle override et (Virtuoso baypas)
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(r"{shim_dir}"))
from rog_to_tog_shim import (
    set_active_subgraph as _set_active_subgraph,
    load_subgraph_index as _load_subgraph_index,
    relation_search_prune as _rog_relation_search_prune,
    entity_search as _rog_entity_search,
    id2entity_name_or_type as _rog_id2entity_name_or_type,
)
# İsimleri ToG'un kullandığı adlarla override et:
relation_search_prune = _rog_relation_search_prune
entity_search = _rog_entity_search
id2entity_name_or_type = _rog_id2entity_name_or_type
set_active_subgraph = _set_active_subgraph
load_subgraph_index = _load_subgraph_index
# === AGEE_SHIM_PATCH SONU ===
'''
    with open(path, "a", encoding="utf-8") as f:
        f.write(patch)
    print(f"  [OK] freebase_func.py yamalandı.")


def patch_utils(tog_pkg, shim_dir):
    path = os.path.join(tog_pkg, "utils.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if _MARK in src:
        print("  utils.py zaten yamalı, atlanıyor.")
        return
    _backup(path)
    patch = f'''

{_MARK}
# run_llm'i lokal Ollama köprüsüyle override et
import sys as _sys2, os as _os2
_sys2.path.insert(0, _os2.path.join(r"{shim_dir}"))
from local_llm_bridge import run_llm_local as _run_llm_local
run_llm = _run_llm_local
# === AGEE_SHIM_PATCH SONU ===
'''
    with open(path, "a", encoding="utf-8") as f:
        f.write(patch)
    print(f"  [OK] utils.py yamalandı (run_llm -> Ollama).")


def patch_main(tog_pkg, dataset):
    path = os.path.join(tog_pkg, "main_freebase.py")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    src = "".join(lines)
    if _MARK in src:
        print("  main_freebase.py zaten yamalı, atlanıyor.")
        return
    _backup(path)

    out = []
    inserted_load = False
    inserted_set = False
    for line in lines:
        # (a) prepare_dataset çağrısından sonra subgraph indeksini yükle
        if (not inserted_load) and "prepare_dataset(args.dataset)" in line:
            out.append(line)
            out.append(
                f'    {_MARK}\n'
                f'    from freebase_func import load_subgraph_index\n'
                f'    load_subgraph_index("rog_subgraphs_{dataset}.jsonl")\n'
            )
            inserted_load = True
            continue
        # (b) her sorunun topic_entity okunduğu yerden hemen önce set_active
        if (not inserted_set) and "topic_entity = data['topic_entity']" in line:
            indent = line[:len(line) - len(line.lstrip())]
            out.append(
                f'{indent}from freebase_func import set_active_subgraph\n'
                f'{indent}set_active_subgraph(str(data.get("ID", "")))\n'
            )
            out.append(line)
            inserted_set = True
            continue
        out.append(line)

    if not (inserted_load and inserted_set):
        print(f"  [uyarı] beklenen satırlar bulunamadı "
              f"(load={inserted_load}, set={inserted_set}). "
              f"main_freebase.py elle kontrol edin.")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"  [OK] main_freebase.py yamalandı "
          f"(load={inserted_load}, set={inserted_set}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tog-root", default="ToG",
                    help="ToG repo kök klasörü (içinde ToG/ paketi var)")
    ap.add_argument("--dataset", default="webqsp",
                    help="rog_subgraphs_<dataset>.jsonl için ad")
    ap.add_argument("--restore", action="store_true",
                    help="Yamaları geri al (.bak'tan)")
    args = ap.parse_args()

    # ToG paket klasörü: <root>/ToG
    tog_pkg = os.path.join(args.tog_root, "ToG")
    if not os.path.isdir(tog_pkg):
        # bazı klonlarda kök doğrudan paket olabilir
        tog_pkg = args.tog_root
    if not os.path.exists(os.path.join(tog_pkg, "freebase_func.py")):
        print(f"[HATA] freebase_func.py bulunamadı: {tog_pkg}")
        sys.exit(1)

    # shim/köprü dosyalarının mutlak yolu (agee_experiments/adapters)
    here = os.path.dirname(os.path.abspath(__file__))
    shim_dir = os.path.abspath(os.path.join(here, "..", "adapters"))

    if args.restore:
        print("Yamalar geri alınıyor...")
        for fn in ["freebase_func.py", "utils.py", "main_freebase.py"]:
            _restore(os.path.join(tog_pkg, fn))
        print("Bitti.")
        return

    print(f"ToG paketi: {tog_pkg}")
    print(f"Shim klasörü: {shim_dir}\n")
    print("1) freebase_func.py")
    patch_freebase_func(tog_pkg, shim_dir)
    print("2) utils.py")
    patch_utils(tog_pkg, shim_dir)
    print("3) main_freebase.py")
    patch_main(tog_pkg, args.dataset)
    print("\n[TAMAM] Yamalar uygulandı. ToG artık RoG subgraph + Ollama ile çalışır.")
    print("Çalıştırmadan önce:")
    print("  - local_llm_bridge.py içinde ACTIVE_BACKBONE'u ayarlayın")
    print("  - 01_prepare_rog_for_tog.py ile veri + subgraph indeksini üretin")


if __name__ == "__main__":
    main()
