#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
AGEE Revizyon — Ana Orkestrasyon: Trajectory -> AGEE -> Sonuç Tabloları
=============================================================================
Akış:
  1) RoG subgraph'larını yükle (soru başına KG).
  2) ToG/PoG (ve isteğe bağlı BFS/Greedy/RandomWalk) AGEE-hazır kayıtlarını oku.
  3) Her trajectory için ilgili subgraph üzerinde AGEE hesapla.
  4) Ajan-bazlı özet tablo + bileşen-Hits@1 korelasyonları üret.
     (Makaledeki Tablo 4 ve Tablo 5'in WebQSP/CWQ karşılıkları)

AGEE hesabı:
  - ÖNCE sizin metricAGEE paketinizi import etmeye çalışır
    (core.agee.AGEECalculator). Böylece makaledeki v3 formülüyle BİREBİR
    aynı skoru üretir.
  - Bulamazsa, makaledeki tanıma sadık MINIMAL bir referans implementasyonu
    kullanır (yalnızca pipeline'ı uçtan uca test etmek için; nihai
    sayılar metricAGEE ile üretilmelidir).

Kullanım:
  python 10_run_agee_on_trajectories.py \
      --dataset webqsp \
      --rog-split test \
      --agee-ready ToG_webqsp_agee_ready.json \
      --agent ToG \
      --out results_tog_webqsp.csv

  Birden çok ajan/dosyayı tek tabloda birleştirmek için scripti her ajan
  için çalıştırıp CSV'leri birleştirebilir ya da --agee-ready'yi
  'agent:dosya' çiftleriyle tekrarlı verebilirsiniz (aşağıya bkz).
=============================================================================
"""
from __future__ import annotations
import sys, os, json, argparse, math
from typing import List, Dict, Any, Tuple

import numpy as np

# --- yerel modüller ---------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "adapters"))
from rog_subgraph_adapter import SubgraphKG          # noqa: E402

try:
    import networkx as nx
except ImportError:
    print("networkx gerekli: pip install networkx"); sys.exit(1)

# Topluluk tespiti (Louvain)
try:
    import community as community_louvain   # python-louvain
    _HAVE_LOUVAIN = True
except ImportError:
    _HAVE_LOUVAIN = False


# =============================================================================
# AGEE hesaplayıcı: önce kullanıcının v3 paketini dene
# =============================================================================
def _load_user_agee():
    """metricAGEE paketini bulmaya çalışır. Yolları gerekirse düzenleyin."""
    candidate_paths = [
        os.environ.get("METRIC_AGEE_PATH", ""),
        os.path.expanduser("~/metricAGEE"),
        "/mnt/c/Users/user/metricAGEE",      # WSL -> Windows (birincil)
        "C:/Users/user/metricAGEE",          # saf Windows Python
    ]
    for p in candidate_paths:
        if p and os.path.isdir(p):
            sys.path.insert(0, p)
            try:
                from core.agee import AGEECalculator   # type: ignore
                return AGEECalculator, p
            except Exception:
                continue
    return None, None


_UserAGEE, _user_path = _load_user_agee()


# =============================================================================
# Referans (minimal) AGEE — yalnızca pipeline testi için.
# Makale tanımı: AGEE = M_p(S', I', E'), p=0.5, w=(0.40,0.35,0.25), eps=0.01
# =============================================================================
def _communities(G) -> Dict[Any, int]:
    if G.number_of_nodes() == 0:
        return {}
    if _HAVE_LOUVAIN:
        try:
            return community_louvain.best_partition(G)
        except Exception:
            pass
    # fallback: bağlı bileşenleri topluluk say
    part = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            part[n] = i
    return part


def _shannon_shrunk(counts: np.ndarray, K: int) -> float:
    """Shannon entropi + HS09 (James-Stein) shrinkage, normalize /log K."""
    N = counts.sum()
    if N == 0 or K <= 1:
        return 0.0
    p_mle = counts / N
    t_k = 1.0 / K
    denom = (N - 1) * np.sum((p_mle - t_k) ** 2)
    if denom <= 0:
        lam = 1.0
    else:
        lam = (1.0 - np.sum(p_mle ** 2)) / denom
        lam = min(1.0, max(0.0, lam))
    p_sh = lam * t_k + (1 - lam) * p_mle
    p_sh = p_sh[p_sh > 0]
    H = -np.sum(p_sh * np.log(p_sh))
    return float(H / math.log(K))


def _reference_agee(trajectory: List[str], G,
                    p=0.5, weights=(0.40, 0.35, 0.25),
                    eps=0.01, beta=1.0) -> Dict[str, float]:
    """Makale tanımına sadık minimal AGEE (yön: yönsüz)."""
    n = G.number_of_nodes()
    if n == 0 or not trajectory:
        return {"AGEE": 0.0, "S": 0.0, "I": 0.0, "E": 0.0}
    part = _communities(G)
    K = max(1, len(set(part.values())))
    visited = [v for v in trajectory if v in G]
    Vs = set(visited)

    # --- S' : yapısal kapsama ---
    if K > 1:
        comm_counts = np.zeros(K)
        cmap = {c: i for i, c in enumerate(sorted(set(part.values())))}
        for v in Vs:
            comm_counts[cmap[part[v]]] += 1
        Hnorm = _shannon_shrunk(comm_counts, K)
        alpha = 1 - 1.0 / K
        S = alpha * Hnorm + (1 - alpha) * (len(Vs) / n)
    else:
        S = len(Vs) / n

    # --- I' : bilgi kazanım oranı (azalan getiri ağırlıklı) ---
    discovered = set()
    T = len(visited)
    isum = 0.0
    for v in visited:
        nbrs = set(G.neighbors(v))
        deg = max(1, len(nbrs))
        new = len(nbrs - discovered)
        g_t = new / deg
        w_t = (1 - len(discovered) / n) ** beta
        isum += w_t * g_t
        discovered |= nbrs | {v}
    I = isum / max(1, T)

    # --- E' : kapsama-AUC verimliliği (düğüm bazlı) ---
    nu = len(Vs)
    seen = set(); curve = []
    for v in visited:
        seen.add(v)
        curve.append(len(seen) / n)
    auc = sum(curve) / max(1, T)
    # ideal AUC (kapalı form)
    ideal = (sum((t + 1) / n for t in range(nu)) + (T - nu) * (nu / n)) / max(1, T) if nu <= T \
        else nu / n
    E = min(1.0, auc / ideal) if ideal > 0 else 0.0

    # --- power mean aggregation ---
    xs = [max(S, eps), max(I, eps), max(E, eps)]
    M = sum(w * (x ** p) for w, x in zip(weights, xs)) ** (1.0 / p)
    return {"AGEE": float(M), "S": float(S), "I": float(I), "E": float(E)}


def compute_agee(trajectory, G):
    if _UserAGEE is not None:
        try:
            calc = _UserAGEE()  # imzanız farklıysa burayı uyarlayın
            score = calc.compute(trajectory, G)  # <-- v3 API'nize göre düzenleyin
            if isinstance(score, dict):
                return score
            return {"AGEE": float(score), "S": None, "I": None, "E": None}
        except Exception as e:
            print(f"  [uyarı] metricAGEE çağrısı başarısız ({e}); "
                  f"referans implementasyona düşülüyor")
    return _reference_agee(trajectory, G)


# =============================================================================
# RoG yükleme + soru eşleme
# =============================================================================
def load_rog(dataset: str, split: str = "test") -> Dict[str, dict]:
    """rmanluo/RoG-{webqsp,cwq} -> {qid: example}."""
    from datasets import load_dataset
    name = {"webqsp": "rmanluo/RoG-webqsp", "cwq": "rmanluo/RoG-cwq"}[dataset]
    ds = load_dataset(name, split=split)
    out = {}
    for ex in ds:
        qid = ex.get("id") or ex.get("qid") or ex.get("question")
        out[str(qid)] = ex
    return out


# =============================================================================
# Ana
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["webqsp", "cwq"], required=True)
    ap.add_argument("--rog-split", default="test")
    ap.add_argument("--agee-ready", required=True,
                    help="parser çıktısı *_agee_ready.json")
    ap.add_argument("--agent", required=True, help="ToG / PoG / BFS ...")
    ap.add_argument("--out", default="agee_results.csv")
    ap.add_argument("--limit", type=int, default=0, help="ilk N kayıt (pilot)")
    args = ap.parse_args()

    print(f"metricAGEE bulundu mu? -> "
          f"{'EVET ('+_user_path+')' if _UserAGEE else 'HAYIR, referans impl. kullanılacak'}")
    print(f"Louvain mevcut mu? -> {_HAVE_LOUVAIN}")

    print(f"RoG-{args.dataset} ({args.rog_split}) yükleniyor...")
    rog = load_rog(args.dataset, args.rog_split)
    print(f"  {len(rog)} soru subgraph'ı yüklendi")

    recs = json.load(open(args.agee_ready, encoding="utf-8"))
    if args.limit:
        recs = recs[:args.limit]
    print(f"  {len(recs)} trajectory kaydı okundu (ajan={args.agent})")

    rows = []
    for i, r in enumerate(recs):
        qid = str(r.get("qid") or r.get("question"))
        ex = rog.get(qid)
        if ex is None:
            # qid eşleşmezse soru metniyle dene
            ex = rog.get(str(r.get("question")))
        if ex is None:
            continue
        kg = SubgraphKG.from_rog_example(ex)
        G = kg.to_networkx(undirected=True)
        a = compute_agee(r["trajectory"], G)
        rows.append({
            "agent": args.agent, "qid": qid,
            "hits1": r["hits1"], "traj_len": r["traj_len"],
            "n_unique": r["n_unique"],
            "AGEE": a["AGEE"], "S": a.get("S"),
            "I": a.get("I"), "E": a.get("E"),
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        })
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(recs)} işlendi")

    # CSV yaz
    import csv
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n{len(rows)} satır yazıldı -> {args.out}")

        # Özet
        arr = np.array([[x["hits1"], x["AGEE"]] for x in rows], dtype=float)
        print(f"\n=== ÖZET ({args.agent} / {args.dataset}) ===")
        print(f"  Hits@1 ort.: {arr[:,0].mean():.3f}")
        print(f"  AGEE  ort.: {arr[:,1].mean():.3f}")
        # bileşen-Hits@1 nokta-biserial korelasyonu (yeterli veri varsa)
        if len(rows) >= 10:
            from scipy.stats import pointbiserialr
            for comp in ["AGEE", "S", "I", "E"]:
                vals = [x[comp] for x in rows if x[comp] is not None]
                if len(vals) == len(rows) and len(set(x["hits1"] for x in rows)) > 1:
                    rpb, pval = pointbiserialr([x["hits1"] for x in rows], vals)
                    print(f"  {comp:5s} ~ Hits@1: r_pb={rpb:+.3f}, p={pval:.4f}")
    else:
        print("[uyarı] Hiç satır üretilmedi — qid eşleşmesini kontrol edin.")


if __name__ == "__main__":
    main()
