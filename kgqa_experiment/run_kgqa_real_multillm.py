"""
Sprint 2 Task 2 supplement — Real Llama-3.1-8B / Mistral-7B measurement.

This script replaces the synthetic estimates for Llama-3.1-8B and
Mistral-7B-v0.3 in §6.5 Multi-LLM scaling with real measurements.

Reuses the same MetaQA-2hop questions, prompt template, subgraph
extraction, and ReAct loop as run_kgqa_experiment.py (Sprint 1), but
parameterised by model name.

CRITICAL: This script ONLY runs the LLM-ReAct agent. We do not re-run
the algorithmic baselines (BFS/Greedy/RW) because (i) their behaviour
does not depend on the LLM and (ii) we already have those results in
Sprint 1.

Output: kgqa_experiment/results/kgqa_real_llm_<suffix>.csv

CLI:
    python kgqa_experiment/run_kgqa_real_multillm.py \\
        --model llama3.1:8b \\
        --suffix llama3_1_8b

    python kgqa_experiment/run_kgqa_real_multillm.py \\
        --model mistral:7b-instruct-v0.3 \\
        --suffix mistral_7b_v0_3

Prerequisites:
  1. Ollama installed and running locally (http://localhost:11434)
  2. The model already pulled:
       ollama pull llama3.1:8b
       ollama pull mistral:7b-instruct-v0.3
  3. The Sprint 1 kgqa_experiment infrastructure intact

Wall-clock estimate (Xeon E5-1660 v3, 16 threads, CPU-only inference):
  - Llama-3.1-8B Q4_K_M:  ~30-50 sec/q × 200 q = ~2.5-4 hours
  - Mistral-7B-v0.3 Q4_K_M: ~25-40 sec/q × 200 q = ~2-3 hours

Checkpointing: row-by-row CSV append + resume on restart.
"""
import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Reuse all the infrastructure from the main Sprint 1 script
from kgqa_experiment.run_kgqa_experiment import (
    OLLAMA_URL,
    MAX_HOPS,
    N_QUESTIONS,
    SEED,
    download_metaqa,
    load_knowledge_graph,
    load_questions,
    extract_subgraph,
    call_ollama,
    run_react_agent,
)
from core.agee import AGEECalculator, DEFAULT_CONFIG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Ollama model name, e.g. 'llama3.1:8b'")
    p.add_argument("--suffix", required=True,
                   help="Filename suffix, e.g. 'llama3_1_8b'")
    p.add_argument("--n_questions", type=int, default=N_QUESTIONS,
                   help=f"Number of questions (default {N_QUESTIONS})")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature (default 0.3, matches Sprint 1)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 72)
    print(f"  Real-LLM KGQA experiment — model={args.model}")
    print("=" * 72)
    print(f"  Questions: {args.n_questions}, temperature: {args.temperature}")

    results_dir = os.path.join(PROJECT_ROOT, "kgqa_experiment", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f"kgqa_real_llm_{args.suffix}.csv")
    ckpt = os.path.join(results_dir, f"checkpoint_real_llm_{args.suffix}.jsonl")

    # Probe Ollama health and presence of model
    print("\n  Probing Ollama...")
    try:
        rsp = requests.post(OLLAMA_URL,
                            json={"model": args.model,
                                  "prompt": "ping",
                                  "stream": False,
                                  "options": {"num_predict": 4}},
                            timeout=30)
        if rsp.status_code != 200:
            print(f"  [FAIL] Ollama returned {rsp.status_code}: {rsp.text[:200]}")
            print(f"  Check that you ran:  ollama pull {args.model}")
            sys.exit(1)
        print(f"  [OK] Model '{args.model}' responds.")
    except Exception as e:
        print(f"  [FAIL] Cannot reach Ollama at {OLLAMA_URL}: {e}")
        print(f"  Make sure 'ollama serve' is running.")
        sys.exit(1)

    # Resume from checkpoint if present
    completed_qids = set()
    all_rows = []
    if os.path.exists(ckpt):
        with open(ckpt, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                all_rows.append(row)
                completed_qids.add(row["question_id"])
        print(f"\n  RESUMING from {len(completed_qids)} completed questions")
    else:
        print(f"\n  Fresh start")

    # Load data
    print("\n  Loading MetaQA-2hop...")
    kb_path, q_path = download_metaqa()
    G_full, edge_relations = load_knowledge_graph(kb_path)
    questions = load_questions(q_path, n=args.n_questions, seed=SEED)
    print(f"    KG: |V|={len(G_full.nodes()):,}, |E|={len(G_full.edges()):,}")
    print(f"    Questions: {len(questions)}")

    # Override call_ollama in the imported module so it uses our chosen model.
    # The default `model=MODEL` argument is locked at import time, so we must
    # patch the function itself, not just the MODEL constant.
    import kgqa_experiment.run_kgqa_experiment as mod
    _orig_call_ollama = mod.call_ollama

    def _patched_call_ollama(prompt, model=args.model, max_retries=3):
        return _orig_call_ollama(prompt, model=model, max_retries=max_retries)

    mod.call_ollama = _patched_call_ollama
    print(f"  Patched call_ollama to use model '{args.model}'")

    # Run LLM-ReAct only
    print(f"\n  Running LLM-ReAct with {args.model}...")
    t_start_all = time.time()
    n_done = 0

    for q_idx, q in enumerate(questions):
        if q_idx in completed_qids:
            continue

        topic = q["topic_entity"]
        ans_list = q.get("answers", [])
        if topic not in G_full or not ans_list:
            continue

        subgraph, sub_edges = extract_subgraph(G_full, edge_relations,
                                                topic, n_hops=2)
        if subgraph.number_of_nodes() < 5:
            continue

        t_q = time.time()
        try:
            result = run_react_agent(q["question"], topic, subgraph,
                                       sub_edges, max_hops=MAX_HOPS)
            traj = result.get("trajectory", [topic])
            pred = result.get("answer", "")
            steps = result.get("steps", 0)
        except Exception as e:
            print(f"    q{q_idx}: LLM error: {e}")
            traj = [topic]
            pred = ""
            steps = 0

        # Hit metric: exact match against any gold answer
        hit = int(any(a in pred for a in ans_list) or
                  any(a in traj for a in ans_list))

        # Compute AGEE
        if len(traj) >= 2:
            calc = AGEECalculator(subgraph, config=DEFAULT_CONFIG,
                                   graph_name=f"q{q_idx}")
            r = calc.compute(traj, "llm_react")
            skip = ""
            agee, S, I, E = r.agee, r.coverage, r.info_rate, r.efficiency
            n_unique = len(set(traj))
        else:
            agee = S = I = E = np.nan
            n_unique = 1
            skip = "trajectory_too_short"

        t_elapsed = time.time() - t_q

        row = {
            "question_id": q_idx,
            "agent": "llm_react",
            "model": args.model,
            "question": q["question"],
            "topic": topic,
            "answer_pred": pred,
            "answer_gold": "|".join(ans_list),
            "hit": hit,
            "agee": agee, "coverage": S, "info_rate": I, "efficiency": E,
            "traj_length": len(traj),
            "n_unique": n_unique,
            "n_nodes_graph": subgraph.number_of_nodes(),
            "n_edges_graph": subgraph.number_of_edges(),
            "n_steps": steps,
            "time_sec": t_elapsed,
            "skip_reason": skip,
        }
        all_rows.append(row)
        with open(ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        n_done += 1

        if n_done % 10 == 1 or t_elapsed > 60:
            t_elapsed_total = time.time() - t_start_all
            n_remain = len(questions) - q_idx - 1
            t_per_q = t_elapsed_total / max(n_done, 1)
            eta_min = (n_remain * t_per_q) / 60
            print(f"    q{q_idx:>3d}: hit={hit} agee={agee:.2f} "
                  f"|τ|={len(traj)} t={t_elapsed:.1f}s "
                  f"(rate={t_per_q:.1f}s/q, eta={eta_min:.0f}min)")

    # Save final CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    t_total = time.time() - t_start_all
    print(f"\n  DONE. {len(df)} rows saved to {out_csv}")
    print(f"  Total wall-clock: {t_total/60:.1f} minutes")

    # Per-agent summary (matching Sprint 1 reporting style)
    valid = df[df["skip_reason"] == ""]
    if len(valid):
        print(f"\n  Hits@1: {valid['hit'].mean():.3f} (N={len(valid)})")
        print(f"  Mean AGEE: {valid['agee'].mean():.3f}")
        print(f"  Mean |τ|: {valid['traj_length'].mean():.1f}")
        memory_mode = df[df["traj_length"] <= 1]
        print(f"  Memory mode rate: {len(memory_mode)/len(df):.3f} "
              f"({len(memory_mode)}/{len(df)})")


if __name__ == "__main__":
    main()
