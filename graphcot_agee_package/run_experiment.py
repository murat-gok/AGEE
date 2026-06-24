"""
Crash-safe Graph-CoT x AGEE runner.

For each RoG question: run the Graph-CoT agent, score Hits@1 against the gold
answers, compute AGEE on the recorded trajectory, and append results. Re-runs
skip already-completed question ids (resume after a crash).

Outputs (under OUT_DIR):
  results_GraphCoT_<dataset>_<backbone>.csv   per-question rows
  trajectories_GraphCoT_<dataset>_<backbone>.jsonl
  summary_GraphCoT_<dataset>_<backbone>.json  cell-level summary

Cell summary mirrors the handover-report tables: mean AGEE over valid
(traversal-mode) trajectories, exploration rate, overall Hits@1.
"""
import os
import csv
import json
import random
import numpy as np

import config
from graph_env import build_undirected_graph, norm
import agee_metric
import rog_loader


def hits1(pred, gold_answers):
    p = norm(pred)
    if not p:
        return 0
    for g in gold_answers:
        gg = norm(g)
        if not gg:
            continue
        if p == gg or gg in p or p in gg:
            return 1
    return 0


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    os.makedirs(config.OUT_DIR, exist_ok=True)

    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        from smoke_test import MockLLM, mock_examples
        llm = MockLLM()
        examples = list(mock_examples())
        print("[run] SMOKE MODE: mock LLM + synthetic graph")
    else:
        from ollama_bridge import OllamaBridge
        llm = OllamaBridge()
        ok, tags = llm.health_check()
        if not ok:
            raise SystemExit(
                f"Ollama model {config.OLLAMA_MODEL} not available. Installed: {tags}\n"
                f"Run:  ollama pull {config.OLLAMA_MODEL}")
        examples = list(rog_loader.load_examples(limit=config.N_QUESTIONS))
        print(f"[run] dataset={config.DATASET} backbone={config.ACTIVE_BACKBONE} "
              f"model={config.OLLAMA_MODEL} N={len(examples)}")

    from graphcot_agent import GraphCoTAgent

    done = load_done_ids(config.TRAJ_JSONL)
    if done:
        print(f"[run] resuming -- {len(done)} questions already done")

    new_header = not os.path.exists(config.RESULTS_CSV)
    csv_f = open(config.RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if new_header:
        writer.writerow(["id", "dataset", "backbone", "n_nodes", "n_steps",
                         "n_visited", "valid", "hits1", "S", "I", "E", "AGEE",
                         "K", "lambda_star", "pred", "gold"])
    traj_f = open(config.TRAJ_JSONL, "a", encoding="utf-8")
    tog_f = open(config.TOG_FMT_JSONL, "a", encoding="utf-8")

    for i, ex in enumerate(examples):
        if ex["id"] in done:
            continue
        G = build_undirected_graph(ex["triples"])
        agent = GraphCoTAgent(llm, ex["triples"], config.MAX_STEPS)
        try:
            res = agent.run(ex["question"], ex.get("q_entity"))
        except Exception as e:
            res = {"answer": "", "visited": [], "n_steps": 0,
                   "scratchpad": f"FATAL {e}"}

        m = agee_metric.compute_agee(res["visited"], G)
        valid = int(m["n_visited"] >= config.VALID_MIN_VISITED)
        hit = hits1(res["answer"], ex["answers"])

        writer.writerow([ex["id"], config.DATASET, config.ACTIVE_BACKBONE,
                         G.number_of_nodes(), res["n_steps"], m["n_visited"],
                         valid, hit,
                         round(m["S"], 4) if m["S"] == m["S"] else "",
                         round(m["I"], 4) if m["I"] == m["I"] else "",
                         round(m["E"], 4) if m["E"] == m["E"] else "",
                         round(m["AGEE"], 4) if m["AGEE"] == m["AGEE"] else "",
                         m["K"], m.get("lambda_star"),
                         res["answer"], " | ".join(ex["answers"][:5])])
        csv_f.flush()
        traj_f.write(json.dumps({
            "id": ex["id"], "question": ex["question"],
            "q_entity": ex.get("q_entity", []),
            "visited": res["visited"], "answer": res["answer"],
            "n_hops": res.get("n_hops", 0), "edges": res.get("edges", []),
            "gold": ex["answers"], "valid": valid, "hits1": hit,
            "agee": m, "scratchpad": res["scratchpad"],
        }) + "\n")
        traj_f.flush()

        # ToG-format record for your tog_pog_parser.py (reasoning_chains = edges)
        tog_f.write(json.dumps({
            "id": ex["id"], "question": ex["question"],
            "results": res["answer"], "answer": ex["answers"],
            "reasoning_chains": [list(e) for e in res.get("edges", [])],
        }, ensure_ascii=False) + "\n")
        tog_f.flush()

        if (i + 1) % 10 == 0:
            print(f"[run] {i+1}/{len(examples)} done")

    csv_f.close()
    traj_f.close()
    tog_f.close()
    summarise()
    print(f"\n[run] ToG-format file ready -> {config.TOG_FMT_JSONL}")
    print( "[run] NEXT (your pipeline, identical convention):")
    print(f"  python tog_pog_parser.py {config.TOG_FMT_JSONL}")
    print(f"  python 10_run_agee_on_trajectories.py --dataset {config.DATASET} "
          f"--agent GraphCoT --agee-ready "
          f"{config.TOG_FMT_JSONL.rsplit('.',1)[0]}_agee_ready.json "
          f"--out results_GraphCoT_{config.DATASET}_{config.ACTIVE_BACKBONE}.csv")


def summarise():
    rows = []
    with open(config.RESULTS_CSV, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    n = len(rows)
    valid = [r for r in rows if r["valid"] == "1"]

    def fmean(rs, k):
        vals = [float(r[k]) for r in rs if r[k] not in ("", None)]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "dataset": config.DATASET,
        "backbone": config.ACTIVE_BACKBONE,
        "n_questions": n,
        "n_valid_traversal": len(valid),
        "exploration_rate": round(len(valid) / n, 4) if n else None,
        "hits1_overall": round(sum(int(r["hits1"]) for r in rows) / n, 4) if n else None,
        "AGEE_mean_valid": fmean(valid, "AGEE"),
        "S_mean_valid": fmean(valid, "S"),
        "I_mean_valid": fmean(valid, "I"),
        "E_mean_valid": fmean(valid, "E"),
    }
    with open(config.SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== CELL SUMMARY ===")
    print(json.dumps(summary, indent=2))


def regen_tog_format():
    """Regenerate the ToG-format JSONL from the trajectory JSONL (crash recovery).
    Each line carries reasoning_chains (edges) for tog_pog_parser.py."""
    if not os.path.exists(config.TRAJ_JSONL):
        return
    with open(config.TRAJ_JSONL, "r", encoding="utf-8") as f, \
         open(config.TOG_FMT_JSONL, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.write(json.dumps({
                "id": r["id"], "question": r["question"],
                "results": r.get("answer", ""), "answer": r.get("gold", []),
                "reasoning_chains": [list(e) for e in r.get("edges", [])],
            }, ensure_ascii=False) + "\n")
    print(f"[regen] rewrote {config.TOG_FMT_JSONL} from trajectory JSONL")


if __name__ == "__main__":
    import sys
    if "--regen-tog" in sys.argv:
        regen_tog_format()
    else:
        main()
