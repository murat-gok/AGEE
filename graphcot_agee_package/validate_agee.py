"""
Calibration / validation.

Recompute AGEE on your EXISTING ToG or PoG trajectories and check that this
module reproduces the published per-cell AGEE (e.g. ToG WebQSP Qwen ~ 0.588,
PoG WebQSP Qwen ~ 0.640). If it matches, the Graph-CoT AGEE values produced by
this package are directly comparable. If it does NOT match, the most likely
cause is a Leiden setting mismatch (objective / seed / resolution) or a
discovered-set convention -- adjust config.py accordingly and re-run.

Usage:
    python validate_agee.py --traj path/to/your_ToG_webqsp_qwen.jsonl \
        [--visited-field reasoning_chain] [--triples-field graph] \
        [--rog webqsp]   # if triples are not stored in the traj file

The traj file is JSONL; each line is one question. You tell the script which
field holds the ordered visited-entity list and (optionally) which holds the
subgraph triples. If triples are absent, pass --rog to rebuild the subgraph
from the matching RoG example by id.
"""
import argparse
import json
import numpy as np

import config
from graph_env import build_undirected_graph
import agee_metric


def get_field(row, dotted):
    cur = row
    for key in dotted.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="your ToG/PoG trajectory JSONL")
    ap.add_argument("--visited-field", default="visited",
                    help="field holding the ordered visited-entity list")
    ap.add_argument("--triples-field", default="triples",
                    help="field holding the subgraph triples (h,r,t)")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--rog", default="", choices=["", "webqsp", "cwq"],
                    help="rebuild subgraph from RoG by id if triples missing")
    ap.add_argument("--valid-min", type=int, default=config.VALID_MIN_VISITED)
    args = ap.parse_args()

    rog_index = {}
    if args.rog:
        import os
        os.environ["DATASET"] = args.rog
        import importlib, rog_loader
        importlib.reload(config)
        importlib.reload(rog_loader)
        for ex in rog_loader.load_examples():
            rog_index[ex["id"]] = ex["triples"]
        print(f"[validate] indexed {len(rog_index)} RoG subgraphs ({args.rog})")

    agees, comps = [], {"S": [], "I": [], "E": []}
    n_rows = n_valid = 0
    with open(args.traj, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            visited = get_field(row, args.visited_field)
            if not visited:
                continue
            triples = get_field(row, args.triples_field)
            if not triples and rog_index:
                triples = rog_index.get(str(get_field(row, args.id_field)))
            if not triples:
                continue
            triples = [tuple(t) for t in triples]
            G = build_undirected_graph(triples)
            m = agee_metric.compute_agee(list(visited), G)
            if m["n_visited"] >= args.valid_min and m["AGEE"] == m["AGEE"]:
                n_valid += 1
                agees.append(m["AGEE"])
                for k in comps:
                    comps[k].append(m[k])

    print(f"\n[validate] rows={n_rows}  valid={n_valid}")
    if agees:
        print(f"[validate] AGEE mean (valid) = {np.mean(agees):.4f} "
              f"(sd {np.std(agees):.4f})")
        for k in comps:
            print(f"[validate] {k} mean (valid) = {np.mean(comps[k]):.4f}")
        print("\nCompare AGEE mean to your published value for this cell. "
              "Within ~0.01 => calibrated; larger gap => check Leiden settings "
              "in config.py (LEIDEN_OBJECTIVE / LEIDEN_SEED).")
    else:
        print("[validate] no valid trajectories parsed -- check --visited-field "
              "/ --triples-field names against your JSONL schema.")


if __name__ == "__main__":
    main()
