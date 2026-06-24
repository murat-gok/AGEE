"""
Loader for RoG pre-processed subgraphs (rmanluo/RoG-webqsp, rmanluo/RoG-cwq).

Each RoG example carries a *retrieved subgraph* as a list of [head, relation,
tail] triples plus the topic entity(ies) and gold answer(s). This is the same
source your ToG/PoG runs used, so no Virtuoso/Freebase deployment is needed.

normalise() returns a uniform dict:
    {
      "id":        str,
      "question":  str,
      "q_entity":  [str, ...],   # topic / seed entities
      "answers":   [str, ...],   # gold answer surface forms (a_entity + answer)
      "triples":   [(h, r, t), ...],
    }
"""
import json
import config


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x]
    return [str(x)]


def normalise(ex):
    triples = []
    for tr in ex.get("graph", []) or []:
        if isinstance(tr, (list, tuple)) and len(tr) == 3:
            h, r, t = tr
            triples.append((str(h), str(r), str(t)))
    answers = _as_list(ex.get("a_entity")) + _as_list(ex.get("answer"))
    # de-dup while preserving order
    seen, ans = set(), []
    for a in answers:
        k = a.strip().lower()
        if k and k not in seen:
            seen.add(k)
            ans.append(a)
    return {
        "id":       str(ex.get("id", ex.get("qid", ""))),
        "question": str(ex.get("question", "")).strip(),
        "q_entity": _as_list(ex.get("q_entity")),
        "answers":  ans,
        "triples":  triples,
    }


def load_examples(limit=None):
    """Yield normalised examples. Prefers a local file if ROG_LOCAL_PATH is set."""
    if config.ROG_LOCAL_PATH:
        path = config.ROG_LOCAL_PATH
        print(f"[rog_loader] reading local file: {path}")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            else:
                rows = json.load(f)
        for i, ex in enumerate(rows):
            if limit and i >= limit:
                break
            yield normalise(ex)
        return

    # HuggingFace path -- identical source to your ToG/PoG runs.
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' package is required for HF loading. "
            "Install it (pip install datasets) or set ROG_LOCAL_PATH to a local "
            f"RoG json/jsonl file. Original error: {e}"
        )
    print(f"[rog_loader] loading {config.HF_DATASET_NAME} split={config.HF_SPLIT}")
    ds = load_dataset(config.HF_DATASET_NAME, split=config.HF_SPLIT)
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        yield normalise(ex)


if __name__ == "__main__":
    n = 0
    for ex in load_examples(limit=3):
        n += 1
        print(f"\n--- example {n} ---")
        print("Q:", ex["question"])
        print("seed:", ex["q_entity"], "| answers:", ex["answers"][:5])
        print("triples:", len(ex["triples"]), "(first 3)", ex["triples"][:3])
