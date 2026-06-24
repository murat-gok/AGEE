"""
Turn 2 — Real LLM Agent on Real KG-QA (W2 Fix)

Downloads MetaQA-2hop dataset, builds knowledge subgraphs,
runs a ReAct-style agent via Ollama (Qwen2.5:7b),
logs entity-visit trajectories, computes AGEE, correlates with Hits@1.

Usage:
    python run_kgqa_experiment.py

Requirements:
    - Ollama running with qwen2.5:7b model
    - Internet connection (first run downloads MetaQA ~15MB)
    - metricAGEE_v3 folder in same parent directory
"""

import os
import sys
import json
import csv
import time
import random
import requests
import zipfile
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import networkx as nx

# Resolve the project root so `core/` and `agents/` can be imported
# regardless of which working directory the user runs this script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Project root must come first on sys.path so `import core.agee` resolves
# to <project>/core/agee.py rather than any same-named installed package.
for candidate in (PROJECT_ROOT, SCRIPT_DIR):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from core.agee import AGEECalculator
from core.metrics import compute_usr


# ================================================================
# CONFIG
# ================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
MAX_HOPS = 10          # max traversal steps per question
N_QUESTIONS = 200      # number of questions to evaluate
SEED = 42
DATA_DIR = os.path.join(SCRIPT_DIR, "data_metaqa")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ================================================================
# STEP 1: Download and parse MetaQA
# ================================================================

def download_metaqa():
    """Download MetaQA dataset if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    kb_path = os.path.join(DATA_DIR, "kb.txt")
    q_path = os.path.join(DATA_DIR, "2hop_test.txt")

    if os.path.exists(kb_path) and os.path.exists(q_path):
        print("  MetaQA data already present.")
        return kb_path, q_path

    print("  Downloading MetaQA dataset...")

    # Download KB
    kb_url = "https://raw.githubusercontent.com/yuyuz/MetaQA/master/kb.txt"
    print(f"    Fetching kb.txt ...")
    resp = requests.get(kb_url, timeout=60)
    if resp.status_code != 200:
        # Fallback: try alternative source
        kb_url = "https://raw.githubusercontent.com/shijx12/TransferNet/master/data/MetaQA/kb.txt"
        resp = requests.get(kb_url, timeout=60)

    if resp.status_code == 200:
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"    Saved kb.txt ({len(resp.text) // 1024} KB)")
    else:
        raise RuntimeError(f"Failed to download kb.txt (status {resp.status_code}). "
                           f"Please download manually from GitHub and place in {DATA_DIR}")

    # Download 2-hop test questions
    q_url = "https://raw.githubusercontent.com/yuyuz/MetaQA/master/2-hop/vanilla/qa_test.txt"
    print(f"    Fetching 2hop test questions...")
    resp = requests.get(q_url, timeout=60)
    if resp.status_code != 200:
        q_url = "https://raw.githubusercontent.com/shijx12/TransferNet/master/data/MetaQA/2-hop/vanilla/qa_test.txt"
        resp = requests.get(q_url, timeout=60)

    if resp.status_code == 200:
        with open(q_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"    Saved 2hop_test.txt ({len(resp.text) // 1024} KB)")
    else:
        raise RuntimeError(f"Failed to download questions (status {resp.status_code}). "
                           f"Please download manually.")

    return kb_path, q_path


def load_knowledge_graph(kb_path: str) -> Tuple[nx.Graph, Dict]:
    """
    Load MetaQA KB into a NetworkX graph.
    Each line: head|relation|tail
    Returns: (graph, edge_data) where edge_data maps (h,t) -> [relations]
    """
    G = nx.Graph()
    edge_relations = defaultdict(list)
    triples = []

    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            G.add_edge(h, t)
            edge_relations[(h, t)].append(r)
            edge_relations[(t, h)].append(r + "_inv")
            triples.append((h, r, t))

    print(f"  KG loaded: {len(G.nodes())} entities, {len(G.edges())} edges, "
          f"{len(triples)} triples")
    return G, edge_relations


def load_questions(q_path: str, n: int = 200, seed: int = 42) -> List[Dict]:
    """
    Load MetaQA 2-hop questions.
    Format: question[tab]answer1|answer2|...
    Topic entity is in [brackets] in the question.
    """
    questions = []
    with open(q_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            q_text = parts[0].strip()
            answers = [a.strip() for a in parts[1].split("|")]

            # Extract topic entity from brackets
            topic = None
            if "[" in q_text and "]" in q_text:
                start = q_text.index("[") + 1
                end = q_text.index("]")
                topic = q_text[start:end]

            if topic:
                questions.append({
                    "question": q_text,
                    "topic_entity": topic,
                    "answers": answers,
                })

    random.seed(seed)
    random.shuffle(questions)
    selected = questions[:n]
    print(f"  Loaded {len(questions)} questions, selected {len(selected)}")
    return selected


def extract_subgraph(G: nx.Graph, edge_relations: Dict,
                     topic: str, n_hops: int = 3) -> Tuple[nx.Graph, Dict]:
    """Extract n-hop subgraph around topic entity."""
    if topic not in G:
        return nx.Graph(), {}

    # BFS to get n-hop neighborhood
    nodes = {topic}
    frontier = {topic}
    for _ in range(n_hops):
        next_frontier = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in nodes:
                    next_frontier.add(nb)
                    nodes.add(nb)
        frontier = next_frontier

    subG = G.subgraph(nodes).copy()

    # Extract relevant edge relations
    sub_relations = {}
    for u, v in subG.edges():
        key1 = (u, v)
        key2 = (v, u)
        rels = edge_relations.get(key1, []) + edge_relations.get(key2, [])
        sub_relations[(u, v)] = rels

    return subG, sub_relations


# ================================================================
# STEP 2: LLM Agent (ReAct-style)
# ================================================================

def call_ollama(prompt: str, model: str = MODEL, max_retries: int = 3) -> str:
    """Call Ollama API."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200}
                },
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                print(f"    Ollama error {resp.status_code}, retry {attempt+1}")
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"    Connection error, retry {attempt+1}. Is Ollama running?")
            time.sleep(3)
        except Exception as e:
            print(f"    Error: {e}, retry {attempt+1}")
            time.sleep(2)
    return ""


def run_react_agent(question: str, topic: str, subG: nx.Graph,
                    sub_relations: Dict, max_hops: int = MAX_HOPS) -> Dict:
    """
    ReAct-style KG traversal agent.

    At each step:
    1. Agent sees: question, current entity, available edges (neighbor + relation)
    2. Agent decides: which neighbor to visit next, or ANSWER
    3. We log every entity visited

    Returns trajectory and final answer.
    """
    if topic not in subG:
        return {"trajectory": [], "answer": "", "steps": 0, "success": False}

    trajectory = [topic]
    current = topic
    visited_log = []

    for step in range(max_hops):
        # Get available edges from current node
        neighbors = list(subG.neighbors(current))
        if not neighbors:
            break

        # Build edge descriptions
        edge_options = []
        for i, nb in enumerate(neighbors[:15]):  # limit to 15 to fit context
            key = (current, nb)
            key2 = (nb, current)
            rels = sub_relations.get(key, sub_relations.get(key2, ["connected_to"]))
            rel_str = rels[0] if rels else "connected_to"
            edge_options.append(f"  {i+1}. {nb} (via: {rel_str})")

        edges_text = "\n".join(edge_options)

        prompt = f"""You are a knowledge graph reasoning agent. Answer the question by traversing the knowledge graph.

Question: {question}
Current entity: {current}
Step: {step+1}/{max_hops}
Entities visited so far: {', '.join(trajectory[-5:])}

Available edges from "{current}":
{edges_text}

Choose the BEST next entity to visit to answer the question, or say ANSWER if you can answer now.

Reply with ONLY one of:
- A number (1-{len(neighbors[:15])}) to visit that entity
- ANSWER: [your answer entity] if you know the answer

Your choice:"""

        response = call_ollama(prompt)

        # Parse response
        response_clean = response.strip().split("\n")[0].strip()

        if "ANSWER" in response_clean.upper():
            # Extract answer
            answer = response_clean.split(":")[-1].strip().strip("[]\"'")
            return {
                "trajectory": trajectory,
                "answer": answer,
                "steps": step + 1,
                "response_log": visited_log,
            }

        # Try to parse number
        try:
            # Extract first number from response
            import re
            nums = re.findall(r'\d+', response_clean)
            if nums:
                choice = int(nums[0]) - 1
                if 0 <= choice < len(neighbors[:15]):
                    next_entity = neighbors[:15][choice]
                    trajectory.append(next_entity)
                    current = next_entity
                    visited_log.append({
                        "step": step + 1,
                        "from": trajectory[-2],
                        "to": next_entity,
                        "response": response_clean[:50]
                    })
                    continue
        except (ValueError, IndexError):
            pass

        # If parsing failed, pick a random unvisited neighbor
        unvisited = [nb for nb in neighbors if nb not in set(trajectory)]
        if unvisited:
            next_entity = random.choice(unvisited)
        else:
            next_entity = random.choice(neighbors)
        trajectory.append(next_entity)
        current = next_entity

    # Ran out of steps — last entity is the answer guess
    return {
        "trajectory": trajectory,
        "answer": current,
        "steps": max_hops,
        "response_log": visited_log,
    }


# ================================================================
# STEP 3: Baseline agents (same ones from v3)
# ================================================================

def random_walk_agent(topic, subG, max_hops=MAX_HOPS):
    """Random walk baseline."""
    if topic not in subG:
        return {"trajectory": [], "answer": "", "steps": 0}
    current = topic
    trajectory = [current]
    for _ in range(max_hops - 1):
        neighbors = list(subG.neighbors(current))
        if not neighbors:
            break
        current = random.choice(neighbors)
        trajectory.append(current)
    return {"trajectory": trajectory, "answer": current, "steps": len(trajectory)}


def greedy_novelty_agent(topic, subG, max_hops=MAX_HOPS):
    """Greedy: pick neighbor with most unvisited neighbors."""
    if topic not in subG:
        return {"trajectory": [], "answer": "", "steps": 0}
    current = topic
    trajectory = [current]
    discovered = {current}
    discovered.update(subG.neighbors(current))
    for _ in range(max_hops - 1):
        neighbors = list(subG.neighbors(current))
        if not neighbors:
            break
        scores = [(len(set(subG.neighbors(nb)) - discovered), nb) for nb in neighbors]
        mx = max(s[0] for s in scores)
        best = [s[1] for s in scores if s[0] == mx]
        current = random.choice(best)
        trajectory.append(current)
        discovered.add(current)
        discovered.update(subG.neighbors(current))
    return {"trajectory": trajectory, "answer": current, "steps": len(trajectory)}


def bfs_agent(topic, subG, max_hops=MAX_HOPS):
    """BFS baseline."""
    if topic not in subG:
        return {"trajectory": [], "answer": "", "steps": 0}
    trajectory = [topic]
    visited = {topic}
    queue = list(subG.neighbors(topic))
    random.shuffle(queue)
    step = 1
    while step < max_hops and queue:
        node = queue.pop(0)
        if node in visited:
            continue
        trajectory.append(node)
        visited.add(node)
        step += 1
        nn = [n for n in subG.neighbors(node) if n not in visited]
        random.shuffle(nn)
        queue.extend(nn)
    return {"trajectory": trajectory, "answer": trajectory[-1], "steps": len(trajectory)}


# ================================================================
# STEP 4: Compute AGEE + Hits@1
# ================================================================

def compute_hits(answer: str, gold_answers: List[str]) -> int:
    """Check if predicted answer matches any gold answer (case-insensitive)."""
    answer_lower = answer.lower().strip()
    for gold in gold_answers:
        if gold.lower().strip() in answer_lower or answer_lower in gold.lower().strip():
            return 1
    return 0


def run_experiment():
    """Run the full Turn 2 experiment with crash-safe checkpointing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkpoint_path = os.path.join(RESULTS_DIR, "checkpoint.jsonl")
    csv_path = os.path.join(RESULTS_DIR, "kgqa_trajectories.csv")

    print("=" * 70)
    print("  TURN 2: Real LLM Agent on MetaQA-2hop KG-QA")
    print("=" * 70)

    # Step 1: Load data
    print("\n  [1/4] Loading MetaQA data...")
    kb_path, q_path = download_metaqa()
    G, edge_relations = load_knowledge_graph(kb_path)
    questions = load_questions(q_path, n=N_QUESTIONS, seed=SEED)

    # ── CHECKPOINT: Load previous progress if exists ──
    # We track (question_id, agent) pairs, not just question_ids.
    # This means an interrupted/missing LLM run on a question can be resumed
    # without re-running the algorithmic agents.
    completed_pairs = set()  # set of (question_id, agent_name)
    all_results = []

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                # Back-compat: older rows have no skip_reason field
                if "skip_reason" not in row:
                    row["skip_reason"] = ""
                all_results.append(row)
                completed_pairs.add((row["question_id"], row["agent"]))
        n_unique_q = len(set(p[0] for p in completed_pairs))
        print(f"\n  RESUMING from checkpoint: {len(completed_pairs)} (question, agent) pairs "
              f"already logged, across {n_unique_q} unique questions.")
    else:
        print(f"\n  Starting fresh (no checkpoint found)")

    # Step 2: Run agents
    agents = {
        "llm_react": None,  # special handling
        "random_walk": random_walk_agent,
        "greedy": greedy_novelty_agent,
        "bfs": bfs_agent,
    }

    agent_summaries = defaultdict(lambda: {"hits": [], "agee": [], "n": 0})

    # Rebuild summaries from loaded checkpoint data (skip NaN-AGEE rows)
    for row in all_results:
        agent_summaries[row["agent"]]["hits"].append(row["hit"])
        agee_val = row["agee"]
        if isinstance(agee_val, (int, float)) and not (agee_val != agee_val):  # not NaN
            agent_summaries[row["agent"]]["agee"].append(agee_val)
        agent_summaries[row["agent"]]["n"] += 1

    missing_pairs = sum(
        1 for qi in range(min(N_QUESTIONS, len(questions)))
        for an in agents
        if (qi, an) not in completed_pairs
    )
    print(f"\n  [2/4] Running on {missing_pairs} missing (question, agent) pairs...")

    for qi, q in enumerate(questions):
        # Determine which agents still need to run for this question
        agents_to_run = [an for an in agents if (qi, an) not in completed_pairs]
        if not agents_to_run:
            continue

        topic = q["topic_entity"]
        answers = q["answers"]

        # Extract 3-hop subgraph
        subG, sub_rels = extract_subgraph(G, edge_relations, topic, n_hops=3)

        if len(subG.nodes()) < 2:
            continue

        # Map string nodes to integers for AGEE
        node_list = list(subG.nodes())
        node_to_id = {n: i for i, n in enumerate(node_list)}
        id_to_node = {i: n for n, i in node_to_id.items()}

        G_int = nx.Graph()
        G_int.add_nodes_from(range(len(node_list)))
        for u, v in subG.edges():
            G_int.add_edge(node_to_id[u], node_to_id[v])

        calc = AGEECalculator(G_int, graph_name=f"q{qi}")

        question_rows = []

        for agent_name in agents_to_run:
            if agent_name == "llm_react":
                result = run_react_agent(
                    q["question"], topic, subG, sub_rels, max_hops=MAX_HOPS
                )
            else:
                result = agents[agent_name](topic, subG, max_hops=MAX_HOPS)

            traj = result["trajectory"]
            answer = result["answer"]

            # NEW (Sprint 1): every attempted (question, agent) pair produces a row.
            # If the trajectory is unusable for AGEE, we still log the row with
            # NaN metrics and skip_reason set, so the audit trail is complete.
            skip_reason = ""
            traj_int = []
            if len(traj) < 2:
                skip_reason = "trajectory_lt_2"
            else:
                traj_int = [node_to_id[n] for n in traj if n in node_to_id]
                if len(traj_int) < 2:
                    skip_reason = "traj_int_lt_2"

            if skip_reason:
                hit = compute_hits(answer, answers) if answer else 0
                row = {
                    "question_id": qi,
                    "agent": agent_name,
                    "question": q["question"][:80],
                    "topic": topic,
                    "answer_pred": answer,
                    "answer_gold": "|".join(answers[:3]),
                    "hit": hit,
                    "agee": float("nan"),
                    "coverage": float("nan"),
                    "info_rate": float("nan"),
                    "efficiency": float("nan"),
                    "usr": float("nan"),
                    "traj_length": len(traj),
                    "n_unique": 0,
                    "n_nodes_graph": len(subG.nodes()),
                    "n_edges_graph": len(subG.edges()),
                    "n_communities": calc.n_communities,
                    "shrinkage_lambda": float("nan"),
                    "skip_reason": skip_reason,
                    "trajectory": "|".join(map(str, traj)) if traj else "",
                }
                question_rows.append(row)
                all_results.append(row)
                continue

            # Compute AGEE
            agee_result = calc.compute(traj_int, agent_name)

            # Compute Hits@1
            hit = compute_hits(answer, answers)

            row = {
                "question_id": qi,
                "agent": agent_name,
                "question": q["question"][:80],
                "topic": topic,
                "answer_pred": answer,
                "answer_gold": "|".join(answers[:3]),
                "hit": hit,
                "agee": agee_result.agee,
                "coverage": agee_result.coverage,
                "info_rate": agee_result.info_rate,
                "efficiency": agee_result.efficiency,
                "usr": agee_result.usr,
                "traj_length": len(traj),
                "n_unique": agee_result.n_unique_nodes,
                "n_nodes_graph": len(subG.nodes()),
                "n_edges_graph": len(subG.edges()),
                "n_communities": calc.n_communities,
                "shrinkage_lambda": agee_result.shrinkage_lambda,
                "skip_reason": "",
                # NEW: save full trajectory as JSON string for downstream
                # robustness analyses (e.g., Louvain vs Leiden re-analysis).
                # Stored as string-pipe-separated to keep CSV readable.
                "trajectory": "|".join(map(str, traj)),
            }
            question_rows.append(row)
            all_results.append(row)
            agent_summaries[agent_name]["hits"].append(hit)
            agent_summaries[agent_name]["agee"].append(agee_result.agee)
            agent_summaries[agent_name]["n"] += 1

        # ── CHECKPOINT: Save after EACH question ──
        if question_rows:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                for row in question_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in question_rows:
                completed_pairs.add((row["question_id"], row["agent"]))

        done = len(set(p[0] for p in completed_pairs))
        if done % 10 == 0:
            print(f"    [{done}/{N_QUESTIONS}] questions touched "
                  f"(LLM Hits@1: {np.mean(agent_summaries['llm_react']['hits']):.1%} "
                  f"on {agent_summaries['llm_react']['n']} Qs)")

    # Step 3: Save final CSV
    print(f"\n  [3/4] Saving final results...")
    fieldnames = list(all_results[0].keys()) if all_results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"    Saved {len(all_results)} rows to {csv_path}")

    # Step 4: Analysis
    print(f"\n  [4/4] Analysis...")

    # NEW: report skip statistics first
    skip_counts = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        reason = r.get("skip_reason", "") or "ok"
        skip_counts[r["agent"]][reason] += 1
    print(f"\n  Row counts and skip reasons per agent:")
    print(f"  {'Agent':<16} {'OK':>6} {'traj<2':>8} {'int<2':>8} {'Total':>8}")
    print("  " + "-" * 50)
    for agent_name in ["llm_react", "bfs", "greedy", "random_walk"]:
        c = skip_counts[agent_name]
        ok = c.get("ok", 0)
        t2 = c.get("trajectory_lt_2", 0)
        i2 = c.get("traj_int_lt_2", 0)
        total = ok + t2 + i2
        print(f"  {agent_name:<16} {ok:>6} {t2:>8} {i2:>8} {total:>8}")

    print(f"\n  {'Agent':<16} {'Hits@1 (all)':>14} {'Hits@1 (valid)':>16} {'AGEE':>8} {'N_valid':>8}")
    print("  " + "-" * 64)

    from scipy import stats as sp_stats

    # Use only valid (non-skipped) rows for AGEE-based analysis;
    # report Hits@1 both on ALL rows (including skipped) and on valid-only rows.
    valid_rows = [r for r in all_results if not r.get("skip_reason", "")]

    for agent_name in ["llm_react", "bfs", "greedy", "random_walk"]:
        all_agent = [r for r in all_results if r["agent"] == agent_name]
        valid_agent = [r for r in valid_rows if r["agent"] == agent_name]
        if not all_agent:
            continue
        h_all = np.mean([r["hit"] for r in all_agent]) if all_agent else float("nan")
        h_valid = np.mean([r["hit"] for r in valid_agent]) if valid_agent else float("nan")
        a_valid = np.mean([r["agee"] for r in valid_agent]) if valid_agent else float("nan")
        print(f"  {agent_name:<16} {h_all:>14.3f} {h_valid:>16.3f} {a_valid:>8.4f} {len(valid_agent):>8}")

    # Correlation: AGEE vs Hits@1 — restricted to valid rows
    print(f"\n  Correlation: AGEE vs Hits@1 (valid rows only, N={len(valid_rows)})")
    print(f"  {'Metric':<16} {'r_pb':>8} {'p-value':>10} {'sig':>4} {'N':>6}")
    print("  " + "-" * 46)

    all_hits = [r["hit"] for r in valid_rows]
    metrics_to_test = [
        ("AGEE", [r["agee"] for r in valid_rows]),
        ("Coverage", [r["coverage"] for r in valid_rows]),
        ("InfoRate", [r["info_rate"] for r in valid_rows]),
        ("Efficiency", [r["efficiency"] for r in valid_rows]),
        ("USR", [r["usr"] for r in valid_rows]),
    ]

    for mname, mvals in metrics_to_test:
        if len(set(all_hits)) > 1 and np.std(mvals) > 0:
            r_pb, p_val = sp_stats.pointbiserialr(all_hits, mvals)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {mname:<16} {r_pb:>8.3f} {p_val:>10.6f} {sig:>4} {len(all_hits):>6}")

    # Per-agent AGEE vs Hits@1 (valid rows)
    print(f"\n  Per-agent AGEE vs Hits@1 (valid rows):")
    for agent_name in ["llm_react", "bfs", "greedy", "random_walk"]:
        agent_rows = [r for r in valid_rows if r["agent"] == agent_name]
        hits = [r["hit"] for r in agent_rows]
        agees = [r["agee"] for r in agent_rows]
        if len(set(hits)) > 1 and len(agees) > 1:
            r_pb, p_val = sp_stats.pointbiserialr(hits, agees)
            print(f"    {agent_name:<14} r_pb={r_pb:>7.3f}  p={p_val:.4f}  N={len(hits)}")
        else:
            print(f"    {agent_name:<14} (insufficient variance, N={len(hits)})")

    # Clean up checkpoint after successful completion
    print(f"\n  Results saved to: {csv_path}")
    print(f"  Checkpoint at: {checkpoint_path}")
    print(f"  (Delete checkpoint to re-run from scratch)")
    print("=" * 70)
    print("  Turn 2 complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
