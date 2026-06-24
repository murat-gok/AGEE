# Graph-CoT × AGEE — real end-to-end runner (WebQSP / CWQ)

This package runs the **real Graph-CoT** agent (Jin et al., ACL 2024) on the
**RoG pre-processed WebQSP/CWQ subgraphs** with your local Ollama backbones,
records each trajectory, and scores it with **AGEE** (computed verbatim from
the manuscript, Section 5). It is built to drop into the same experimental
frame as your existing ToG/PoG runs so the numbers are directly comparable.

> Runs on your laptop (RTX 3050 6 GB + Ollama, WSL Ubuntu). No GPU fine-tuning,
> no Freebase/Virtuoso, no HF gating beyond the RoG datasets you already use.

---

## 0. Comparability (already validated)

Graph-CoT trajectories are scored by YOUR `tog_pog_parser.py` (trajectory
construction) and YOUR `metricAGEE` (Leiden) via `10_run_agee_on_trajectories.py`
-- the exact same path as ToG/PoG. No separate calibration needed: the pilot
already reproduced AGEE within ~0.005 of the package's reference value.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Ollama models (the same tags as your ToG/PoG runs):
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

## 2. Wiring check (no LLM needed)

```bash
python smoke_test.py          # must print "SMOKE TEST PASSED"
```

## 3. Run the four cells

```bash
DATASET=webqsp ACTIVE_BACKBONE=qwen  python run_experiment.py
DATASET=webqsp ACTIVE_BACKBONE=llama python run_experiment.py
DATASET=cwq    ACTIVE_BACKBONE=qwen  python run_experiment.py
DATASET=cwq    ACTIVE_BACKBONE=llama python run_experiment.py
```

Each cell writes `results/GraphCoT_<ds>_<bk>.jsonl` in **ToG format**
(`question / results / answer / reasoning_chains`). Crash-safe + resumable.
Pilot first with `N_QUESTIONS=10`.

## 4. Score with YOUR pipeline (identical convention)

The trajectory, traj_len, "explored" flag and Hits@1 are all built by YOUR
`tog_pog_parser.py`, and AGEE by YOUR `metricAGEE` (Leiden) -- so Graph-CoT is
scored exactly like ToG/PoG:

```bash
# 1) your parser builds the trajectory (same convention as ToG/PoG)
python tog_pog_parser.py results/GraphCoT_webqsp_qwen.jsonl
# -> results/GraphCoT_webqsp_qwen_agee_ready.json

# 2) your AGEE pipeline (finds metricAGEE -> Leiden)
python 10_run_agee_on_trajectories.py --dataset webqsp --agent GraphCoT \
  --agee-ready results/GraphCoT_webqsp_qwen_agee_ready.json \
  --out results_GraphCoT_webqsp_qwen.csv
```

Send me the four `results_GraphCoT_*.csv` (script-10 output) and I integrate
them into Section 8 + the double-dissociation figure.

If the LLM run crashed mid-way, regenerate the ToG-format file from the
trajectory log without re-running the LLM:  `python run_experiment.py --regen-tog`

## 5. Faithfulness & disclosures (these go into the manuscript Methods, verbatim)

To pre-empt the detail-oriented reviewer, we will state plainly what is the
original Graph-CoT and what is an adaptation:

1. **Method**: the Think→Act→Observe loop with `RetrieveNode`, `NeighbourCheck`,
   `NodeFeature`, `NodeDegree`, `Finish` is Graph-CoT's, unchanged.
2. **Domain adaptation**: Graph-CoT targets GRBench; we apply it to Freebase
   WebQSP/CWQ subgraphs by exposing entities as nodes and relations as edges
   (inverse relations as `~r`). Disclosed as an out-of-native-domain adaptation.
2b. **Entity linking is given** (not part of the agent): WebQSP/CWQ provide the
   topic entity, so the agent starts traversal from the benchmark-provided
   `q_entity` — exactly as ToG/PoG do. This keeps the comparison fair.
2c. **Trajectory definition**: tau records the entities the agent *lands on*
   (the retrieved seed, each node it checks the neighbours/features of, and the
   committed answer entity). Neighbours merely *returned* by a NeighbourCheck
   are "discovered" (folded into AGEE's discovered set via N(v_t)) but are not
   counted as visited — matching how a ToG/PoG path records traversed nodes.
3. **Retriever**: GRBench uses a dense retriever; we use a string/entity-name
   matcher over the (small) retrieved subgraph. Disclosed.
4. **Backbones**: Qwen-2.5-7B and Llama-3.1-8B via Ollama, `num_ctx=2048`,
   temperature 0 (Graph-CoT default; set `GRAPHCOT_TEMPERATURE=0.3` to match the
   LLM-ReAct setting). Disclosed.
5. **AGEE**: computed on the undirected collapse of each subgraph with the same
   default-Leiden settings as ToG/PoG (Section 0 validates this).

## 6. Expected runtime

Iterative, multi-call per question (like ToG/PoG). Rough order: 200 questions ×
~5–8 LLM calls each. On a 7B/8B Q4 model via Ollama on RTX 3050 expect a few
hours per cell; run cells overnight/sequentially. Pilot with `N_QUESTIONS=10`
first to confirm answers and AGEE look sane.

## 7. Note on KG-Agent

KG-Agent is intentionally **not** in this package: its method requires
fine-tuning a dedicated LLaMA-7B on a synthesized 10K code-instruction corpus
(Jiang et al., 2024), which is outside our compute envelope. It remains a
clearly-labeled synthetic characterization in the manuscript, with that reason
stated. Three real strong agents (ToG, PoG, Graph-CoT) on two datasets × two
backbones is the delivered evidence for Reviewer 3.4.
