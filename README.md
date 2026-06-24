<<<<<<< HEAD
# AGEE — Agent Graph Exploration Efficiency

A multi-dimensional, **process-level** metric for evaluating how AI agents *explore*
knowledge graphs — independently of whether they get the final answer right.
Where outcome metrics such as Hits@1 score only the terminal answer, AGEE scores
the **trajectory**: how well the agent covers the relevant structure, how much new
information it gains, and how efficiently it does so.

> **Status:** Manuscript under review at *Knowledge-Based Systems* (Elsevier),
> major revision (ref. KNOSYS-D-26-10689). This repository accompanies that
> submission.

## What AGEE measures

AGEE aggregates three complementary, bounded dimensions of an exploration trajectory `τ`
over a knowledge graph `G`:

- **Structural coverage `S'`** — community-visit entropy with James–Stein shrinkage over
  Leiden communities (how broadly and evenly the relevant structure is covered).
- **Information-gain rate `I'`** — diminishing-returns-weighted novelty per step.
- **Exploration efficiency `E'`** — area under the realized coverage curve against a
  closed-form ideal-coverage curve.

These are combined with a **non-compensatory weighted power mean** (`p = 0.5`), so a
collapse on any single dimension cannot be masked by the others. AGEE requires **no
answer annotation** and introduces **no evaluator variance** (it is deterministic).

## Headline finding

Across a `ToG / PoG / Graph-CoT × WebQSP / CWQ × Qwen-2.5-7B / Llama-3.1-8B` matrix
(N = 200 per cell; 2,400 trajectories), AGEE reveals an **outcome–process double
dissociation**: as task difficulty rises (2-hop → 4-hop), Hits@1 falls for *every*
agent, while the *exploration response* dissociates by architecture (one agent
abandons exploration, another sustains it, another intensifies it). Outcome-only
metrics cannot see this; AGEE can.

## Repository structure

> Adjust to your actual layout before the first push.

```
.
├── src/                  # AGEE implementation (S', I', E', power-mean, Leiden, shrinkage)
├── scripts/              # trajectory parsing + AGEE-over-trajectories runners
├── results/              # per-cell AGEE CSV/JSONL outputs
├── figures/              # generated figures (vector PDF + 300 dpi PNG)
├── README.md
├── LICENSE
└── .gitignore
```

## Data

Experiments use the RoG subgraphs of WebQSP and CWQ:
- `rmanluo/RoG-webqsp` — https://huggingface.co/datasets/rmanluo/RoG-webqsp
- `rmanluo/RoG-cwq` — https://huggingface.co/datasets/rmanluo/RoG-cwq

Large trajectory and subgraph files are **not** committed (see `.gitignore`); download
the datasets from the sources above and regenerate trajectories locally.

## Reproducing

Local agents are served with [Ollama](https://ollama.com) (`qwen2.5:7b`, `llama3.1:8b`)
on a single consumer GPU (RTX 3050, 6 GB). See `scripts/` for the trajectory → AGEE
pipeline. Runs use crash-safe checkpointing throughout.

## Citation

A formal citation will be added on acceptance. Until then, please cite as a manuscript
under review at *Knowledge-Based Systems* (KNOSYS-D-26-10689), M. Gök, Yalova University.

## License

Code is released under the MIT License (see `LICENSE`). The manuscript is under review
and is **not** included in this repository.

## Contact

Murat Gök — Department of Computer Engineering, Yalova University — murat.gok@yalova.edu.tr
=======
# AGEE: Agent Graph Exploration Efficiency

**Author:** Murat Gök
**Affiliation:** Department of Computer Engineering, Yalova University, Türkiye
**ORCID:** [0000-0003-2261-9288](https://orcid.org/0000-0003-2261-9288)

This repository is the reference implementation of **AGEE** — a 3-component
composite metric for evaluating LLM-agent exploration on knowledge graphs.
The method, formulation, theoretical analysis, and experimental design in
this repository are the original work of Murat Gök.

> **Note on release timing.** This repository is paired with a manuscript
> currently under peer review. The manuscript text itself is *not* included
> here; only the code, scripts, and data needed to reproduce reported
> numbers are provided.

## What is AGEE?

AGEE evaluates an agent's exploration trajectory $\tau = (v_0, v_1, \ldots, v_T)$
on a graph $G$ via three sub-components combined under a weighted power mean:

- **Coverage $S'$** — Shannon entropy of visited communities (Leiden
  partition, with Jensen–Shannon shrinkage)
- **Information rate $I'$** — diminishing-returns information gain
- **Efficiency $E'$** — area under the coverage–time curve

The composite is

$$
\mathrm{AGEE}(G, \tau) = \left( w_S\, S'^{\,p} + w_I\, I'^{\,p} + w_E\, E'^{\,p} \right)^{1/p}
$$

with default $p = 0.5$ and weights $w = (w_S, w_I, w_E) = (0.40, 0.35, 0.25)$.

## Quickstart

```bash
git clone https://github.com/murat-gok/AGEE.git
cd AGEE
python -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Minimal worked example (Karate club)
python -c "
from core.agee import AGEECalculator, DEFAULT_CONFIG
import networkx as nx
G = nx.karate_club_graph()
calc = AGEECalculator(G, config=DEFAULT_CONFIG, graph_name='karate')
traj = [0, 1, 2, 3, 5, 6, 16]
r = calc.compute(traj, 'demo')
print(f'AGEE = {r.agee:.4f} (S\\'={r.coverage:.4f}, I\\'={r.info_rate:.4f}, E\\'={r.efficiency:.4f})')
"
```

Expected output:
```
AGEE = 0.5046 (S'=0.6584, I'=0.1903, E'=0.8696)
```

## Reproducing reported results

### 1. Headline KG-QA experiment

```bash
# Requires Ollama with qwen2.5:7b pulled
python kgqa_experiment/run_kgqa_experiment.py
```

Runs 4 agents (BFS, Greedy, Random-Walk, LLM-ReAct via Qwen-2.5-7B) on
200 MetaQA-2hop questions. Outputs `kgqa_experiment/results/kgqa_trajectories.csv`.
A pre-computed CSV is included so you can inspect results without re-running.

### 2. Statistical rigor (bootstrap CIs, sensitivity, ablations)

```bash
python analysis/run_table3_bootstrap.py          # bootstrap CIs on agent metrics
python analysis/run_sensitivity.py               # Dirichlet stability + group-Sobol
python analysis/run_remaining_ablations.py       # p-sweep, β-sweep, weight variation
```

### 3. SOTA baselines and cross-metric comparison

```bash
python analysis/run_sota_synthetic.py            # ToG / PoG / KG-Agent / Graph-CoT (synthetic)
python analysis/run_cross_metric_analysis.py     # Spearman + discriminative power
python analysis/run_multi_llm_synthetic.py       # 5-LLM scaling base table
```

### 4. Real multi-LLM measurements

```bash
# Pre-requisite: ollama pull llama3.1:8b mistral:7b-instruct
python kgqa_experiment/run_kgqa_real_multillm.py --model llama3.1:8b --suffix llama3_1_8b
python kgqa_experiment/run_kgqa_real_multillm.py --model mistral:7b-instruct --suffix mistral_7b_v0_3
python analysis/merge_real_llm.py                # merge into final multi-LLM table
python analysis/regenerate_fig5.py               # regenerate figure
```

Pre-computed CSVs for Llama-3.1-8B and Mistral-7B-v0.3 are included.

### 5. Scalability benchmarks

```bash
python analysis/run_scalability_benchmark.py     # n in [10^3, 10^5], 4 topologies
python analysis/run_scalability_n1e6.py          # n=10^6 (BA, ER, WS); 32 GB recommended
python analysis/merge_scalability.py             # merge and regenerate figure
```

## Repository structure

```
AGEE/
├── core/                       AGEE metric implementation
│   ├── agee.py                 Top-level AGEECalculator
│   ├── metrics.py              S', I', E' computation
│   ├── graph_utils.py          Leiden / Louvain partition
│   └── composite.py            Weighted power mean combiner
├── agents/                     Baseline agents (BFS, Greedy, MCTS, RandomWalk)
├── data/                       Synthetic graph generators (SBM, BA, ER, WS, …)
├── experiments/                Dirichlet stability + p-grid sensitivity
├── prompts/                    ReAct prompt template for KG-QA agents
├── kgqa_experiment/            MetaQA-2hop KG-QA pipeline
│   ├── data_metaqa/            MetaQA KB and 2-hop test questions
│   ├── run_kgqa_experiment.py  Main experiment script
│   ├── run_kgqa_real_multillm.py  Multi-LLM real Ollama runs
│   └── results/                Pre-computed agent trajectories
├── analysis/                   Statistical analysis scripts
│   ├── figures/                Plotting code for paper figures
│   └── results/                Pre-computed analysis CSVs
├── main.py                     CLI entry point (small smoke tests)
├── requirements.txt            Pinned dependencies
├── seeds.yaml                  Seed configuration for determinism
├── LLM_CONFIG.md               Ollama setup notes
└── Dockerfile                  Container build recipe
```

## Requirements

- Python 3.11 or 3.12 (3.10 also works; 3.13+ has wheel issues for some
  packages at the time of writing)
- ~8 GB disk for full setup; ~15 GB extra if pulling LLM models
- 16 GB RAM for default experiments; 32 GB recommended for `run_scalability_n1e6.py`
- Optional: Ollama (for LLM-ReAct experiments)

## Determinism notes

All randomised components use `seeds.yaml` (default seed = 42):

- Leiden partition: `random_state=42`
- Trajectory generation (Random-Walk agent): `numpy.random.seed(42)`
- Bootstrap resampling: `numpy.random.default_rng(42)`

Identical seeds reproduce identical aggregate numbers across platforms;
individual trajectory token sequences for LLM agents are non-deterministic
even at temperature 0.3 (Ollama backend nondeterminism), which is why
pre-computed CSVs are provided.

Cross-platform reproducibility has been verified on Linux x86_64
(Ubuntu 22.04) and Windows 11 x86_64 (Python 3.12.10). Aggregate
numbers reproduce to within 1% across platforms; Leiden partition is
byte-identical given the same `leidenalg` version.

## Citation

A formal BibTeX entry will be added once the accompanying manuscript is
accepted and published. Until then, please cite as:

```
Gök, M. (2026). AGEE: Agent Graph Exploration Efficiency.
GitHub repository: https://github.com/murat-gok/AGEE
```

## License

[See LICENSE file](./LICENSE). The author retains the attribution rights
granted by the license; downstream users should preserve the copyright
notice in derivative works.

## Contact

- **GitHub issues:** https://github.com/murat-gok/AGEE/issues
- **Author:** Murat Gök, Department of Computer Engineering, Yalova University
- **ORCID:** [0000-0003-2261-9288](https://orcid.org/0000-0003-2261-9288)
>>>>>>> ee75ed4bd00f2a7e1791cc7bf4c8e28871e4e378
