# AGEE: Agent Graph Exploration Efficiency

**AGEE** is a deterministic, process-level metric for evaluating how AI agents explore knowledge graphs.

Unlike outcome-only metrics such as Hits@1, which evaluate only the final answer, AGEE evaluates the full exploration trajectory: how broadly the agent covers relevant graph structure, how much new information it gains, and how efficiently it explores.

**Author:** Murat Gök<br>
**Affiliation:** Department of Computer Engineering, Yalova University, Türkiye<br>
**ORCID:** [0000-0003-2261-9288](https://orcid.org/0000-0003-2261-9288)

> **Publication status:** This repository accompanies the article titled
> **"AGEE: A Multi-Dimensional Process-Level Metric for Evaluating Agent Exploration over Knowledge Graphs"**, accepted for publication in *Knowledge-Based Systems*.
> The manuscript text itself is not included in this repository; only the source code, scripts, reproducibility files, and related materials are provided.

## Related article

This repository provides the source code and reproducibility materials for the following article:

**Murat Gök.**
**"AGEE: A Multi-Dimensional Process-Level Metric for Evaluating Agent Exploration over Knowledge Graphs."**
Accepted for publication in *Knowledge-Based Systems*.

A formal bibliographic citation, including volume, pages, year, and DOI, will be added after the article is published online.

## What is AGEE?

AGEE evaluates an agent's exploration trajectory

$$
\tau = (v_0, v_1, \ldots, v_T)
$$

on a knowledge graph $G$ using three complementary components:

* **Structural coverage $S'$** — measures how broadly and evenly the trajectory covers graph communities.
* **Information-gain rate $I'$** — measures diminishing-returns-weighted novelty gained during exploration.
* **Exploration efficiency $E'$** — measures how efficiently coverage is accumulated over time.

These components are combined using a non-compensatory weighted power mean:

$$
\mathrm{AGEE}(G, \tau) =
\left(
w_S S'^{p} + w_I I'^{p} + w_E E'^{p}
\right)^{1/p}
$$

with default parameters:

$$
p = 0.5,\quad
w = (w_S, w_I, w_E) = (0.40, 0.35, 0.25).
$$

Because AGEE is computed from the exploration trajectory itself, it does not require final-answer annotations and introduces no evaluator variance.

## Key idea

AGEE is designed to reveal differences in agent behavior that outcome-only metrics may miss.

For example, two agents can obtain the same final-answer accuracy while following very different exploration strategies. AGEE makes these differences measurable by evaluating the process of graph exploration rather than only the terminal answer.

## Quickstart

```bash
git clone https://github.com/murat-gok/AGEE.git
cd AGEE

python -m venv .venv
source .venv/bin/activate     # Windows PowerShell: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Run a minimal worked example on the Karate Club graph:

```bash
python - <<'PY'
from core.agee import AGEECalculator, DEFAULT_CONFIG
import networkx as nx

G = nx.karate_club_graph()
calc = AGEECalculator(G, config=DEFAULT_CONFIG, graph_name="karate")

trajectory = [0, 1, 2, 3, 5, 6, 16]
result = calc.compute(trajectory, "demo")

print(
    f"AGEE = {result.agee:.4f} "
    f"(S'={result.coverage:.4f}, "
    f"I'={result.info_rate:.4f}, "
    f"E'={result.efficiency:.4f})"
)
PY
```

Expected output:

```text
AGEE = 0.5046 (S'=0.6584, I'=0.1903, E'=0.8696)
```

## Reproducing reported results

### 1. KG-QA experiment

```bash
python kgqa_experiment/run_kgqa_experiment.py
```

This script runs the KG-QA experiment and writes the output trajectories to:

```text
kgqa_experiment/results/kgqa_trajectories.csv
```

A pre-computed CSV is included so that results can be inspected without re-running the full experiment.

### 2. Statistical analysis

```bash
python analysis/run_table3_bootstrap.py
python analysis/run_sensitivity.py
python analysis/run_remaining_ablations.py
```

These scripts reproduce the bootstrap confidence intervals, sensitivity analyses, and ablation studies.

### 3. SOTA baselines and cross-metric comparison

```bash
python analysis/run_sota_synthetic.py
python analysis/run_cross_metric_analysis.py
python analysis/run_multi_llm_synthetic.py
```

These scripts reproduce the synthetic SOTA baseline comparisons, cross-metric analyses, and multi-LLM scaling experiments.

### 4. Real multi-LLM measurements

Before running these experiments, install Ollama and pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct
```

Then run:

```bash
python kgqa_experiment/run_kgqa_real_multillm.py --model llama3.1:8b --suffix llama3_1_8b
python kgqa_experiment/run_kgqa_real_multillm.py --model mistral:7b-instruct --suffix mistral_7b_v0_3
python analysis/merge_real_llm.py
python analysis/regenerate_fig5.py
```

Pre-computed CSV files are included for reproducibility and inspection.

### 5. Scalability benchmarks

```bash
python analysis/run_scalability_benchmark.py
python analysis/run_scalability_n1e6.py
python analysis/merge_scalability.py
```

The large-scale benchmark with $n = 10^6$ nodes may require at least 32 GB RAM.

## Repository structure

```text
AGEE/
├── core/                         AGEE metric implementation
│   ├── agee.py                   Top-level AGEECalculator
│   ├── metrics.py                S', I', and E' computation
│   ├── graph_utils.py            Community detection and graph utilities
│   └── composite.py              Weighted power mean combiner
├── agents/                       Baseline agents
├── data/                         Synthetic graph generators and data utilities
├── experiments/                  Sensitivity and stability experiments
├── prompts/                      Prompt templates for KG-QA agents
├── kgqa_experiment/              KG-QA experimental pipeline
│   ├── data_metaqa/              MetaQA knowledge base and test questions
│   ├── run_kgqa_experiment.py    Main KG-QA experiment script
│   ├── run_kgqa_real_multillm.py Multi-LLM real Ollama runs
│   └── results/                  Pre-computed trajectories and outputs
├── analysis/                     Statistical analysis scripts
│   ├── figures/                  Figure generation scripts
│   └── results/                  Pre-computed analysis outputs
├── main.py                       CLI entry point and smoke tests
├── requirements.txt              Python dependencies
├── seeds.yaml                    Seed configuration for reproducibility
├── LLM_CONFIG.md                 Ollama setup notes
├── Dockerfile                    Container build recipe
├── README.md
├── LICENSE
└── .zenodo.json
```

## Data

The experiments use publicly available knowledge-graph QA resources and generated trajectories.

Large trajectory and subgraph files are not committed to the repository. Users should download the required datasets from their original sources and regenerate trajectories locally when needed.

For experiments involving RoG subgraphs, the relevant public datasets are:

* `rmanluo/RoG-webqsp`: https://huggingface.co/datasets/rmanluo/RoG-webqsp
* `rmanluo/RoG-cwq`: https://huggingface.co/datasets/rmanluo/RoG-cwq

## Requirements

* Python 3.10, 3.11, or 3.12
* Approximately 8 GB disk space for the default setup
* Approximately 15 GB additional disk space if local LLM models are pulled through Ollama
* 16 GB RAM for standard experiments
* 32 GB RAM recommended for million-node scalability experiments
* Optional: Ollama for local LLM-ReAct experiments

## Reproducibility and determinism

All randomized components use the seed configuration in `seeds.yaml`.

Default seed:

```text
42
```

The main deterministic settings are:

* Community detection: fixed random state
* Random-walk trajectory generation: fixed NumPy seed
* Bootstrap resampling: fixed random generator seed

Aggregate results are expected to reproduce across platforms within small numerical variation.

LLM-generated trajectories may show limited nondeterminism even under fixed settings because local inference backends can introduce implementation-level variation. Therefore, pre-computed CSV outputs are provided where appropriate.

## Code availability

The source code is publicly available at:

```text
https://github.com/murat-gok/AGEE
```

An archived version will be made available through Zenodo after the first public release.

After Zenodo assigns a DOI, add it here in the following form:

```text
https://doi.org/10.5281/zenodo.xxxxxxx
```

## Citation

A formal citation will be added after the accompanying article is published.

Until then, please cite this repository as:

```text
Gök, M. (2026). AGEE: A Multi-Dimensional Process-Level Metric for Evaluating Agent Exploration over Knowledge Graphs. Source code repository. GitHub: https://github.com/murat-gok/AGEE
```

After Zenodo archiving, please cite the archived software release using the Zenodo DOI.

## License

This project is released under the MIT License.

See the [LICENSE](./LICENSE) file for details.

The accompanying manuscript is not included in this repository.

## Contact

**Murat Gök**<br>
Department of Computer Engineering<br>
Yalova University, Türkiye<br>

GitHub issues: https://github.com/murat-gok/AGEE/issues<br>
ORCID: [0000-0003-2261-9288](https://orcid.org/0000-0003-2261-9288)
