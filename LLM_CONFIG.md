# LLM configuration for the AGEE KG-QA experiment

This document discloses every aspect of the LLM runtime, addressing
reproducibility concern W14 ("LLM details incomplete") in the revision plan.

## Model

- **Family**: Qwen 2.5
- **Size**: 7B parameters
- **Quantisation**: 4-bit
- **Quantisation scheme**: Q4_0 GGML (Ollama default for `qwen2.5:7b`)
- **Serving runtime**: [Ollama](https://ollama.com) v0.1.30 or newer
- **Endpoint**: `http://localhost:11434/api/generate`

## Sampling parameters

| Parameter | Value | Source |
|---|---|---|
| `temperature` | 0.3 | `run_kgqa_experiment.py:218` |
| `num_predict` | 200 | `run_kgqa_experiment.py:218` |
| `top_p` | 1.0 (Ollama default) | implicit |
| `repeat_penalty` | 1.1 (Ollama default) | implicit |

## ReAct loop parameters

| Parameter | Value | Source |
|---|---|---|
| `MAX_HOPS` | 10 | `run_kgqa_experiment.py:51` |
| Max neighbours shown per step | 15 | `run_kgqa_experiment.py:263` |
| Visited-history window in prompt | last 5 entities | `run_kgqa_experiment.py:277` |
| HTTP timeout per call | 120 s | `run_kgqa_experiment.py:220` |
| Retries on failure | 3 | `run_kgqa_experiment.py:208` |

## Fallback behaviour

If the model's response cannot be parsed as either a numeric choice or an
explicit `ANSWER:` declaration, the runner falls back to a uniformly random
unvisited neighbour (`run_kgqa_experiment.py:326-331`). If all neighbours
are already visited, a uniformly random neighbour is chosen. This fallback
is reported in the manuscript.

## Hardware

- **GPU**: NVIDIA RTX 3050 (4 GB) — Ollama auto-detects and uses CUDA when
  available; CPU fallback otherwise.
- **CPU**: Intel Core i7 (manuscript hardware)
- **RAM**: 16 GB minimum recommended for Qwen-2.5-7B-Q4_0

## Reproduction recipe

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the exact model
ollama pull qwen2.5:7b

# 3. Verify it's the right digest
ollama show qwen2.5:7b | head

# 4. Start the experiment
cd kgqa_experiment
python run_kgqa_experiment.py
```

## Wall-clock budget

On the reference hardware (RTX 3050, Q4_0), expect:
- ~15–30 seconds per LLM step
- ~2–5 minutes per 10-hop question
- ~4–8 hours for the full 200-question run
