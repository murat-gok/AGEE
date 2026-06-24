"""
Central configuration for the Graph-CoT x AGEE experiment.

All knobs that affect comparability with the existing ToG/PoG runs are
collected here and documented. The two settings that MUST match your
existing ToG/PoG AGEE pipeline are flagged with ``# [MUST-MATCH]``.
"""
import os

# ---------------------------------------------------------------------------
# Backbone selection (mirrors your ACTIVE_BACKBONE convention)
# ---------------------------------------------------------------------------
# Set with:  export ACTIVE_BACKBONE=qwen   (or)   export ACTIVE_BACKBONE=llama
ACTIVE_BACKBONE = os.environ.get("ACTIVE_BACKBONE", "qwen").lower()

# Ollama model tags. These are the exact tags you used for ToG/PoG.
OLLAMA_MODEL = {
    "qwen":  "qwen2.5:7b",
    "llama": "llama3.1:8b",
}[ACTIVE_BACKBONE]

# ---------------------------------------------------------------------------
# Ollama serving parameters
# ---------------------------------------------------------------------------
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434")
NUM_CTX         = 2048          # matches your ToG/PoG context window
TEMPERATURE     = float(os.environ.get("GRAPHCOT_TEMPERATURE", "0.0"))
#   Graph-CoT is deterministic by design (temp 0). If you prefer to match the
#   LLM-ReAct setting in the paper (temp 0.3), set GRAPHCOT_TEMPERATURE=0.3.
MAX_NEW_TOKENS  = 256
REQUEST_TIMEOUT = 180           # seconds per LLM call

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
# Set with:  export DATASET=webqsp   (or)   export DATASET=cwq
DATASET = os.environ.get("DATASET", "webqsp").lower()

# Where to read RoG pre-processed subgraphs from. Two options:
#   (1) HuggingFace name (default) -- works if you have internet / HF cache,
#       exactly the source your ToG/PoG runs used.
#   (2) A local .json/.jsonl path (set ROG_LOCAL_PATH) -- offline fallback.
HF_DATASET_NAME = {
    "webqsp": "rmanluo/RoG-webqsp",
    "cwq":    "rmanluo/RoG-cwq",
}[DATASET]
HF_SPLIT       = os.environ.get("ROG_SPLIT", "test")
ROG_LOCAL_PATH = os.environ.get("ROG_LOCAL_PATH", "")   # optional offline file

# ---------------------------------------------------------------------------
# Protocol (mirrors the paper's protocol exactly)
# ---------------------------------------------------------------------------
N_QUESTIONS = int(os.environ.get("N_QUESTIONS", "200"))   # 200 per cell, as in the paper
SEED        = 42                                           # torch/numpy/random = 42
MAX_STEPS   = 10                                           # max reasoning iterations (== 10 hops in paper)

# A trajectory is "valid" / traversal-mode if it visits >= 2 distinct nodes
# (i.e. performed at least one edge traversal). Paper Sec. 6.4 / Sec. 7.3.
VALID_MIN_VISITED = 2          # [MUST-MATCH] paper's "trajectory length >= 2" rule

# ---------------------------------------------------------------------------
# AGEE parameters (verbatim from the manuscript, Sec. 5)
# ---------------------------------------------------------------------------
AGEE_WEIGHTS = (0.40, 0.35, 0.25)   # (w_S, w_I, w_E)
AGEE_P       = 0.5                   # power-mean exponent
AGEE_EPS     = 0.01                  # epsilon floor
AGEE_BETA    = 1.0                   # diminishing-returns exponent in I'

# Leiden community detection. "default Leiden" in the paper == leidenalg with
# the modularity objective and a fixed seed. [MUST-MATCH] your ToG/PoG run.
LEIDEN_OBJECTIVE = "modularity"      # leidenalg.ModularityVertexPartition
LEIDEN_SEED      = 42

# Discovered-set bookkeeping choices that the manuscript's Algorithm 1 fixes.
# Defaults follow the printed pseudocode; flip only to match your code if it
# differs (then re-run the validator).
START_NODE_IN_DISCOVERED = False     # Algorithm 1 sets D_0 = {} (empty)
W_USES_PREUPDATE_DISCOVERED = True   # w_t uses |D_{t-1}| (pre-update), per Alg. 1

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR        = os.environ.get("OUT_DIR", "results")
RESULTS_CSV    = os.path.join(OUT_DIR, f"results_GraphCoT_{DATASET}_{ACTIVE_BACKBONE}.csv")
TRAJ_JSONL     = os.path.join(OUT_DIR, f"trajectories_GraphCoT_{DATASET}_{ACTIVE_BACKBONE}.jsonl")
SUMMARY_JSON   = os.path.join(OUT_DIR, f"summary_GraphCoT_{DATASET}_{ACTIVE_BACKBONE}.json")
# ToG-format JSONL (question/results/answer/reasoning_chains) -> feed to your
# tog_pog_parser.py so the trajectory is built with the IDENTICAL convention.
TOG_FMT_JSONL  = os.path.join(OUT_DIR, f"GraphCoT_{DATASET}_{ACTIVE_BACKBONE}.jsonl")
