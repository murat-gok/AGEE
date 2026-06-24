# Turn 2: Real LLM Agent on MetaQA-2hop KG-QA

## What this does
1. Downloads MetaQA knowledge graph (400K+ triples) and 2-hop test questions
2. For each question, extracts the 3-hop subgraph around the topic entity
3. Runs 4 agents on each subgraph:
   - **llm_react**: Qwen2.5:7b via Ollama (ReAct-style traversal)
   - **bfs**: Breadth-first search baseline
   - **greedy**: Max-novelty neighbor selection
   - **random_walk**: Random neighbor selection
4. Logs every entity visited in each trajectory
5. Computes AGEE v3 on each trajectory
6. Correlates AGEE with Hits@1 (real KG-QA accuracy)

## How to run

```powershell
# 1. Make sure Ollama is running with the model loaded
ollama run qwen2.5:7b "test"
# (then type /bye to exit)

# 2. Place this folder next to metricAGEE_v3:
#    C:\Users\user\
#      metricAGEE_v3\    (the v3 codebase)
#      kgqa_experiment\  (this folder)

# 3. Run the experiment
cd C:\Users\user\kgqa_experiment
python run_kgqa_experiment.py
```

## Expected runtime
- ~30-60 minutes for 200 questions (depends on GPU speed)
- LLM agent is the bottleneck (~5-10 sec per question)
- Baseline agents finish instantly

## Output
- `results/kgqa_trajectories.csv` — all trajectories + AGEE scores + Hits@1
- Console output with summary statistics

## After running
Paste the FULL console output to Claude. It contains the numbers
needed for the revised manuscript.
