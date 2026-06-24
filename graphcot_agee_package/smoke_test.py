"""
End-to-end smoke test WITHOUT Ollama.

A scripted MockLLM drives the agent through a tiny synthetic subgraph to verify
the whole pipeline (agent loop -> trajectory -> AGEE -> CSV/JSON) is wired
correctly. Run before the real experiment:

    SMOKE=1 python run_experiment.py
or directly:
    python smoke_test.py
"""


class MockLLM:
    """Returns one combined 'Thought ... Action: Verb[...]' per call."""
    PLAN = [
        "NeighbourCheck[A, r1]",
        "NeighbourCheck[B, r2]",
        "NeighbourCheck[C, r3]",
        "Finish[ANSWER]",
    ]

    def __init__(self):
        self.aptr = 0

    def generate(self, prompt, stop=("\n",), temperature=None, max_tokens=None):
        action = self.PLAN[min(self.aptr, len(self.PLAN) - 1)]
        self.aptr += 1
        # deliberately wrap in prose to exercise robust extraction
        return f"I reason about the next move.\nAction: I will {action} now."

    def health_check(self):
        return True, ["mock"]


def mock_examples():
    triples = [
        ("A", "r1", "B"), ("A", "r1", "B2"),
        ("B", "r2", "C"), ("B", "r2", "C2"),
        ("C", "r3", "ANSWER"), ("C", "r3", "WRONG"),
        ("B2", "r2", "D"), ("D", "r3", "E"),
    ]
    yield {
        "id": "smoke-1",
        "question": "What is the answer reachable from A via r1, r2, r3?",
        "q_entity": ["A"],
        "answers": ["ANSWER"],
        "triples": triples,
    }


if __name__ == "__main__":
    import config
    from graphcot_agent import GraphCoTAgent
    from graph_env import build_undirected_graph
    import agee_metric

    llm = MockLLM()
    ex = next(mock_examples())
    G = build_undirected_graph(ex["triples"])
    agent = GraphCoTAgent(llm, ex["triples"], config.MAX_STEPS)
    res = agent.run(ex["question"])
    print("answer :", res["answer"])
    print("visited:", res["visited"])
    print("n_steps:", res["n_steps"])
    m = agee_metric.compute_agee(res["visited"], G)
    print("AGEE   :", {k: m[k] for k in ("n", "T", "n_visited", "S", "I", "E", "AGEE", "K")})
    assert res["answer"] == "ANSWER", "answer extraction failed"
    assert m["n_visited"] >= 2, "trajectory too short"
    print("\nSMOKE TEST PASSED")
