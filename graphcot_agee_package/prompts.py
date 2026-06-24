"""
KG-adapted Graph-CoT prompts.

Faithful to PeterGriffinJin/Graph-CoT's prompt scaffold (graph definition +
few-shot Thought/Action/Observation exemplars + scratchpad), but the graph
definition and exemplars are written for a Freebase-style KG-QA subgraph
(entities + relation edges) instead of the GRBench domain graphs.
"""

GRAPH_DEFINITION = (
    "There is a knowledge graph. Nodes are entities (people, places, films, "
    "organisations, dates, etc.). Each entity has a 'name' feature. Edges are "
    "labelled by a relation; you can traverse a relation from an entity to its "
    "connected entities, or traverse its inverse (prefixed with '~') in the "
    "opposite direction."
)

# Available actions are described to the model. RetrieveNode also returns the
# list of relations available at the matched entity, so the agent knows what it
# can traverse next.
INSTRUCTION = (
    "Solve a knowledge-graph question-answering task by reasoning step by step "
    "and interacting with the graph. At each step write one short Thought and "
    "then exactly one Action. An Action MUST be a bare command of the form "
    "Verb[arguments] -- never a sentence. The available actions are:\n"
    "(1) RetrieveNode[entity name]: find the graph entity best matching the "
    "name; returns its node id and the relations available at it.\n"
    "(2) NeighbourCheck[node id, relation]: return the entities connected to "
    "the node via that relation.\n"
    "(3) NodeFeature[node id, name]: return the surface name of the node.\n"
    "(4) NodeDegree[node id, relation]: return how many entities are connected "
    "via that relation.\n"
    "(5) Finish[answer]: return the final answer entity (or entities).\n"
    "IMPORTANT: if the topic entity is already located for you (shown in "
    "Observation 0), do NOT call RetrieveNode again -- immediately use "
    "NeighbourCheck[topic entity, relation] to traverse toward the answer, "
    "choosing the most relevant relation from those listed. Use at most "
    "{max_steps} steps."
)

# Two in-context exemplars in the exact Thought/Action/Observation format.
EXEMPLAR = """Question: What language is spoken in the country where the Eiffel Tower is located?
Observation 0: The topic entity has been located: "Eiffel Tower". Available relations: ['location.location.containedby'].
Thought 1: The topic entity is already located, so I traverse the containedby relation to find its country.
Action 1: NeighbourCheck[Eiffel Tower, location.location.containedby]
Observation 1: The location.location.containedby neighbors of Eiffel Tower are: ['France'].
Thought 2: Now I need the language spoken in France.
Action 2: NeighbourCheck[France, location.country.official_language]
Observation 2: The location.country.official_language neighbors of France are: ['French'].
Thought 3: The language spoken is French.
Action 3: Finish[French]

Question: Who directed the film Inception?
Observation 0: The topic entity has been located: "Inception". Available relations: ['film.film.directed_by', 'film.film.starring'].
Thought 1: The topic entity is located; I traverse directed_by to find the director.
Action 1: NeighbourCheck[Inception, film.film.directed_by]
Observation 1: The film.film.directed_by neighbors of Inception are: ['Christopher Nolan'].
Thought 2: The director is Christopher Nolan.
Action 2: Finish[Christopher Nolan]"""


def build_prompt(question, scratchpad, max_steps, step_n):
    return (
        f"{INSTRUCTION.format(max_steps=max_steps)}\n\n"
        f"Graph definition: {GRAPH_DEFINITION}\n\n"
        f"Here are two examples:\n{EXEMPLAR}\n\n"
        f"Now solve this question.\n"
        f"Question: {question}{scratchpad}\n"
        f"Thought {step_n}:"
    )
