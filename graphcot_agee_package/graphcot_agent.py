"""
Graph-CoT agent for KG-QA.

Faithful to PeterGriffinJin/Graph-CoT's GraphAgent (Think -> Act -> Observe
loop with RetrieveNode / NeighbourCheck / NodeFeature / NodeDegree / Finish),
adapted to a Freebase-style subgraph. While running, it records the ordered
sequence of entities the agent *focuses on* (the trajectory tau) so AGEE can be
computed on it, comparably to your ToG/PoG trajectories.
"""
import re
import prompts
from graph_env import GraphFuncs, Retriever, build_graphcot_graph, norm

ACTION_RE = re.compile(r"^\s*(\w+)\[(.+)\]\s*$")
# Robust: find the first valid action command anywhere in the model output,
# even when Llama wraps it in prose ("I should NeighbourCheck[X, r] now").
VALID_VERBS = ("RetrieveNode", "NeighbourCheck", "NodeFeature",
               "NodeDegree", "Finish")
ACTION_FINDALL = re.compile(
    r"(RetrieveNode|NeighbourCheck|NodeFeature|NodeDegree|Finish)\s*\[([^\]]+)\]")


def extract_action(text):
    """Return 'Verb[args]' for the first valid command in `text`, else None."""
    if not text:
        return None
    m = ACTION_FINDALL.search(text)
    if m:
        return f"{m.group(1)}[{m.group(2).strip()}]"
    return None


def parse_action(text):
    m = ACTION_RE.match(text.strip())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def strip_quotes(s):
    s = s.strip()
    if len(s) >= 2 and s[0] in "'\"" and s[-1] in "'\"":
        return s[1:-1]
    return s


class GraphCoTAgent:
    def __init__(self, llm, triples, max_steps):
        self.llm = llm
        self.max_steps = max_steps
        graph, _ = build_graphcot_graph(triples)
        self.funcs = GraphFuncs(graph)
        self.retriever = Retriever(self.funcs.index.keys())
        self.reset()

    def reset(self):
        self.scratchpad = ""
        self.step_n = 1
        self.finished = False
        self.answer = ""
        self.visited = []          # ordered trajectory tau (entity ids)
        self._visited_set = set()
        self.n_hops = 0            # successful NeighbourCheck edge traversals
        self.edges = []            # traversed (h, r, t) triples == reasoning_chains
        self._pending = None       # (X, r, frozenset(neighbours)) from last check

    def _visit(self, node):
        """Record that the agent focused on `node` (in graph)."""
        if node is None:
            return
        self.visited.append(node)
        self._visited_set.add(node)

    def _commit_to(self, node):
        """If `node` was a neighbour returned by the most recent NeighbourCheck,
        the agent has moved along that edge -> record the traversed triple
        (X, r, node). This reconstructs the same edge-based reasoning_chain that
        ToG/PoG emit, so tog_pog_parser.py builds an identical trajectory."""
        if node is None or self._pending is None:
            return
        X, r, neigh = self._pending
        if node in neigh and node != X:
            self.edges.append((X, r, node))
            self._pending = None

    def _match_node(self, text):
        """Return a graph node id whose name exactly matches `text` (normalised),
        else None. Used to record the committed answer entity as a visited node."""
        if not text:
            return None
        t = norm(text)
        # exact normalised match only (no fuzzy) to avoid false positives
        for nid in self.funcs.index:
            if norm(nid) == t:
                return nid
        return None

    def _resolve_seed(self, q_entity):
        """Map the benchmark-provided topic entity (RoG q_entity) to a graph
        node. WebQSP/CWQ provide the topic entity, so the agent starts from it
        rather than guessing it via string retrieval (matches ToG/PoG)."""
        if not q_entity:
            return None
        cands = q_entity if isinstance(q_entity, list) else [q_entity]
        # exact normalised match first
        for q in cands:
            nq = norm(q)
            for nid in self.funcs.index:
                if norm(nid) == nq:
                    return nid
        # fall back to the retriever for near-matches
        for q in cands:
            hit = self.retriever.search_single(q)
            if hit is not None:
                return hit
        return None

    def _generate_thought_action(self):
        """Single generation of one Thought + one Action. Robust to Llama-style
        outputs that bury the command in prose: extract the first valid Verb[...]
        anywhere; if none, re-prompt once strictly for an action."""
        prompt = prompts.build_prompt(
            self.question, self.scratchpad, self.max_steps, self.step_n)
        gen = self.llm.generate(prompt, stop=("\nObservation", "Observation:"),
                                max_tokens=200)
        action = extract_action(gen)
        # thought = text before the action verb (best-effort, for the log)
        thought = gen
        m = ACTION_FINDALL.search(gen)
        if m:
            thought = gen[:m.start()]
        thought = re.sub(r"(?is)\bAction\s*\d*\s*:.*$", "", thought)
        thought = thought.replace("\n", " ").strip()[:300] or "(reasoning)"

        if action is None:
            # strict single re-prompt: force a bare command
            strict = (prompt + f" {thought}\nAction {self.step_n}: ")
            gen2 = self.llm.generate(strict, stop=("\n",), max_tokens=80)
            action = extract_action(gen2) or extract_action("X[" + gen2 + "]")
        return thought, action

    def step(self):
        thought, action = self._generate_thought_action()
        self.scratchpad += f"\nThought {self.step_n}: {thought}"
        if action is None:
            # could not obtain a valid action -> stop (answer falls back to v_T)
            self.scratchpad += (f"\nAction {self.step_n}: (no valid action)"
                                f"\nObservation {self.step_n}: stopping.")
            self.finished = True
            self.step_n += 1
            return
        self.scratchpad += f"\nAction {self.step_n}: {action}"

        atype, arg = parse_action(action)
        obs = ""
        if atype == "Finish":
            self.answer = strip_quotes(arg)
            # Fix 1: record the committed answer entity as the terminal visited
            # node, so a 1-hop seed->answer traversal is a length-2 trajectory
            # (matches the paper's "trajectory length >= 2" validity rule).
            ans_node = self._match_node(self.answer)
            if ans_node is not None:
                self._commit_to(ans_node)   # edge X->answer if it was a neighbour
                self._visit(ans_node)
            self.finished = True
            self.scratchpad += f"\nObservation {self.step_n}: Finished."
            self.step_n += 1
            return
        elif atype == "RetrieveNode":
            nid = self.retriever.search_single(strip_quotes(arg))
            if nid is None:
                obs = "No matching entity found. Try another name."
            else:
                self._commit_to(nid)
                self._visit(nid)
                self._pending = None         # retrieval is a jump; break the chain
                rels = self.funcs.relations(nid)
                obs = (f'The id of this entity is "{nid}". '
                       f"Available relations: {rels[:25]}.")
        elif atype == "NeighbourCheck":
            try:
                node_id, rel = [strip_quotes(x) for x in arg.split(",", 1)]
                node_id, rel = node_id.strip(), rel.strip()
                self._commit_to(node_id)     # moved to node_id if it was a neighbour
                neigh = self.funcs.check_neighbours(node_id, rel)
                self._visit(node_id)
                if neigh:
                    self.n_hops += 1         # one edge traversal (cf. ToG entity_search)
                    self._pending = (node_id, rel, frozenset(neigh))
                obs = f"The {rel} neighbors of {node_id} are: {neigh[:30]}."
            except KeyError:
                obs = "That node or relation does not exist. Modify it."
            except Exception:
                obs = ("NeighbourCheck needs two arguments: node id and relation.")
        elif atype == "NodeFeature":
            try:
                node_id, feat = [strip_quotes(x) for x in arg.split(",", 1)]
                node_id = node_id.strip()
                self._commit_to(node_id)
                self._visit(node_id)
                obs = f"The name feature of {node_id} is: {self.funcs.check_nodes(node_id, 'name')}."
            except Exception:
                obs = "NodeFeature needs: node id and feature name."
        elif atype == "NodeDegree":
            try:
                node_id, rel = [strip_quotes(x) for x in arg.split(",", 1)]
                node_id, rel = node_id.strip(), rel.strip()
                self._commit_to(node_id)
                self._visit(node_id)
                obs = f"The {rel} degree of {node_id} is: {self.funcs.check_degree(node_id, rel)}."
            except Exception:
                obs = "NodeDegree needs: node id and relation."
        else:
            obs = ("Invalid Action. Valid: RetrieveNode[name] "
                   "NeighbourCheck[node, relation] NodeFeature[node, name] "
                   "NodeDegree[node, relation] Finish[answer].")

        self.scratchpad += f"\nObservation {self.step_n}: {obs}"
        self.step_n += 1

    def run(self, question, q_entity=None):
        self.reset()
        self.question = question
        # Fix 2: start from the benchmark-provided topic entity (entity linking
        # is given for WebQSP/CWQ, as in ToG/PoG), instead of guessing it via
        # fuzzy string retrieval (which mis-linked e.g. "James K. Polk" -> a
        # place named "Polk").
        seed = self._resolve_seed(q_entity)
        if seed is not None:
            self._visit(seed)
            rels = self.funcs.relations(seed)
            self.scratchpad += (
                f"\nObservation 0: The topic entity has been located: \"{seed}\". "
                f"Available relations: {rels[:25]}.")
        while not self.finished and self.step_n <= self.max_steps:
            try:
                self.step()
            except Exception as e:
                self.scratchpad += f"\nObservation {self.step_n}: internal error {e}"
                self.step_n += 1
        # Answer-extraction fallback (manuscript protocol). If the agent ended
        # without an explicit Finish, the answer is the last entity it actually
        # traversed TO (the tail of the last committed edge) -- not the starting
        # seed. Only if no edge was traversed do we fall back to the last node.
        if not str(self.answer).strip():
            if self.edges:
                self.answer = self.edges[-1][2]
            elif self.visited:
                self.answer = self.visited[-1]
        return {
            "answer": self.answer,
            "visited": self.visited,          # trajectory tau for AGEE
            "n_steps": self.step_n - 1,
            "n_hops": self.n_hops,
            "edges": self.edges,            # reasoning_chains for tog_pog_parser
            "scratchpad": self.scratchpad,
        }
