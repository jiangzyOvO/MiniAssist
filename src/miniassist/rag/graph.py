"""Actual LangGraph graph with per-thread checkpoints and bounded turns."""
from threading import Lock
from dataclasses import replace
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from .nodes import Nodes
from .settings import Settings
from .state import RAGState


def build_graph(llm, retriever, settings=None, checkpointer=None):
    settings = settings or Settings()
    if settings.policy not in {'B', 'B0', 'C', 'C_no_postcheck'}:
        raise ValueError('Unknown policy')
    is_c = settings.policy.startswith('C')
    if is_c:
        from .aspects import AspectNodes
        nodes = AspectNodes(llm, retriever, replace(settings, surface_verification=False))
    else:
        nodes = Nodes(llm, retriever, replace(settings, surface_verification=False) if settings.policy == 'B0' else settings)
    graph = StateGraph(RAGState)
    names = ("understand", "clarify", "retrieve", "retrieve_more", "assess", "generate", "review_answer", "insufficient", "finish")
    for name in names + (("plan_aspects", "audit_answer") if is_c else ()):
        graph.add_node(name, nodes.guarded(name))
    graph.add_edge(START, "understand")
    # Missing slots are hypotheses until the available evidence has been inspected.
    graph.add_conditional_edges("understand", lambda s: "finish" if s.get("error") else "retrieve")
    graph.add_conditional_edges("retrieve", lambda s: "finish" if s.get("error") else "plan_aspects" if is_c else "assess")
    if is_c:
        graph.add_conditional_edges("plan_aspects", lambda s: "finish" if s.get("error") else "assess")
        graph.add_edge("audit_answer", "finish")
    graph.add_conditional_edges("assess", lambda s: "finish" if s.get("error") else
                                {"answer": "generate", "clarify": "clarify",
                                 "retrieve_more": "retrieve_more",
                                 "insufficient": "insufficient"}[s["assessment_decision"]])
    graph.add_conditional_edges("retrieve_more", lambda s: "finish" if s.get("error") else "assess")
    def after_generation(s):
        if s.get('error') or s.get('status') not in {'answered', 'partial'}:
            return 'finish'
        if is_c:
            return 'audit_answer' if settings.policy == 'C' else 'finish'
        return 'review_answer' if settings.review_answer else 'finish'
    graph.add_conditional_edges("generate", after_generation)
    for name in ("clarify", "review_answer", "insufficient"):
        graph.add_edge(name, "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


class RAGService:
    def __init__(self, llm, retriever, settings=None, checkpointer=None):
        self.llm = llm
        self.graph = build_graph(llm, retriever, settings, checkpointer)
        # CLI service serializes calls; callers must not concurrently modify a checkpoint.
        self.lock = Lock()

    def ask(self, thread_id, message, new_topic=False):
        if not isinstance(thread_id, str) or not thread_id.strip() or len(thread_id) > 128:
            raise ValueError("Invalid thread ID")
        if not isinstance(message, str) or not message.strip() or len(message) > 1500:
            raise ValueError("Message must contain 1–1500 characters")
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 16}
        with self.lock:
            if hasattr(self.llm, 'reset_budget'):
                self.llm.reset_budget()
            previous = self.graph.get_state(config).values
            reset = new_topic or not previous.get("pending", False)
            turn = {"message": message.strip(), "force_new_topic": new_topic,
                    "user_utterances": ([message.strip()] if reset else
                                        previous.get("user_utterances", [previous["original_question"]]) + [message.strip()]),
                    "status": "running", "response": "", "error": "", "trace": [],
                    "evidence": [], "citations": [], "selected_ids": [], "sufficient": False,
                    "reason": "", "query": "", "missing": [], "clarification": "",
                    "initial_evidence": [], "retrieval_rounds": 0, "retrieval_log": [],
                    "context_expansions": [],
                    "generation_repair": {},
                    "generation_skipped": {}, "budget_fallback": {},
                    "evidence_checks": [], "critical_conditions": [], "decision_override": {},
                    "condition_evidence_added": [],
                    "assessment_format_repair": {},
                    "assessment_route_repair": {}, "generation_format_repair": {},
                    "condition_locator_repairs": [],
                    "verification_requests": [],
                    "draft_answer": {}, "generation_review": {}, "generation_base_clarification_count": 0,
                    "retrieval_queries": [], "answerable_parts": [], "missing_parts": [],
                    "aspect_plan": {}, "aspect_coverage": [], "aspect_queries": [],
                    "output_audits": [], "aspect_revision": {}, "output_fallback": {},
                    "boundary": "none", "partial_answer": False, "assessment_decision": ""}
            if reset:
                turn.update(question="", original_question=message.strip(),
                            slots={}, clarification_count=0, pending=False)
            if new_topic:
                turn["history"] = []
            return self.graph.invoke(turn, config=config)
