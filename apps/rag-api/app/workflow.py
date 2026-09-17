import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langsmith import traceable
from .config import settings
from .neo4j_client import expand as expand_knowledge_graph
from .qdrant_client import search
from .llm_clients import embed, chat

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    question: str
    chunk_ids: list[str]
    vector_chunks: list[str]
    graph_chunks: list[str]
    graph_context: str
    context: str
    answer: str


def retrieve(state: RAGState) -> RAGState:
    vec = embed([state["question"]])[0]
    hits = search(vec)
    state["vector_chunks"] = [h.payload.get("text", "") for h in hits]
    # Chunk ids double as Neo4j (:Chunk) ids — see ingestion-worker's chunk_id().
    state["chunk_ids"] = [str(h.id) for h in hits]
    return state


@traceable(name="graph-expand", run_type="retriever")
def expand_graph(state: RAGState) -> RAGState:
    """Non-fatal: falls back to vector chunks alone if Neo4j is down or empty."""
    state["graph_chunks"] = []
    state["graph_context"] = ""
    if not settings.graph_enabled:
        return state

    try:
        found = expand_knowledge_graph(state["chunk_ids"], state["question"])
    except Exception as e:
        logger.warning("Graph expansion failed (%s), answering from vector context only", e)
        return state

    state["graph_chunks"] = found["chunks"]

    sections = []
    if found["entities"]:
        sections.append("Entities:\n" + "\n".join(f"- {e}" for e in found["entities"]))
    if found["relations"]:
        sections.append("Known relations:\n" + "\n".join(f"- {r}" for r in found["relations"]))
    state["graph_context"] = "\n\n".join(sections)
    return state


def fuse(state: RAGState) -> RAGState:
    """Combines vector and graph chunk lists via Reciprocal Rank Fusion — rank-based,
    since cosine similarity and entity-overlap counts aren't on a comparable scale."""
    fused = _reciprocal_rank_fusion(
        [state["vector_chunks"], state["graph_chunks"]], k=settings.rrf_k
    )
    state["context"] = "\n\n".join(fused[:settings.fusion_top_k])
    return state


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def generate(state: RAGState) -> RAGState:
    context = state["context"]
    if state.get("graph_context"):
        context += (
            "\n\n--- Knowledge graph (entities and relations extracted from the "
            "same documents) ---\n" + state["graph_context"]
        )
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer:"
    )
    state["answer"] = chat([{"role": "user", "content": prompt}])
    return state


def build_workflow():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("expand_graph", expand_graph)
    g.add_node("fuse", fuse)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "expand_graph")
    g.add_edge("expand_graph", "fuse")
    g.add_edge("fuse", "generate")
    g.add_edge("generate", END)
    return g.compile()


rag_workflow = build_workflow()
