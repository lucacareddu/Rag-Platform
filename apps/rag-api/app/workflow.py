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
    graph_meta: dict
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
    state["graph_meta"] = {}
    state["graph_context"] = ""
    if not settings.graph_enabled:
        return state

    try:
        found = expand_knowledge_graph(state["chunk_ids"], state["question"])
    except Exception as e:
        logger.warning("Graph expansion failed (%s), answering from vector context only", e)
        return state

    state["graph_chunks"] = found["chunks"]
    state["graph_meta"] = found.get("chunk_meta", {})

    sections = []
    if found["entities"]:
        sections.append("Entities:\n" + "\n".join(f"- {e}" for e in found["entities"]))
    if found["relations"]:
        sections.append("Known relations:\n" + "\n".join(f"- {r}" for r in found["relations"]))
    state["graph_context"] = "\n\n".join(sections)
    return state


def fuse(state: RAGState) -> RAGState:
    """Ranks vector and graph candidates into one context window.

    RRF is wrong for this pair. It rewards agreement between lists, but a chunk
    the graph reached and vector search did not is exactly the chunk worth
    adding — under RRF it scores once, from one list, and loses to any chunk
    both lists merely agree on. So graph-unique chunks, the only ones that can
    beat a pure-vector baseline, were the first thing discarded.
    """
    state["context"] = "\n\n".join(
        _rank(state["vector_chunks"], state["graph_chunks"],
              state.get("graph_meta") or {})[:settings.fusion_top_k]
    )
    return state


def _rank(vector_chunks: list[str], graph_chunks: list[str],
          meta: dict[str, dict]) -> list[str]:
    """Vector rank order is the relevance backbone; graph-unique chunks compete
    for a reserved share of the budget on how *specific* their link was."""
    vector_set = set(vector_chunks)
    ranked = list(vector_chunks)

    novel = [c for c in graph_chunks if c not in vector_set]
    if not novel:
        return ranked

    def novelty(chunk: str) -> float:
        """Scored against measured chunk quality on this corpus, not intuition.

        Two plausible-sounding signals were checked and both ran the wrong way:
        2-hop chunks scored worse than 1-hop (0.10 vs 0.16 overlap with
        reference answers), and chunks reached via a *rare* entity scored worse
        than mid-frequency ones (0.14 vs 0.22) — a rare entity is usually an
        extraction artefact, not a precise link. Only shared-entity support
        tracked quality monotonically (0.10 -> 0.19 from 1 to 3+ shared
        entities), so it carries the ranking and distance is a mild penalty.
        """
        m = meta.get(chunk, {})
        # `evidence` is overlap with each entity discounted by how common it is;
        # falls back to raw overlap for chunks from an older index.
        return m.get("evidence", m.get("overlap", 1)) - 0.5 * (m.get("hop", 1) - 1)

    novel.sort(key=novelty, reverse=True)

    reserved = max(0, min(settings.graph_reserved_slots,
                          settings.fusion_top_k - settings.graph_min_vector_slots))
    # Interleave, don't append: reserved chunks must land inside the top_k cut.
    out = ranked[:settings.graph_min_vector_slots]
    out += novel[:reserved]
    out += [c for c in ranked[settings.graph_min_vector_slots:] if c not in out]
    out += [c for c in novel[reserved:] if c not in out]
    return out


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
