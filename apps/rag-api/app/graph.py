from typing import TypedDict
from langgraph.graph import StateGraph, END
from .qdrant_client import search
from .gemini_client import embed, chat


class RAGState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: RAGState) -> RAGState:
    vec = embed([state["question"]])[0]
    hits = search(vec)
    state["context"] = "\n".join(h.payload.get("text", "") for h in hits)
    return state


def generate(state: RAGState) -> RAGState:
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{state['context']}\n\nQuestion: {state['question']}\nAnswer:"
    )
    state["answer"] = chat([{"role": "user", "content": prompt}])
    return state


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


rag_graph = build_graph()
