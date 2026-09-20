# Usage: python compare.py [path/to/test_book.json]
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "rag-api"))

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    BleuScore,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    NonLLMStringSimilarity,
    RougeScore,
)

from app.config import settings
from app.llm_clients import _chat_ollama
from app.neo4j_client import expand as graph_expand
from app.qdrant_client import search
from app.workflow import _rank, embed

chat = _chat_ollama


def _retry(fn, *args, attempts=3, delay=3, **kwargs):
    # port-forward tunnels drop requests intermittently even when the pod is healthy.
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"  retrying {fn.__name__} after {e}", file=sys.stderr)
            time.sleep(delay)


METRICS = {
    "rouge_l": RougeScore(rouge_type="rougeL"),
    "bleu": BleuScore(),
    "string_similarity": NonLLMStringSimilarity(),
    "context_precision": NonLLMContextPrecisionWithReference(),
    "context_recall": NonLLMContextRecall(),
}


def run_vector_only(question: str) -> dict:
    t0 = time.time()
    vec = embed([question])[0]
    hits = _retry(search, vec)
    chunks = [h.payload.get("text", "") for h in hits]
    context = "\n\n".join(chunks)
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = chat([{"role": "user", "content": prompt}])
    return {
        "answer": answer, "retrieval_context": chunks, "prompt": prompt,
        "latency": time.time() - t0, "context_chars": len(context),
        "n_entities": 0, "n_relations": 0,
    }


def run_vector_plus_graph(question: str) -> dict:
    t0 = time.time()
    vec = embed([question])[0]
    hits = _retry(search, vec)
    vector_chunks = [h.payload.get("text", "") for h in hits]
    chunk_ids = [str(h.id) for h in hits]

    found = _retry(graph_expand, chunk_ids, question)
    graph_chunks = found["chunks"]

    fused = _rank(vector_chunks, graph_chunks, found.get("chunk_meta", {}))
    context = "\n\n".join(fused[:settings.fusion_top_k])

    sections = []
    if found["entities"]:
        sections.append("Entities:\n" + "\n".join(f"- {e}" for e in found["entities"]))
    if found["relations"]:
        sections.append("Known relations:\n" + "\n".join(f"- {r}" for r in found["relations"]))
    graph_context = "\n\n".join(sections)

    full_context = context
    if graph_context:
        full_context += (
            "\n\n--- Knowledge graph (entities and relations extracted from the "
            "same documents) ---\n" + graph_context
        )
    prompt = f"Answer the question using only the context below.\n\nContext:\n{full_context}\n\nQuestion: {question}\nAnswer:"
    answer = chat([{"role": "user", "content": prompt}])

    return {
        "answer": answer, "retrieval_context": fused[:settings.fusion_top_k], "prompt": prompt,
        "latency": time.time() - t0, "context_chars": len(full_context),
        "n_entities": len(found["entities"]), "n_relations": len(found["relations"]),
        "n_graph_only_chunks": len(set(graph_chunks) - set(vector_chunks)),
    }


def _resolve_contexts(item: dict) -> list[str]:
    """v2 books store reference CHUNK IDS; v1 stored pasted text.

    Ids are resolved against the live collection so re-chunking can never
    silently invalidate the gold set — the failure mode that made v1 score a
    correct retrieval as 0.000 once SP 800-184 was re-ingested.
    """
    if "reference_contexts" in item:
        return item["reference_contexts"]
    ids = item.get("reference_chunk_ids") or []
    if not ids:
        return []
    from app.qdrant_client import client as _qc
    recs = _qc.retrieve(collection_name=settings.qdrant_collection, ids=ids, with_payload=True)
    found = {str(r.id): r.payload.get("text", "") for r in recs}
    missing = [i for i in ids if i not in found]
    if missing:
        raise SystemExit(f"test book references {len(missing)} chunk id(s) absent from the "
                         f"collection — rebuild it with build_test_book.py")
    return [found[i] for i in ids]


def score(question: str, reference_answer: str, reference_contexts: list[str], result: dict) -> dict:
    sample = SingleTurnSample(
        user_input=question,
        response=result["answer"],
        reference=reference_answer,
        retrieved_contexts=result["retrieval_context"],
        reference_contexts=reference_contexts,
    )
    out = {}
    for name, metric in METRICS.items():
        # A negative control has no correct context by construction; scoring
        # context metrics against an empty gold set would report a fake 0.
        if not reference_contexts and name.startswith("context_"):
            out[name] = None
        else:
            out[name] = metric.single_turn_score(sample)
    return out


RESULTS_PATH = Path(__file__).parent / "results.json"


def main():
    book_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "test_book.json")
    test_book = json.loads(Path(book_path).read_text())

    # Resumable: skips already-scored questions, checkpoints after each one.
    rows = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    done_ids = {r["id"] for r in rows}

    for item in test_book:
        if item["id"] in done_ids:
            continue
        print(f"[{item['id']}] {item['question']}", file=sys.stderr)

        v = run_vector_only(item["question"])
        vg = run_vector_plus_graph(item["question"])

        refs = _resolve_contexts(item)
        v_scores = score(item["question"], item["reference_answer"], refs, v)
        vg_scores = score(item["question"], item["reference_answer"], refs, vg)

        rows.append({
            "id": item["id"], "category": item["category"], "question": item["question"],
            "vector": {**v, "scores": v_scores},
            "graph": {**vg, "scores": vg_scores},
        })
        RESULTS_PATH.write_text(json.dumps(rows, indent=2))

    rows.sort(key=lambda r: r["id"])
    print_report(rows)


def print_report(rows: list[dict]):
    metric_names = list(METRICS.keys())

    print("\n" + "=" * 100)
    print(f"{'metric':<20}{'vector mean':>14}{'graph mean':>14}{'delta':>10}{'wins(v/g/tie)':>16}")
    print("=" * 100)
    for m in metric_names:
        pairs = [(r["vector"]["scores"][m], r["graph"]["scores"][m]) for r in rows
                 if r["vector"]["scores"][m] is not None and r["graph"]["scores"][m] is not None]
        v_vals = [p[0] for p in pairs]
        g_vals = [p[1] for p in pairs]
        if not v_vals:
            continue
        wins_v = sum(1 for v, g in zip(v_vals, g_vals) if v > g)
        wins_g = sum(1 for v, g in zip(v_vals, g_vals) if g > v)
        ties = len(pairs) - wins_v - wins_g
        v_mean, g_mean = statistics.mean(v_vals), statistics.mean(g_vals)
        print(f"{m:<20}{v_mean:>14.3f}{g_mean:>14.3f}{g_mean-v_mean:>+10.3f}{f'{wins_v}/{wins_g}/{ties}':>16}")

    print("\n" + "-" * 100)
    print("operational stats")
    print("-" * 100)
    v_lat = [r["vector"]["latency"] for r in rows]
    g_lat = [r["graph"]["latency"] for r in rows]
    v_ctx = [r["vector"]["context_chars"] for r in rows]
    g_ctx = [r["graph"]["context_chars"] for r in rows]
    graph_only = [r["graph"]["n_graph_only_chunks"] for r in rows]
    print(f"latency (s):        vector mean={statistics.mean(v_lat):.2f}  graph mean={statistics.mean(g_lat):.2f}  overhead={statistics.mean(g_lat)-statistics.mean(v_lat):+.2f}")
    print(f"context size (chr): vector mean={statistics.mean(v_ctx):.0f}  graph mean={statistics.mean(g_ctx):.0f}")
    print(f"graph-only chunks contributed per question: mean={statistics.mean(graph_only):.1f}  max={max(graph_only)}")

    print("\nper-question breakdown (context_recall, the metric most sensitive to retrieval gaps):")
    for r in rows:
        v, g = r["vector"]["scores"]["context_recall"], r["graph"]["scores"]["context_recall"]
        if v is None or g is None:
            print(f"  [{r['id']:>2}] {r['category']:<26} (no gold context — negative control)")
            continue
        flag = "  <-- graph helped" if g > v + 0.05 else ("  <-- graph hurt" if g < v - 0.05 else "")
        print(f"  [{r['id']:>2}] {r['category']:<26} vector={v:.2f}  graph={g:.2f}{flag}")


if __name__ == "__main__":
    main()
