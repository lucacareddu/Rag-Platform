"""Deterministic ragas metrics, no judge: retrieval (precision/recall vs gold) and answer quality
(ROUGE/BLEU/string/semantic similarity). ROUGE-family metrics favour short answers -- read with care.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = Path(__file__).parent / "results_graphrag.json"
GOLD = Path(__file__).parent / "gold_context.json"
BOOK = Path(__file__).parent / "test_book_v3.json"
OUT = Path(__file__).parent / "results_ragas_nonllm.json"

ARMS = ["basic", "local", "global", "dynamic"]


def _embedder():
    """Direct client rather than ragas' SemanticSimilarity, which drives the
    embeddings asynchronously and raises "Event loop is closed" when scored one
    sample at a time. Cosine over the same text-embedding-3-small deployment is
    the same computation without the async plumbing."""
    from langchain_openai import AzureOpenAIEmbeddings
    env = Path(__file__).resolve().parents[1] / "graphrag" / ".env"
    v = dict(line.split("=", 1) for line in env.read_text().splitlines()
             if "=" in line and not line.startswith("#"))
    return AzureOpenAIEmbeddings(
        azure_endpoint=v["GRAPHRAG_EMBED_API_BASE"], api_key=v["GRAPHRAG_EMBED_API_KEY"],
        api_version=v["GRAPHRAG_EMBED_API_VERSION"],
        azure_deployment=v["GRAPHRAG_EMBED_DEPLOYMENT"], max_retries=8)


def _cosine(emb, a: str, b: str) -> float:
    va, vb = emb.embed_documents([a[:30000], b[:30000]])
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(x * x for x in vb) ** 0.5
    return dot / (na * nb)


def build():
    from ragas.metrics import (BleuScore, NonLLMContextPrecisionWithReference,
                               NonLLMContextRecall, NonLLMStringSimilarity,
                               RougeScore)
    return {
        "retriever": {
            "context_precision": NonLLMContextPrecisionWithReference(),
            "context_recall": NonLLMContextRecall(),
        },
        "answer": {
            "rouge_l": RougeScore(rouge_type="rougeL"),
            "bleu": BleuScore(),
            "string_similarity": NonLLMStringSimilarity(),
        },
    }


def main():
    from ragas.dataset_schema import SingleTurnSample

    rows = json.loads(RESULTS.read_text())
    gold = json.loads(GOLD.read_text())
    book = {b["id"]: b for b in json.loads(BOOK.read_text())}
    metrics = build()
    emb = _embedder()

    out = []
    for row in rows:
        gold_texts = gold.get(str(row["id"]), {}).get("gold_texts", [])
        reference = book[row["id"]]["reference_answer"]
        for arm in ARMS:
            if arm not in row["arms"]:
                continue
            d = row["arms"][arm]
            scores = {}

            if gold_texts:
                # Full context: these metrics are free, and local search puts
                # its source text units at the TAIL of its payload, so any
                # head-only cap would discard exactly what is being scored.
                s = SingleTurnSample(
                    user_input=row["question"], response=d["answer"],
                    retrieved_contexts=d["retrieval_context"] or ["(none)"],
                    reference_contexts=gold_texts)
                for name, m in metrics["retriever"].items():
                    try:
                        scores[name] = round(float(m.single_turn_score(s)), 4)
                    except Exception as e:
                        print(f"  [{row['id']}|{arm}] {name}: {e}", file=sys.stderr)
                        scores[name] = None

            s = SingleTurnSample(user_input=row["question"], response=d["answer"],
                                 retrieved_contexts=d["retrieval_context"] or ["(none)"],
                                 reference=reference)
            for name, m in metrics["answer"].items():
                try:
                    scores[name] = round(float(m.single_turn_score(s)), 4)
                except Exception as e:
                    print(f"  [{row['id']}|{arm}] {name}: {str(e)[:80]}", file=sys.stderr)
                    scores[name] = None

            try:
                scores["semantic_similarity"] = round(_cosine(emb, d["answer"], reference), 4)
            except Exception as e:
                print(f"  [{row['id']}|{arm}] semantic_similarity: {str(e)[:70]}", file=sys.stderr)
                scores["semantic_similarity"] = None

            out.append({"id": row["id"], "tier": row["tier"], "arm": arm,
                        "has_gold": bool(gold_texts),
                        "answer_chars": len(d["answer"]),
                        "n_context": len(d["retrieval_context"]),
                        "scores": scores})
        print(f"[{row['id']}] done", file=sys.stderr)

    OUT.write_text(json.dumps(out, indent=2))
    report(out)


def report(out):
    def mean(rows, key):
        v = [r["scores"].get(key) for r in rows if r["scores"].get(key) is not None]
        return statistics.mean(v) if v else None

    print("\n" + "=" * 86)
    print("RETRIEVER — deterministic, retrieved context vs gold text units")
    print("=" * 86)
    g = [r for r in out if r["has_gold"]]
    for label, sub in [("all answerable (n=9)", g),
                       ("local / factual (n=4)", [r for r in g if r["tier"] == "local"]),
                       ("cross-document (n=5)", [r for r in g if r["tier"] == "cross-document"])]:
        print(f"\n{label}")
        print(f"  {'metric':<22}" + "".join(f"{a:>13}" for a in ARMS))
        for k in ["context_precision", "context_recall"]:
            cells = []
            for a in ARMS:
                m = mean([r for r in sub if r["arm"] == a], k)
                cells.append(f"{m:>13.3f}" if m is not None else f"{'-':>13}")
            print(f"  {k:<22}" + "".join(cells))
    print("\n  global and dynamic retrieve community reports rather than chunks, so their")
    print("  scores here describe WHAT they retrieve, not how well.")

    print("\n" + "=" * 86)
    print("ANSWER — deterministic, generated answer vs reference answer")
    print("=" * 86)
    for tier in ["local", "cross-document", "global", "negative-control"]:
        sub = [r for r in out if r["tier"] == tier]
        if not sub:
            continue
        print(f"\n{tier}  (n={len({r['id'] for r in sub})})")
        print(f"  {'metric':<22}" + "".join(f"{a:>13}" for a in ARMS))
        for k in ["rouge_l", "bleu", "string_similarity", "semantic_similarity"]:
            cells = []
            for a in ARMS:
                m = mean([r for r in sub if r["arm"] == a], k)
                cells.append(f"{m:>13.3f}" if m is not None else f"{'-':>13}")
            print(f"  {k:<22}" + "".join(cells))
        cells = []
        for a in ARMS:
            v = [r["answer_chars"] for r in sub if r["arm"] == a]
            cells.append(f"{statistics.mean(v):>13.0f}" if v else f"{'-':>13}")
        print(f"  {'(answer chars)':<22}" + "".join(cells))
    print("\n  rouge_l / bleu / string_similarity reward surface overlap with the reference")
    print("  wording and so penalise long answers. Compare them against the answer-length")
    print("  row before reading any of them as a quality difference.")


if __name__ == "__main__":
    main()
