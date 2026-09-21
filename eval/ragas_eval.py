"""Ragas evaluation, including retrieval (skipped by earlier passes): non-LLM context metrics
vs gold chunks, plus LLM-judged faithfulness/relevancy applicable to all four arms.
substantial chunks, measuring verbosity rather than retrieval quality. Sampling
from both ends bounds the call count equally across arms while keeping local's
trailing source units in view.

Run: python3 eval/ragas_eval.py        (system python — it has a working ragas)
"""
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from azure_judge import _load_credentials  # noqa: E402

RESULTS = Path(__file__).parent / "results_graphrag.json"
GOLD = Path(__file__).parent / "gold_context.json"
OUT = Path(__file__).parent / "results_ragas.json"

ARMS = ["basic", "local", "global", "dynamic"]
LLM_CONTEXT_ITEMS = 12


def _embed_credentials() -> dict:
    env = Path(__file__).resolve().parents[1] / "graphrag" / ".env"
    v = dict(line.split("=", 1) for line in env.read_text().splitlines()
             if "=" in line and not line.startswith("#"))
    return {"api_key": v["GRAPHRAG_EMBED_API_KEY"], "endpoint": v["GRAPHRAG_EMBED_API_BASE"],
            "api_version": v["GRAPHRAG_EMBED_API_VERSION"],
            "deployment": v["GRAPHRAG_EMBED_DEPLOYMENT"]}


def build():
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (ContextEntityRecall, Faithfulness, FactualCorrectness,
                               LLMContextPrecisionWithReference, LLMContextRecall,
                               NonLLMContextPrecisionWithReference, NonLLMContextRecall,
                               ResponseRelevancy, SemanticSimilarity)

    c, e = _load_credentials(), _embed_credentials()
    # bypass_temperature/bypass_n stop ragas overwriting params gpt-5-nano rejects (temp != 1, `n`).
    llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=c["endpoint"], api_key=c["api_key"], api_version=c["api_version"],
            azure_deployment=c["deployment"], temperature=1, reasoning_effort="minimal",
            max_completion_tokens=3000, max_retries=8),
        bypass_temperature=True, bypass_n=True)
    emb = LangchainEmbeddingsWrapper(AzureOpenAIEmbeddings(
        azure_endpoint=e["endpoint"], api_key=e["api_key"], api_version=e["api_version"],
        azure_deployment=e["deployment"], max_retries=8))

    return {
        "retriever": {
            "context_precision": NonLLMContextPrecisionWithReference(),
            "context_recall": NonLLMContextRecall(),
        },
        # context_entity_recall: how many reference entities appear in retrieved context.
        "retriever_llm": {
            "context_entity_recall": ContextEntityRecall(llm=llm),
            "llm_context_precision": LLMContextPrecisionWithReference(llm=llm),
            "llm_context_recall": LLMContextRecall(llm=llm),
        },
        "answer": {
            "faithfulness": Faithfulness(llm=llm),
            "response_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
            # ragas' own correctness: claim-overlap scoring, vs DeepEval's holistic GEval number.
            "factual_correctness": FactualCorrectness(llm=llm),
            "semantic_similarity": SemanticSimilarity(embeddings=emb),
        },
    }


def _score_one(task, metrics):
    from ragas.dataset_schema import SingleTurnSample

    row, arm, gold, reference = task
    d = row["arms"][arm]
    full = d["retrieval_context"] or ["(no context returned)"]

    # Head-and-tail sample: equal judge budget per arm; local's source text sits at the tail.
    if len(full) <= LLM_CONTEXT_ITEMS:
        capped = list(full)
    else:
        h = LLM_CONTEXT_ITEMS // 2
        capped = list(full[:h]) + list(full[-(LLM_CONTEXT_ITEMS - h):])

    scores = {}
    if gold:
        sample = SingleTurnSample(user_input=row["question"], response=d["answer"],
                                  retrieved_contexts=full, reference_contexts=gold)
        for name, m in metrics["retriever"].items():
            try:
                scores[name] = round(float(m.single_turn_score(sample)), 4)
            except Exception as ex:
                print(f"   [{row['id']}|{arm}] {name}: {ex}", file=sys.stderr)
                scores[name] = None

    sample = SingleTurnSample(user_input=row["question"], response=d["answer"],
                              retrieved_contexts=capped, reference=reference)
    for name, m in {**metrics["retriever_llm"], **metrics["answer"]}.items():
        try:
            scores[name] = round(float(m.single_turn_score(sample)), 4)
        except Exception as ex:
            print(f"   [{row['id']}|{arm}] {name}: {str(ex)[:90]}", file=sys.stderr)
            scores[name] = None

    print(f"[{row['id']}|{row['tier']}] {arm}: "
          + " ".join(f"{k}={v}" for k, v in scores.items()), file=sys.stderr)
    return {"id": row["id"], "tier": row["tier"], "arm": arm,
            "has_gold": bool(gold), "scores": scores}


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4

    rows = json.loads(RESULTS.read_text())
    gold = json.loads(GOLD.read_text())
    book = {b["id"]: b for b in
            json.loads((Path(__file__).parent / "test_book_v3.json").read_text())}
    metrics = build()

    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["id"], r["arm"]) for r in out}

    tasks = [(r, arm, gold.get(str(r["id"]), {}).get("gold_texts", []),
              book[r["id"]]["reference_answer"])
             for r in rows for arm in ARMS
             if arm in r["arms"] and (r["id"], arm) not in done]
    print(f"ragas: {len(tasks)} arm-results, {workers} workers", file=sys.stderr)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_one, t, metrics) for t in tasks]
        for f in as_completed(futures):
            try:
                res = f.result()
            except Exception as ex:
                print(f"  task failed: {str(ex)[:120]}", file=sys.stderr)
                continue
            with lock:
                out.append(res)
                OUT.write_text(json.dumps(out, indent=2))

    out.sort(key=lambda r: (r["id"], r["arm"]))
    OUT.write_text(json.dumps(out, indent=2))
    report(out)


def report(out):
    def mean(rows, key):
        v = [r["scores"].get(key) for r in rows if r["scores"].get(key) is not None]
        return statistics.mean(v) if v else None

    print("\n" + "=" * 88)
    print("RETRIEVER METRICS — non-LLM, retrieved text vs gold text units")
    print("questions with an answer-bearing chunk only (local + cross-document, n=9)")
    print("=" * 88)
    gold_rows = [r for r in out if r["has_gold"]]
    print(f"  {'metric':<24}" + "".join(f"{a:>15}" for a in ARMS))
    for k in ["context_precision", "context_recall"]:
        cells = []
        for a in ARMS:
            m = mean([r for r in gold_rows if r["arm"] == a], k)
            cells.append(f"{m:>15.3f}" if m is not None else f"{'-':>15}")
        print(f"  {k:<24}" + "".join(cells))
    print("\n  global and dynamic retrieve community reports, not chunks. Their scores")
    print("  here describe what they retrieve, not how well they retrieve it.")

    print("\n" + "=" * 88)
    print("ANSWER METRICS — LLM-judged, all 14 questions")
    print("=" * 88)
    for tier in ["local", "cross-document", "global", "negative-control"]:
        sub = [r for r in out if r["tier"] == tier]
        if not sub:
            continue
        print(f"\n{tier}  (n={len({r['id'] for r in sub})})")
        print(f"  {'metric':<24}" + "".join(f"{a:>15}" for a in ARMS))
        for k in ["faithfulness", "response_relevancy", "factual_correctness",
                  "semantic_similarity", "context_entity_recall",
                  "llm_context_precision", "llm_context_recall"]:
            cells = []
            for a in ARMS:
                m = mean([r for r in sub if r["arm"] == a], k)
                cells.append(f"{m:>15.3f}" if m is not None else f"{'-':>15}")
            print(f"  {k:<24}" + "".join(cells))


if __name__ == "__main__":
    main()
