"""Re-score saved GraphRAG answers with a corrected judging procedure.

Two defects in the first scoring pass made cross-arm comparison invalid:

1. Citation bias. GraphRAG answers carry inline provenance markers such as
   "[Data: Reports (2, 6); Entities (11)]". The judge docked points for them —
   its own stated reason on Q2 was that the answer "restates with [Data:
   Sources ...] annotations not in the expected". Local and global search emit
   far more of these than basic search does, so the metric was systematically
   penalising the arms that cite their evidence. Markers are stripped from
   every arm before judging.

2. Judge variance. gpt-5 models reject any temperature other than 1, so the
   judge cannot be made deterministic. Re-measuring one unchanged test case
   returned 0.5 and then 0.3 — a swing large enough to reorder arms. Each
   metric is therefore sampled `SAMPLES` times and the median is kept, with
   the spread recorded so the noise floor is visible rather than assumed.

Retrieval is not repeated: answers come from results_graphrag.json, where a
single global-search query cost 254 seconds.

Run: .venv-graphrag/bin/python eval/rescore.py [--samples 3]
"""
import argparse
import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from azure_judge import AzureGPT5Nano  # noqa: E402

RESULTS = Path(__file__).parent / "results_graphrag.json"
OUT = Path(__file__).parent / "results_graphrag_rescored.json"
BOOK = Path(__file__).parent / "test_book_v3.json"

# "[Data: Reports (2, 6, +more); Entities (11)]", "[Sources: ...]", trailing "+more".
_CITATION = re.compile(r"\[\s*(?:Data|Sources?|Entities|Relationships|Reports)\s*:[^\]]*\]",
                       re.I)
_TRAILING = re.compile(r"\s*\+more\b", re.I)


def strip_citations(text: str) -> str:
    return _TRAILING.sub("", _CITATION.sub("", text)).strip()


def build_metrics(judge):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCaseParams as P

    # Rubric anchors give the judge fixed reference points instead of letting
    # it invent a scale each call, which is where much of the variance came from.
    correctness = GEval(
        name="correctness",
        model=judge,
        async_mode=False,
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        evaluation_steps=[
            "Identify the distinct factual claims the expected output makes. These are the only things that count.",
            "Score by what fraction of those claims the actual output states correctly, and whether it states anything that contradicts them.",
            "Award a high score when every central claim is present and nothing contradicts the expected output, EVEN IF the wording, ordering, level of detail, formatting or structure differ completely.",
            "IGNORE entirely: differences in phrasing, synonyms, ordering, headings, bullet style, answer length, and any extra correct information not mentioned in the expected output. None of these are errors.",
            "IGNORE entirely: the absence of caveats, hedges or framing sentences that appear in the expected output but carry no factual claim.",
            "Penalise ONLY two things: a statement that contradicts the expected output, and the omission of a central claim the expected output makes.",
        ],
    )
    comprehensiveness = GEval(
        name="comprehensiveness",
        model=judge,
        async_mode=False,
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        evaluation_steps=[
            "The expected output describes the BREADTH a complete answer should cover, not the wording it should use.",
            "Count how many of those distinct aspects the actual output genuinely addresses.",
            "Reward answers that span many distinct aspects, sources or publications; an answer confined to one aspect scores low however well written.",
            "Do not reward length, repetition, or confident tone on its own.",
        ],
    )
    diversity = GEval(
        name="diversity",
        model=judge,
        async_mode=False,
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Judge how many genuinely different perspectives, contrasts or tensions the answer surfaces.",
            "Reward contrasting one source or view against another.",
            "Penalise restating a single idea in several ways.",
        ],
    )
    return {
        "all": [correctness,
                AnswerRelevancyMetric(model=judge, async_mode=False),
                FaithfulnessMetric(model=judge, async_mode=False)],
        "global_only": [comprehensiveness, diversity],
    }


def metric_key(m) -> str:
    name = getattr(m, "name", None)
    if isinstance(name, str) and name:
        return re.sub(r"\W+", "_", name).strip("_").lower()
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(m).__name__.removesuffix("Metric")).lower()


def _score_one(task, samples: int) -> dict:
    """One (question, arm) pair. Builds its own judge and metric objects:
    DeepEval metrics carry per-measure state, so sharing them across threads
    would interleave scores between cases."""
    from deepeval.test_case import LLMTestCase

    r, arm, d, item = task
    judge = AzureGPT5Nano()
    metrics = build_metrics(judge)

    answer = strip_citations(d["answer"])
    tc = LLMTestCase(
        input=r["question"],
        actual_output=answer,
        expected_output=item["reference_answer"],
        retrieval_context=[strip_citations(c) for c in d["retrieval_context"]]
        or ["(no context returned)"],
    )
    applicable = list(metrics["all"])
    if r["tier"] == "global":
        applicable += metrics["global_only"]

    scores = {}
    for m in applicable:
        key, vals = metric_key(m), []
        for _ in range(samples):
            try:
                m.measure(tc)
                vals.append(float(m.score))
            except Exception as e:
                print(f"   [{r['id']}|{arm}] {key} sample failed: {e}", file=sys.stderr)
        if vals:
            scores[key] = round(statistics.median(vals), 4)
            scores[key + "_spread"] = round(max(vals) - min(vals), 4)
            scores[key + "_samples"] = vals
        else:
            scores[key] = None

    print(f"[{r['id']}|{r['tier']}] {arm}: "
          + " ".join(f"{k}={v}" for k, v in scores.items()
                     if not k.endswith(("_spread", "_samples"))), file=sys.stderr)
    return {"id": r["id"], "tier": r["tier"], "category": r["category"],
            "question": r["question"], "arm": arm,
            "citations_stripped": len(d["answer"]) - len(answer),
            "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    rows = json.loads(RESULTS.read_text())
    book = {b["id"]: b for b in json.loads(BOOK.read_text())}

    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["id"], r["arm"]) for r in out}

    tasks = [(r, arm, d, book[r["id"]])
             for r in rows for arm, d in r["arms"].items()
             if (r["id"], arm) not in done]
    print(f"rescoring {len(tasks)} arm-results "
          f"({args.samples} samples, {args.workers} workers)", file=sys.stderr)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_one, t, args.samples) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            with lock:
                out.append(res)
                OUT.write_text(json.dumps(out, indent=2))

    out.sort(key=lambda r: (r["id"], r["arm"]))
    OUT.write_text(json.dumps(out, indent=2))
    report(out)


def report(out):
    arms = ["basic", "local", "global"]
    tiers = ["local", "cross-document", "global", "negative-control"]
    keys = ["correctness", "answer_relevancy", "faithfulness",
            "comprehensiveness", "diversity"]

    print("\n" + "=" * 96)
    print("GraphRAG arms — median of repeated judgements, citations stripped")
    print("=" * 96)
    for tier in tiers:
        sub = [r for r in out if r["tier"] == tier]
        if not sub:
            continue
        print(f"\n{tier}  (n={len({r['id'] for r in sub})})")
        print(f"  {'metric':<22}" + "".join(f"{a:>16}" for a in arms))
        for k in keys:
            cells, any_val = [], False
            for a in arms:
                vals = [r["scores"].get(k) for r in sub if r["arm"] == a]
                vals = [v for v in vals if v is not None]
                sp = [r["scores"].get(k + "_spread", 0) for r in sub if r["arm"] == a]
                sp = [v for v in sp if v is not None]
                if vals:
                    any_val = True
                    cells.append(f"{statistics.mean(vals):>10.3f}±{statistics.mean(sp):<5.2f}")
                else:
                    cells.append(f"{'-':>16}")
            if any_val:
                print(f"  {k:<22}" + "".join(cells))
    print("\n(± is the mean within-case spread across judge samples — the noise floor.)")


if __name__ == "__main__":
    main()
