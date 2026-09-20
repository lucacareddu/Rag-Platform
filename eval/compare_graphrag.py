"""Official Microsoft GraphRAG: vector baseline vs local search vs global search.

Three arms, one index, one pair of models:

  basic  — GraphRAG's basic_search. Plain vector RAG over the same text units,
           the same embedding model and the same chat model. This is the
           control. Using it instead of our own Qdrant pipeline removes the
           confound that sank earlier comparisons: chunking, embedder and
           generator are identical across arms, so any difference is
           attributable to retrieval strategy alone.
  local  — GraphRAG local search: entity-anchored, mixes entities,
           relationships, community reports and source text.
  global — GraphRAG global search: map-reduce over community reports. This is
           the capability a vector index structurally cannot provide, and the
           only arm expected to win the corpus-level questions.

Scoring is DeepEval with Azure gpt-5-nano as judge (see azure_judge.py).
Non-LLM context metrics are deliberately absent: they compare retrieved text to
gold chunks, and community reports are not chunks, so they would score global
search 0.000 by construction rather than by performance.

Run: .venv-graphrag/bin/python eval/compare_graphrag.py [--arms basic,local,global]
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRAPHRAG_ROOT = ROOT / "graphrag"
sys.path.insert(0, str(Path(__file__).parent))

import graphrag.api as api  # noqa: E402
from graphrag.config.load_config import load_config  # noqa: E402

from azure_judge import AzureGPT5Nano  # noqa: E402

RESULTS = Path(__file__).parent / "results_graphrag.json"
BOOK = Path(__file__).parent / "test_book_v3.json"

COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"


# --------------------------------------------------------------------------
# Index artifacts
# --------------------------------------------------------------------------
def load_index():
    cfg = load_config(GRAPHRAG_ROOT)
    out = GRAPHRAG_ROOT / "output"
    art = {n: pd.read_parquet(out / f"{n}.parquet")
           for n in ["entities", "communities", "community_reports",
                     "text_units", "relationships"]}
    return cfg, art


def _context_strings(context) -> list[str]:
    """Flattens GraphRAG's context payload into plain strings for Faithfulness.

    The payload shape differs per arm (dataframes keyed by 'sources',
    'entities', 'reports', ...), so every frame is rendered rather than
    assuming one schema.
    """
    out = []
    if isinstance(context, dict):
        frames = context.values()
    elif isinstance(context, list):
        frames = context
    else:
        frames = [context]
    for f in frames:
        if isinstance(f, pd.DataFrame):
            if f.empty:
                continue
            for col in ("content", "text", "description", "summary", "title"):
                if col in f.columns:
                    out += [str(v) for v in f[col].dropna().tolist()]
                    break
            else:
                out += [" | ".join(map(str, r)) for r in f.head(50).values]
        elif isinstance(f, (str, bytes)):
            out.append(str(f))
        elif isinstance(f, dict):
            out += _context_strings(f)
    # Faithfulness judges every context string; cap so one arm is not charged
    # 10x the others purely for returning a bigger payload.
    return [s for s in out if s.strip()][:60]


async def run_arm(arm: str, question: str, cfg, art) -> dict:
    t0 = time.time()
    if arm == "basic":
        resp, ctx = await api.basic_search(
            config=cfg, text_units=art["text_units"], query=question)
    elif arm == "local":
        resp, ctx = await api.local_search(
            config=cfg, entities=art["entities"], communities=art["communities"],
            community_reports=art["community_reports"], text_units=art["text_units"],
            relationships=art["relationships"], covariates=None,
            community_level=COMMUNITY_LEVEL, response_type=RESPONSE_TYPE,
            query=question)
    elif arm == "global":
        resp, ctx = await api.global_search(
            config=cfg, entities=art["entities"], communities=art["communities"],
            community_reports=art["community_reports"],
            community_level=COMMUNITY_LEVEL, dynamic_community_selection=False,
            response_type=RESPONSE_TYPE, query=question)
    elif arm == "dynamic":
        # Global search with the hierarchical tree walk: rate the 26 root
        # communities for relevance, descend only into the children of those
        # that clear the threshold, instead of map-reducing over all 540.
        # Measured on one question against the static arm: 30 calls vs 43,
        # 87k input tokens vs 521k, 30s vs 259s.
        resp, ctx = await api.global_search(
            config=cfg, entities=art["entities"], communities=art["communities"],
            community_reports=art["community_reports"],
            community_level=COMMUNITY_LEVEL, dynamic_community_selection=True,
            response_type=RESPONSE_TYPE, query=question)
    else:
        raise ValueError(arm)

    strings = _context_strings(ctx)
    return {
        "answer": resp if isinstance(resp, str) else str(resp),
        "retrieval_context": strings,
        "n_context_items": len(strings),
        "context_chars": sum(len(s) for s in strings),
        "latency": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def build_metrics(judge):
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCaseParams as P

    correctness = GEval(
        name="correctness",
        model=judge,
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        async_mode=False,
        evaluation_steps=[
            "Check whether the facts in the actual output contradict any fact in the expected output.",
            "Heavily penalise contradictions and invented specifics such as numbers, dates or names not supported by the expected output.",
            "Penalise omission of the central elements the expected output identifies.",
            "Do not penalise extra correct detail, different wording, or a different order of presentation.",
        ],
    )
    # The GraphRAG paper's own head-to-head criteria for sensemaking questions.
    comprehensiveness = GEval(
        name="comprehensiveness",
        model=judge,
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        async_mode=False,
        evaluation_steps=[
            "Judge how much of the breadth described in the expected output the actual output covers.",
            "Reward answers that span many distinct aspects, sources or publications rather than treating one in depth.",
            "Penalise answers that address only a narrow slice of what was asked.",
            "Judge coverage only; do not reward length or repetition on its own.",
        ],
    )
    diversity = GEval(
        name="diversity",
        model=judge,
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
        async_mode=False,
        evaluation_steps=[
            "Judge how varied and rich the perspectives in the actual output are.",
            "Reward answers that surface different angles, contrasts or tensions rather than restating one point.",
            "Penalise repetitive answers that paraphrase a single idea several times.",
        ],
    )
    return {
        "all": [correctness,
                AnswerRelevancyMetric(model=judge, async_mode=False),
                FaithfulnessMetric(model=judge, async_mode=False)],
        "global_only": [comprehensiveness, diversity],
    }


def _metric_key(m) -> str:
    """GEval carries a `name`; the built-in metrics carry nothing and fall back
    to their class name. Without this both built-ins would land under keys like
    `answerrelevancymetric`, which the report does not look for — they would
    print as blanks rather than as failures."""
    name = getattr(m, "name", None)
    if isinstance(name, str) and name:
        return re.sub(r"\W+", "_", name).strip("_").lower()
    cls = type(m).__name__.removesuffix("Metric")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", cls).lower()


def score(item, result, metrics, judge) -> dict:
    from deepeval.test_case import LLMTestCase

    tc = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["reference_answer"],
        retrieval_context=result["retrieval_context"] or ["(no context returned)"],
    )
    applicable = list(metrics["all"])
    if item["tier"] == "global":
        applicable += metrics["global_only"]

    out = {}
    for m in applicable:
        name = _metric_key(m)
        try:
            m.measure(tc)
            out[name] = round(float(m.score), 4)
        except Exception as e:
            print(f"      metric {name} failed: {e}", file=sys.stderr)
            out[name] = None
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="basic,local,global,dynamic")
    ap.add_argument("--book", default=str(BOOK))
    ap.add_argument("--no-score", action="store_true",
                    help="generate answers only, skip judging")
    args = ap.parse_args()
    arms = args.arms.split(",")

    book = json.loads(Path(args.book).read_text())
    cfg, art = load_index()
    print(f"index: {len(art['text_units'])} text units, {len(art['entities'])} entities, "
          f"{len(art['relationships'])} relationships, "
          f"{len(art['community_reports'])} community reports", file=sys.stderr)

    judge = None if args.no_score else AzureGPT5Nano()
    metrics = None if args.no_score else build_metrics(judge)

    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    by_id = {r["id"]: r for r in rows}

    for item in book:
        row = by_id.setdefault(item["id"], {
            "id": item["id"], "tier": item["tier"], "category": item["category"],
            "question": item["question"], "arms": {},
        })
        for arm in arms:
            if arm in row["arms"] and row["arms"][arm].get("scores"):
                continue
            print(f"[{item['id']}|{item['tier']}] {arm}: {item['question'][:60]}...",
                  file=sys.stderr)
            if arm not in row["arms"]:
                row["arms"][arm] = asyncio.run(run_arm(arm, item["question"], cfg, art))
            if not args.no_score:
                row["arms"][arm]["scores"] = score(item, row["arms"][arm], metrics, judge)
            rows = [by_id[k] for k in sorted(by_id)]
            RESULTS.write_text(json.dumps(rows, indent=2))

    rows = [by_id[k] for k in sorted(by_id)]
    RESULTS.write_text(json.dumps(rows, indent=2))
    if not args.no_score:
        report(rows, arms)


def report(rows, arms):
    tiers = ["local", "cross-document", "global", "negative-control"]
    metric_names = ["correctness", "answer_relevancy", "faithfulness",
                    "comprehensiveness", "diversity"]

    print("\n" + "=" * 92)
    print("GraphRAG arms by question tier (Azure gpt-5-nano judge)")
    print("=" * 92)
    for tier in tiers:
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        print(f"\n{tier}  (n={len(sub)})")
        print(f"  {'metric':<22}" + "".join(f"{a:>14}" for a in arms))
        for m in metric_names:
            cells, any_val = [], False
            for a in arms:
                vals = [r["arms"][a]["scores"].get(m) for r in sub
                        if a in r["arms"] and r["arms"][a].get("scores")]
                vals = [v for v in vals if v is not None]
                if vals:
                    any_val = True
                    cells.append(f"{sum(vals)/len(vals):>14.3f}")
                else:
                    cells.append(f"{'-':>14}")
            if any_val:
                print(f"  {m:<22}" + "".join(cells))

    print("\n" + "-" * 92)
    print("operational")
    print("-" * 92)
    print(f"  {'':<22}" + "".join(f"{a:>14}" for a in arms))
    for label, key in [("latency (s)", "latency"), ("context items", "n_context_items"),
                       ("context chars", "context_chars")]:
        cells = []
        for a in arms:
            vals = [r["arms"][a][key] for r in rows if a in r["arms"]]
            cells.append(f"{sum(vals)/len(vals):>14.1f}" if vals else f"{'-':>14}")
        print(f"  {label:<22}" + "".join(cells))


if __name__ == "__main__":
    main()
