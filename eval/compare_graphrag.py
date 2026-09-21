"""Official Microsoft GraphRAG: basic (vector control) vs local (entity-anchored) vs global
(map-reduce over community reports), one index and model pair, judged by azure_judge.py.
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
def load_index(root=None):
    root = root or GRAPHRAG_ROOT
    cfg = load_config(root)
    out = root / "output"
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
    # Capped so one arm isn't charged 10x others purely for returning a bigger payload.
    return [s for s in out if s.strip()][:60]


async def run_arm(arm: str, question: str, cfg, art) -> dict:
    t0 = time.time()
    if arm == "basic":
        resp, ctx = await api.basic_search(
            config=cfg, text_units=art["text_units"], query=question)
    elif arm == "basic_k40":
        # Both knobs needed: default max_context_tokens=12000 binds before k, so k alone does nothing.
        cfg.basic_search.k = 40
        cfg.basic_search.max_context_tokens = 50_000
        resp, ctx = await api.basic_search(
            config=cfg, text_units=art["text_units"], query=question)
    elif arm == "global_c0":
        # Root communities only -- Edge et al.'s recommendation, ~15x cheaper than level 2.
        resp, ctx = await api.global_search(
            config=cfg, entities=art["entities"], communities=art["communities"],
            community_reports=art["community_reports"],
            community_level=0, dynamic_community_selection=False,
            response_type=RESPONSE_TYPE, query=question)
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
        # Rate root communities, descend only into children that clear threshold, instead of
        # map-reducing over all 540 (measured: 30 calls/87k tokens/30s vs 43/521k/259s).
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
    # Edge et al.'s own head-to-head criteria for sensemaking questions.
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
    ap.add_argument("--root", help="graphrag root (default: graphrag/)")
    ap.add_argument("--out", help="results file override")
    args = ap.parse_args()
    arms = args.arms.split(",")

    global RESULTS
    if args.out:
        RESULTS = Path(args.out)

    book = json.loads(Path(args.book).read_text())
    cfg, art = load_index(Path(args.root) if args.root else None)
    print(f"index: {len(art['text_units'])} text units, {len(art['entities'])} entities, "
          f"{len(art['relationships'])} relationships, "
          f"{len(art['community_reports'])} community reports", file=sys.stderr)

    judge = None if args.no_score else AzureGPT5Nano()
    metrics = None if args.no_score else build_metrics(judge)

    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    by_id = {r["id"]: r for r in rows}

    for item in book:
        row = by_id.setdefault(item["id"], {
            "id": item["id"], "tier": item["tier"],
            "category": item.get("category", item["tier"]),
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
