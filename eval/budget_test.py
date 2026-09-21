"""Two controls the original comparison was missing.

Both exist because the first round compared arms that were never given equal
resources, so "the graph wins sensemaking" was confounded with "the graph arm
read far more of the corpus".

  global_c0   Global search over ROOT communities only (level 0, 26 reports)
              instead of level <=2 (540 reports). This is the configuration
              Edge et al. actually recommend: their Table 2 puts root-level
              summaries at 2.3-2.6% of full-corpus token cost while retaining a
              72%/62% win rate over vector RAG on comprehensiveness/diversity.
              Everything reported so far ran at level 2, which their own table
              places at 55-57% of full cost -- the second most expensive
              setting available. The measured 693k prompt tokens per question
              was therefore substantially a configuration choice, not an
              inherent property of the method.

  basic_k40   Plain vector RAG with the context cap lifted. The default is
              k=10 AND max_context_tokens=12000; with 1200-token chunks the
              TOKEN CAP binds first, which is why the basic arm returned ~9
              items regardless of k. Raising k alone changes nothing. This arm
              raises both, so vector RAG gets a context budget in the same
              league as the graph arms.

The question each answers:

  If basic_k40 closes the sensemaking gap, the graph never contributed
  structure -- it contributed context volume, and the fix is a bigger top_k.
  If global_c0 matches global at 1/20th the reports, the cost objection to
  global search was an artefact of my configuration rather than the method.

Run: .venv-graphrag/bin/python eval/budget_test.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRAPHRAG_ROOT = ROOT / "graphrag"
sys.path.insert(0, str(Path(__file__).parent))

import graphrag.api as api  # noqa: E402
from graphrag.config.load_config import load_config  # noqa: E402

from compare_graphrag import _context_strings  # noqa: E402

BOOK = Path(__file__).parent / "test_book_v3.json"
OUT = Path(__file__).parent / "results_budget.json"

NEW_ARMS = ["basic_k40", "global_c0"]
RESPONSE_TYPE = "Multiple Paragraphs"

# Roughly global's own budget: 40 chunks x ~1200 tokens. The point is parity of
# context volume, not a round number.
BASIC_K = 40
BASIC_MAX_CONTEXT_TOKENS = 50_000


async def run_arm(arm: str, question: str, cfg, art) -> dict:
    t0 = time.time()
    if arm == "basic_k40":
        cfg.basic_search.k = BASIC_K
        cfg.basic_search.max_context_tokens = BASIC_MAX_CONTEXT_TOKENS
        resp, ctx = await api.basic_search(
            config=cfg, text_units=art["text_units"], query=question)
    elif arm == "global_c0":
        resp, ctx = await api.global_search(
            config=cfg, entities=art["entities"], communities=art["communities"],
            community_reports=art["community_reports"],
            community_level=0, dynamic_community_selection=False,
            response_type=RESPONSE_TYPE, query=question)
    else:
        raise ValueError(arm)

    strings = _context_strings(ctx)
    return {"answer": resp if isinstance(resp, str) else str(resp),
            "retrieval_context": strings,
            "n_context_items": len(strings),
            "context_chars": sum(len(s) for s in strings),
            "latency": round(time.time() - t0, 2)}


async def main():
    cfg = load_config(GRAPHRAG_ROOT)
    out_dir = GRAPHRAG_ROOT / "output"
    art = {n: pd.read_parquet(out_dir / f"{n}.parquet")
           for n in ["entities", "communities", "community_reports",
                     "text_units", "relationships"]}

    book = json.loads(BOOK.read_text())
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    by_id = {r["id"]: r for r in rows}

    for item in book:
        row = by_id.setdefault(item["id"], {"id": item["id"], "tier": item["tier"],
                                            "question": item["question"], "arms": {}})
        for arm in NEW_ARMS:
            if arm in row["arms"]:
                continue
            try:
                row["arms"][arm] = await run_arm(arm, item["question"], cfg, art)
            except Exception as e:
                print(f"  [{item['id']}|{arm}] FAILED: {str(e)[:120]}", file=sys.stderr)
                continue
            d = row["arms"][arm]
            print(f"[{item['id']:>2}|{item['tier']:<16}] {arm:<10} "
                  f"{d['latency']:>7.1f}s  ctx={d['n_context_items']:>3} "
                  f"({d['context_chars']:>7,} chars)  ans={len(d['answer']):>6,}",
                  file=sys.stderr)
            rows = [by_id[k] for k in sorted(by_id)]
            OUT.write_text(json.dumps(rows, indent=2))

    rows = [by_id[k] for k in sorted(by_id)]
    OUT.write_text(json.dumps(rows, indent=2))
    summarise(rows)


def summarise(rows):
    import statistics
    print("\n" + "=" * 78)
    print("NEW ARMS — latency and context volume")
    print("=" * 78)
    for tier in ["local", "cross-document", "global", "negative-control"]:
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        print(f"\n{tier} (n={len(sub)})")
        for arm in NEW_ARMS:
            d = [r["arms"][arm] for r in sub if arm in r["arms"]]
            if not d:
                continue
            print(f"  {arm:<12} lat={statistics.mean(x['latency'] for x in d):>7.1f}s"
                  f"  ctx_items={statistics.mean(x['n_context_items'] for x in d):>5.1f}"
                  f"  ctx_chars={statistics.mean(x['context_chars'] for x in d):>9,.0f}"
                  f"  ans_chars={statistics.mean(len(x['answer']) for x in d):>8,.0f}")


if __name__ == "__main__":
    asyncio.run(main())
