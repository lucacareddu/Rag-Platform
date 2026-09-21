"""Isolated latency probe for global search: where do the 339 seconds go?

The 338.7s mean reported for the global arm is wall time from an eval loop, and
wall time conflates three different things:

  1. genuine model latency        — how long Azure takes to answer a map call
  2. the client-side rate limiter — fnllm books tokens_per_minute BEFORE
     issuing a call, so a large map fan-out queues against its own budget
  3. structural serialisation     — the reduce step cannot start until every
     map call has returned, so the critical path is (slowest map) + reduce

Only (1) and (3) are properties of the method. (2) is a property of THIS
deployment's quota and would vanish on a higher tier, so quoting it as "global
search takes 5.6 minutes" overstates the method's cost.

The probe measures all three by wrapping the chat model and recording an
interval per call. Nothing else runs concurrently; queries are issued one at a
time, exactly as the eval loop did.

  wall            end-to-end
  busy            union of all call intervals — time at least one call in flight
  sum_durations   total API time; sum/wall is the achieved parallelism
  critical path   max map duration + reduce duration, the floor the method
                  cannot go below however much quota you throw at it
  stall           wall - busy, time NO call was in flight: rate-limiter queueing

Run twice, once as configured and once with the limiter lifted, and the gap
between them is the quota tax:

  .venv-graphrag/bin/python eval/latency_probe.py                 # as configured
  .venv-graphrag/bin/python eval/latency_probe.py --unthrottled   # limiter lifted

--unthrottled raises rpm/tpm only. Model parameters are untouched: they are part
of GraphRAG's cache key, and changing one silently invalidates the index cache
(learned the expensive way during indexing).
"""
import argparse
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
from graphrag.language_model.manager import ModelManager  # noqa: E402

OUT = Path(__file__).parent / "results_latency_probe.json"

# Deliberately NOT from test_book_v3: those questions have been run before, and
# a cache hit would report a latency of zero and prove nothing.
PROBE_QUESTION = (
    "Across these publications, how do the documents differ in how much they "
    "rely on quantitative versus qualitative assessment?"
)

CALLS: list[dict] = []


def instrument():
    """Wrap the chat model so every call records an interval.

    Patching the manager rather than the search engine keeps this independent of
    how the engine is assembled, and catches the dynamic-selection rating calls
    too, which are issued from a different code path than the map phase.
    """
    real = ModelManager.get_or_create_chat_model

    def patched(self, name, model_type, config=None, **kw):
        model = real(self, name=name, model_type=model_type, config=config, **kw)
        if getattr(model, "_probed", False):
            return model
        model._probed = True

        for meth in ("achat", "achat_stream"):
            original = getattr(model, meth, None)
            if original is None:
                continue
            setattr(model, meth, _wrap(original, meth, model))
        return model

    ModelManager.get_or_create_chat_model = patched


def _wrap(original, meth, model):
    if meth == "achat_stream":
        async def wrapper(*a, **kw):
            rec = _open(meth, a, kw)
            try:
                async for chunk in original(*a, **kw):
                    yield chunk
            finally:
                _close(rec)
        return wrapper

    async def wrapper(*a, **kw):
        rec = _open(meth, a, kw)
        try:
            return await original(*a, **kw)
        finally:
            _close(rec)
    return wrapper


def _open(meth, a, kw):
    prompt = kw.get("prompt") or (a[0] if a else "")
    history = kw.get("history") or []
    chars = len(str(prompt)) + sum(len(str(h.get("content", ""))) for h in history)
    rec = {"method": meth, "start": time.perf_counter(), "end": None,
           "prompt_chars": chars}
    CALLS.append(rec)
    return rec


def _close(rec):
    rec["end"] = time.perf_counter()
    rec["duration"] = round(rec["end"] - rec["start"], 3)


def _union(intervals) -> float:
    """Total time at least one call was in flight."""
    if not intervals:
        return 0.0
    merged, (cs, ce) = [], intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return sum(e - s for s, e in merged)


def _peak_concurrency(intervals) -> int:
    events = [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals]
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def analyse(wall: float, label: str) -> dict:
    done = [c for c in CALLS if c["end"] is not None]
    intervals = sorted((c["start"], c["end"]) for c in done)
    durations = sorted(c["duration"] for c in done)
    busy = _union(intervals)

    # The reduce step is the final call and cannot overlap the map phase; every
    # earlier call is map (or dynamic-selection rating).
    reduce_d = durations and done[-1]["duration"] or 0.0
    map_calls = done[:-1]
    slowest_map = max((c["duration"] for c in map_calls), default=0.0)

    r = {
        "label": label,
        "wall": round(wall, 2),
        "n_calls": len(done),
        "busy": round(busy, 2),
        "stall": round(wall - busy, 2),
        "stall_pct": round(100 * (wall - busy) / wall, 1) if wall else 0.0,
        "sum_durations": round(sum(durations), 2),
        "achieved_parallelism": round(sum(durations) / wall, 2) if wall else 0.0,
        "peak_concurrency": _peak_concurrency(intervals),
        "call_p50": round(durations[len(durations) // 2], 2) if durations else 0.0,
        "call_max": round(durations[-1], 2) if durations else 0.0,
        "slowest_map": round(slowest_map, 2),
        "reduce": round(reduce_d, 2),
        "critical_path": round(slowest_map + reduce_d, 2),
        "prompt_chars_total": sum(c["prompt_chars"] for c in done),
    }
    r["quota_tax"] = round(r["wall"] - r["critical_path"], 2)
    return r


def report(r: dict):
    print(f"\n{'=' * 70}\nGLOBAL SEARCH — {r['label']}\n{'=' * 70}")
    print(f"  wall clock                  {r['wall']:>9.2f} s")
    print(f"  LLM calls                   {r['n_calls']:>9}")
    print(f"  peak concurrency            {r['peak_concurrency']:>9}")
    print(f"  achieved parallelism        {r['achieved_parallelism']:>9.2f} x")
    print()
    print(f"  time >=1 call in flight     {r['busy']:>9.2f} s")
    print(f"  time NO call in flight      {r['stall']:>9.2f} s   "
          f"({r['stall_pct']}% of wall)  <- rate-limiter queueing")
    print(f"  sum of all call durations   {r['sum_durations']:>9.2f} s")
    print()
    print(f"  median call                 {r['call_p50']:>9.2f} s")
    print(f"  slowest map call            {r['slowest_map']:>9.2f} s")
    print(f"  reduce call                 {r['reduce']:>9.2f} s")
    print(f"  critical path (map+reduce)  {r['critical_path']:>9.2f} s   "
          f"<- irreducible floor")
    print(f"  everything else             {r['quota_tax']:>9.2f} s   "
          f"<- quota/scheduling overhead")
    print(f"  total prompt chars          {r['prompt_chars_total']:>9,}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unthrottled", action="store_true",
                    help="raise rpm/tpm to isolate the quota tax")
    ap.add_argument("--dynamic", action="store_true",
                    help="probe the dynamic-selection arm instead")
    ap.add_argument("--level", type=int, default=2,
                    help="community level: 0 = root (26 reports), 2 = 540")
    ap.add_argument("--question", default=PROBE_QUESTION)
    args = ap.parse_args()

    instrument()
    cfg = load_config(GRAPHRAG_ROOT)

    if args.unthrottled:
        for m in cfg.models.values():
            # Only the limiter. Touching model params would invalidate the cache.
            if hasattr(m, "requests_per_minute"):
                m.requests_per_minute = 100000
            if hasattr(m, "tokens_per_minute"):
                m.tokens_per_minute = 100000000
        print("limiter lifted: rpm=100000 tpm=100000000", file=sys.stderr)

    out = GRAPHRAG_ROOT / "output"
    art = {n: pd.read_parquet(out / f"{n}.parquet")
           for n in ["entities", "communities", "community_reports"]}

    label = ("dynamic" if args.dynamic else f"global c{args.level}") + \
            (" / limiter lifted" if args.unthrottled else " / as configured")
    print(f"probing: {label}", file=sys.stderr)
    print(f"question: {args.question}", file=sys.stderr)

    CALLS.clear()
    t0 = time.perf_counter()
    resp, _ = await api.global_search(
        config=cfg, entities=art["entities"], communities=art["communities"],
        community_reports=art["community_reports"], community_level=args.level,
        dynamic_community_selection=args.dynamic,
        response_type="Multiple Paragraphs", query=args.question)
    wall = time.perf_counter() - t0

    r = analyse(wall, label)
    r["answer_chars"] = len(str(resp))
    r["question"] = args.question
    report(r)

    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    prev.append(r)
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"\nappended to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
