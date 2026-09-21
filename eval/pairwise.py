"""Head-to-head comparison of GraphRAG arms, after absolute scoring failed.

Absolute GEval scoring could not separate these arms. Measured on the corrected
rubric with 5 samples per metric, the judge's own spread on an unchanged test
case averaged 0.24 while the gap between arms averaged 0.10 — the differences
sat inside the noise. The variance comes from gpt-5's forced temperature=1 and
survives more reasoning effort, so sampling harder was not going to fix it.

Pairwise judging asks a strictly easier question. Instead of calibrating a
number, the judge reads two answers and says which is better. That removes the
need for a stable internal scale, and it is the protocol the GraphRAG paper
itself uses for sensemaking questions.

Controls:
- Position randomisation. LLM judges favour whichever answer is shown first,
  so each pair is judged in both orders and the orders are pooled. A win only
  counts as a win if it survives the swap; systematic position bias shows up
  as a high tie/disagreement rate rather than silently inflating one arm.
- Citations stripped, as in rescore.py, so an arm is not judged on how much
  provenance markup it emits.
- Blind labels. Answers are presented as "Answer A"/"Answer B" with no mention
  of which retrieval strategy produced them.

Criteria follow the paper: comprehensiveness, diversity, empowerment and
directness.

Run: .venv-graphrag/bin/python eval/pairwise.py [--repeats 2] [--workers 6]
"""
import argparse
import itertools
import json
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from azure_judge import AzureGPT5Nano  # noqa: E402
from rescore import strip_citations  # noqa: E402

RESULTS = Path(__file__).parent / "results_graphrag.json"
OUT = Path(__file__).parent / "results_pairwise.json"
BOOK = Path(__file__).parent / "test_book_v3.json"

ARMS = ["basic", "local", "global", "dynamic"]

CRITERIA = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}

PROMPT = """You are evaluating two answers to the same question about a corpus of NIST cybersecurity publications.

Question: {question}

--- Answer A ---
{a}

--- Answer B ---
{b}

Assess which answer is better on each criterion below.

{criteria}

For each criterion pick "A", "B", or "TIE". Use TIE only when the two are genuinely comparable.
Judge only the answer text. Ignore differences in formatting, length for its own sake, and writing style.
Return JSON: {{"comprehensiveness": "...", "diversity": "...", "empowerment": "...", "directness": "...", "reason": "<one sentence>"}}"""


class Verdict(BaseModel):
    comprehensiveness: str
    diversity: str
    empowerment: str
    directness: str
    reason: str


def _judge_once(judge, question: str, ans_a: str, ans_b: str) -> Verdict:
    criteria = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    return judge.generate(
        PROMPT.format(question=question, a=ans_a, b=ans_b, criteria=criteria), Verdict)


def _compare(task) -> dict:
    """One (question, arm-pair, repeat). Judged in both orders."""
    r, x, y, rep = task
    judge = AzureGPT5Nano()
    ax = strip_citations(r["arms"][x]["answer"])
    ay = strip_citations(r["arms"][y]["answer"])

    out = {"id": r["id"], "tier": r["tier"], "question": r["question"],
           "pair": [x, y], "repeat": rep, "verdicts": {}}

    # Order 1: x shown as A. Order 2: y shown as A. Pooling the two cancels
    # the judge's preference for whichever answer it reads first.
    v1 = _judge_once(judge, r["question"], ax, ay)
    v2 = _judge_once(judge, r["question"], ay, ax)

    for crit in CRITERIA:
        p1 = getattr(v1, crit, "TIE").strip().upper()
        p2 = getattr(v2, crit, "TIE").strip().upper()
        # Translate each order's A/B answer back to the arm it names.
        w1 = x if p1 == "A" else (y if p1 == "B" else "TIE")
        w2 = y if p2 == "A" else (x if p2 == "B" else "TIE")
        if w1 == w2:
            verdict = w1                    # consistent across both orders
        elif "TIE" in (w1, w2):
            verdict = "TIE"
        else:
            verdict = "INCONSISTENT"        # order flipped the winner
        out["verdicts"][crit] = {"order1": w1, "order2": w2, "verdict": verdict}
    out["reason"] = v1.reason
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    # The budget controls (basic_k40, global_c0) live in a second results file
    # and need the same judging protocol, so the inputs are parameterised
    # rather than the script copied. --merge unions the arms of both files on
    # question id, which is what makes a cross-file pair judgeable at all.
    ap.add_argument("--merge", help="second results file to union arms from")
    ap.add_argument("--arms", help="comma-separated arms to compare")
    ap.add_argument("--out", help="override output path")
    args = ap.parse_args()

    global ARMS, OUT
    if args.arms:
        ARMS = args.arms.split(",")
    if args.out:
        OUT = Path(args.out)

    rows = json.loads(RESULTS.read_text())
    if args.merge:
        extra = {r["id"]: r for r in json.loads(Path(args.merge).read_text())}
        for r in rows:
            r["arms"].update(extra.get(r["id"], {}).get("arms", {}))

    rows = [r for r in rows if all(a in r["arms"] for a in ARMS)]
    print(f"questions with all {len(ARMS)} arms: {len(rows)}", file=sys.stderr)

    out = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {(r["id"], tuple(r["pair"]), r["repeat"]) for r in out}

    tasks = [(r, x, y, rep)
             for r in rows
             for x, y in itertools.combinations(ARMS, 2)
             for rep in range(args.repeats)
             if (r["id"], (x, y), rep) not in done]
    random.shuffle(tasks)
    print(f"comparisons to run: {len(tasks)} "
          f"(each = 2 LLM calls, both orders)", file=sys.stderr)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_compare, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                res = f.result()
            except Exception as e:
                print(f"  comparison failed: {e}", file=sys.stderr)
                continue
            with lock:
                out.append(res)
                OUT.write_text(json.dumps(out, indent=2))
            if i % 10 == 0:
                print(f"  {i}/{len(tasks)}", file=sys.stderr)

    OUT.write_text(json.dumps(out, indent=2))
    report(out)


def report(out):
    tiers = ["local", "cross-document", "global", "negative-control"]

    print("\n" + "=" * 96)
    print("Head-to-head win rates (both orders judged; INCONSISTENT = order flipped the winner)")
    print("=" * 96)

    for tier in [None] + tiers:
        sub = [r for r in out if tier is None or r["tier"] == tier]
        if not sub:
            continue
        label = "ALL TIERS" if tier is None else tier
        n_q = len({r["id"] for r in sub})
        print(f"\n{label}  (questions={n_q}, comparisons={len(sub)})")
        for x, y in itertools.combinations(ARMS, 2):
            pair_rows = [r for r in sub if r["pair"] == [x, y]]
            if not pair_rows:
                continue
            print(f"  {x} vs {y}")
            for crit in CRITERIA:
                vs = [r["verdicts"][crit]["verdict"] for r in pair_rows]
                wx, wy = vs.count(x), vs.count(y)
                tie, inc = vs.count("TIE"), vs.count("INCONSISTENT")
                n = len(vs)
                print(f"    {crit:<20} {x}={wx:<3}{y}={wy:<3}tie={tie:<3}inconsistent={inc:<3}"
                      f"  -> {x} wins {100*wx/n:.0f}%, {y} wins {100*wy/n:.0f}%")

    inc_all = [r["verdicts"][c]["verdict"] for r in out for c in CRITERIA]
    print(f"\norder-inconsistency rate overall: "
          f"{100*inc_all.count('INCONSISTENT')/max(len(inc_all),1):.1f}% "
          f"(high values mean the judge is driven by position, not content)")


if __name__ == "__main__":
    main()
