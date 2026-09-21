"""Hallucination check, two layers: deterministic grounding (checkable atoms vs source corpus,
a floor not a clean bill) and claim-level LLM verification (flags for human review, not a verdict).
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from azure_judge import AzureGPT5Nano  # noqa: E402

ARMS = ["basic", "basic_k40", "global_c0"]

# Specific enough that absence from the corpus is evidence, not noise.
PATTERNS = {
    "cve": r"\bCVE-\d{4}-\d{4,7}\b",
    "year": r"\b(?:19|20)\d{2}\b",
    "money": r"\$\s?\d[\d,.]*\s?(?:billion|million|thousand|bn|m)?\b",
    "percent": r"\b\d{1,3}(?:\.\d+)?\s?%",
    "quantity": r"\b\d{1,3}(?:,\d{3})+\b",
    "identifier": r"\bMS\d{2}-\d{3}\b|\bSP\s?800-\d+\b",
}

# Capitalised multiword names, minus sentence-initial noise and markup.
PROPER = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:[-'][A-Z]?[a-z]+)?)"
                    r"(?:\s+(?:of|the|for|and|de)?\s*[A-Z][a-z0-9]+){1,3}\b")

_CITATION = re.compile(r"\[\s*(?:Data|Sources?|Entities|Relationships|Reports)\s*:[^\]]*\]",
                       re.I)

# Generic capitalised phrases that appear in any prose and are not claims.
# First words of headings that structured answers emit; not factual claims.
_HEADING_WORDS = {"overview", "summary", "bottom", "key", "implications",
                  "what", "why", "how", "conclusion", "takeaways", "analysis",
                  "background", "findings", "recommendations", "across",
                  "description", "governance", "threat", "breach", "use",
                  "while", "note", "data", "sources"}

_GENERIC = {
    "the united states", "united states", "the corpus", "the documents",
    "the report", "the reports", "in summary", "overall", "this suggests",
    "key points", "the data", "the analysis", "for example",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def corpus_text(root: Path) -> str:
    tu = pd.read_parquet(root / "output/text_units.parquet")
    return norm(" ".join(tu["text"].astype(str)))


CITE = re.compile(r"Reports?\s*\(([^)]*)\)", re.I)


def citations(answer: str) -> list:
    """Community-report ids the answer claims to cite."""
    out = []
    for m in CITE.finditer(answer):
        out += [int(x) for x in re.findall(r"\b\d+\b", m.group(1))]
    return out


def atoms(answer: str) -> dict:
    text = _CITATION.sub(" ", answer)
    found = {}
    for kind, pat in PATTERNS.items():
        found[kind] = sorted({m.group(0).strip() for m in re.finditer(pat, text)})
    # Split conjunctions ("Israel and Iran") so two grounded entities aren't scored as one
    # ungrounded atom; drop invented section headings, which are formatting, not claims.
    names = set()
    for m in PROPER.finditer(text):
        for part in re.split(r"\s+(?:and|or|vs\.?|versus)\s+", m.group(0)):
            n = norm(part)
            if n in _GENERIC or len(n) <= 6:
                continue
            if n.split()[0] in _HEADING_WORDS:
                continue
            names.add(n)
    found["proper_noun"] = sorted(names)
    return found


def grounded(atom: str, kind: str, corpus: str) -> bool:
    a = norm(atom)
    if kind == "money":
        # "$4.4 million" may appear as "4.4 million" or "$4.4m"; check the digits.
        digits = re.sub(r"[^\d.]", "", a)
        return bool(digits) and digits in corpus
    if kind == "quantity":
        # 18,000 may be written 18000.
        return a in corpus or a.replace(",", "") in corpus
    return a in corpus


def layer1(rows, corpus: str, tiers, valid_reports=None) -> list:
    out = []
    for row in rows:
        if row["tier"] not in tiers:
            continue
        for arm in ARMS:
            if arm not in row["arms"]:
                continue
            ans = row["arms"][arm]["answer"]
            found = atoms(ans)
            ungrounded, total = {}, 0
            for kind, items in found.items():
                bad = [a for a in items if not grounded(a, kind, corpus)]
                total += len(items)
                if bad:
                    ungrounded[kind] = bad
            n_bad = sum(len(v) for v in ungrounded.values())
            cites = citations(ans)
            bad_cites = ([c for c in cites if c not in valid_reports]
                         if valid_reports is not None else [])
            out.append({
                "id": row["id"], "tier": row["tier"], "arm": arm,
                "answer_chars": len(ans),
                "atoms_total": total, "atoms_ungrounded": n_bad,
                "ungrounded_rate": round(n_bad / total, 3) if total else 0.0,
                "ungrounded": ungrounded,
                "citations": len(cites), "citations_nonexistent": len(bad_cites),
                "bad_citation_ids": sorted(set(bad_cites))[:20],
            })
    return out


CLAIM_PROMPT = """Split the following answer into atomic factual claims.

A claim is one verifiable assertion. Ignore hedging, structure and commentary
("this suggests", "in summary", headings). Keep each claim self-contained:
resolve pronouns so it can be checked on its own.

Return ONLY a JSON array of strings, at most 15 claims, most specific first.

ANSWER:
{answer}

JSON array:"""

VERIFY_PROMPT = """You are checking claims against source material.

For each claim, decide using ONLY the SOURCE TEXT:
  SUPPORTED    - the source states this, or directly implies it
  CONTRADICTED - the source states something incompatible
  UNSUPPORTED  - the source does not address it either way

Do not use outside knowledge. A claim that is true in the real world but absent
from the source is UNSUPPORTED.

SOURCE TEXT:
{context}

CLAIMS:
{claims}

Return ONLY a JSON array of objects, one per claim, in order:
[{{"claim_index": 0, "verdict": "SUPPORTED|CONTRADICTED|UNSUPPORTED", "why": "<12 words"}}]

JSON array:"""


def _json_array(raw: str):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def retrieve(query: str, tu: pd.DataFrame, k: int = 6) -> str:
    """Lexical top-k over the corpus. Deliberately not the arm's own retrieved
    context: checking an answer against the text that produced it would only
    measure internal consistency, not truth. The whole corpus is the standard."""
    words = {w for w in re.findall(r"[a-z0-9]{4,}", query.lower())}
    scored = []
    for r in tu.itertuples():
        t = str(r.text).lower()
        scored.append((sum(1 for w in words if w in t), r.text))
    scored.sort(key=lambda x: -x[0])
    return "\n\n---\n\n".join(t[:2500] for _, t in scored[:k])


def layer2(rows, root: Path, tiers, workers: int) -> list:
    judge = AzureGPT5Nano()
    tu = pd.read_parquet(root / "output/text_units.parquet")

    tasks = [(row, arm) for row in rows if row["tier"] in tiers
             for arm in ARMS if arm in row["arms"]]

    def one(task):
        row, arm = task
        ans = _CITATION.sub(" ", row["arms"][arm]["answer"])
        raw = judge.generate(CLAIM_PROMPT.format(answer=ans[:14000]))
        claims = _json_array(raw) or []
        claims = [c for c in claims if isinstance(c, str)][:15]
        if not claims:
            return {"id": row["id"], "tier": row["tier"], "arm": arm,
                    "claims": [], "error": "no claims extracted"}

        ctx = retrieve(row["question"] + " " + " ".join(claims), tu)
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
        raw = judge.generate(VERIFY_PROMPT.format(context=ctx[:60000],
                                                  claims=numbered))
        verdicts = _json_array(raw) or []
        by_idx = {v.get("claim_index"): v for v in verdicts
                  if isinstance(v, dict)}
        merged = []
        for i, c in enumerate(claims):
            v = by_idx.get(i, {})
            merged.append({"claim": c,
                           "verdict": str(v.get("verdict", "UNKNOWN")).upper(),
                           "why": v.get("why", "")})
        counts = {}
        for m in merged:
            counts[m["verdict"]] = counts.get(m["verdict"], 0) + 1
        print(f"[{row['id']:>2}|{arm:<10}] {len(merged)} claims  {counts}",
              file=sys.stderr)
        return {"id": row["id"], "tier": row["tier"], "arm": arm,
                "claims": merged, "counts": counts}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, tasks))


def report(l1, l2):
    print("\n" + "=" * 78)
    print("LAYER 1 — DETERMINISTIC GROUNDING (no LLM; floor on hallucination)")
    print("=" * 78)
    print(f"  {'arm':<12}{'answers':>9}{'atoms':>8}{'ungrounded':>12}{'rate':>9}"
          f"{'chars/ans':>11}")
    for arm in ARMS:
        sub = [r for r in l1 if r["arm"] == arm]
        if not sub:
            continue
        tot = sum(r["atoms_total"] for r in sub)
        bad = sum(r["atoms_ungrounded"] for r in sub)
        ch = sum(r["answer_chars"] for r in sub) / len(sub)
        print(f"  {arm:<12}{len(sub):>9}{tot:>8}{bad:>12}"
              f"{(bad/tot if tot else 0):>9.3f}{ch:>11,.0f}")

    print("\n  CITATION VALIDITY — do cited report ids exist in the index?")
    print(f"  {'arm':<12}{'citations':>11}{'nonexistent':>13}{'rate':>9}")
    for arm in ARMS:
        sub = [r for r in l1 if r["arm"] == arm]
        c = sum(r.get("citations", 0) for r in sub)
        b = sum(r.get("citations_nonexistent", 0) for r in sub)
        if c:
            print(f"  {arm:<12}{c:>11}{b:>13}{b/c:>9.3f}")

    print("\n  ungrounded atoms by arm (first 8):")
    for arm in ARMS:
        items = []
        for r in l1:
            if r["arm"] != arm:
                continue
            for kind, vals in r["ungrounded"].items():
                items += [f"q{r['id']}:{kind}:{v}" for v in vals]
        print(f"    {arm:<12}{len(items):>4} | " + "; ".join(items[:8]))

    if not l2:
        return
    print("\n" + "=" * 78)
    print("LAYER 2 — CLAIM VERIFICATION vs corpus (flags for human review)")
    print("=" * 78)
    print(f"  {'arm':<12}{'claims':>8}{'supported':>11}{'unsupported':>13}"
          f"{'contradicted':>14}{'support rate':>14}")
    for arm in ARMS:
        sub = [r for r in l2 if r["arm"] == arm]
        c = {}
        for r in sub:
            for k, v in (r.get("counts") or {}).items():
                c[k] = c.get(k, 0) + v
        tot = sum(c.values())
        if not tot:
            continue
        sup = c.get("SUPPORTED", 0)
        print(f"  {arm:<12}{tot:>8}{sup:>11}{c.get('UNSUPPORTED', 0):>13}"
              f"{c.get('CONTRADICTED', 0):>14}{sup/tot:>14.3f}")

    print("\n  CONTRADICTED claims (highest priority for human review):")
    n = 0
    for r in l2:
        for m in r.get("claims", []):
            if m["verdict"] == "CONTRADICTED":
                n += 1
                print(f"    [q{r['id']}|{r['arm']}] {m['claim'][:100]}")
                print(f"        why: {m['why'][:80]}")
    if not n:
        print("    none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="graphrag_incidents")
    ap.add_argument("--results", default="eval/results_incidents.json")
    ap.add_argument("--out", default="eval/results_hallucination.json")
    ap.add_argument("--tiers", default="cross-document,global",
                    help="where long answers live, and fabrication with them")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--layer1-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    tiers = set(args.tiers.split(","))
    rows = json.loads(Path(args.results).read_text())

    print(f"corpus: {root} | tiers: {sorted(tiers)}", file=sys.stderr)
    try:
        valid = set(pd.read_parquet(root / "output/community_reports.parquet")
                    ["community"].astype(int))
    except Exception:
        valid = None
    l1 = layer1(rows, corpus_text(root), tiers, valid)
    l2 = [] if args.layer1_only else layer2(rows, root, tiers, args.workers)

    Path(args.out).write_text(json.dumps({"layer1": l1, "layer2": l2}, indent=2))
    report(l1, l2)


if __name__ == "__main__":
    main()
