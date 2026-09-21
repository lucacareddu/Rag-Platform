"""Resolve gold text units for test_book_v3's local/cross-document questions, by regex anchor
so the set can't drift from the corpus. Sensemaking and negative-control tiers have no gold set.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "gold_context.json"

# (document-title fragment, regex that appears in the answer-bearing unit)
ANCHORS = {
    1: [("SP800-88", r"Clear applies logical techniques"),
        ("SP800-88", r"Purge applies physical or logical")],
    # Short anchors: PDF-extraction line breaks mean a long literal phrase never matches.
    2: [("SP800-207", r"considered resources"),
        ("SP800-207", r"secured regardless")],
    3: [("SP800-37", r"Categorize|Prepare.{0,80}Categorize"),
        ("SP800-37", r"seven steps|Authorize.{0,40}Monitor")],
    4: [("SP800-63-3", r"Identity Assurance Level"),
        ("SP800-63-3", r"Authenticator Assurance Level|Federation Assurance Level")],
    5: [("SP800-61", r"lessons learned meeting"),
        ("SP800-184", r"lessons learned|continually improve")],
    6: [("SP800-82", r"availability.{0,120}integrity|performance and reliability requirements"),
        ("SP800-82", r"ICS.{0,80}differ|safety")],
    7: [("SP800-40", r"prioriti.{0,80}risk|risk response"),
        ("SP800-30", r"threat sources|likelihood.{0,60}impact"),
        ("SP800-37", r"continuous monitoring|ongoing authorization")],
    8: [("CSWP", r"Identify, Protect, Detect, Respond"),
        ("SP800-61", r"Containment, Eradication"),
        ("SP800-184", r"RC\.RP|Recovery Planning")],
    9: [("SP800-88", r"sanitiz.{0,80}confidentiality|media.{0,60}reuse"),
        ("SP800-171", r"sanitize or destroy|media protection")],
}

# Chunks that name a topic without discussing it match topical anchors
# spuriously — the failure that put an acknowledgments paragraph in the
# previous gold set.
# Anchored to line starts: a bare substring match rejected the whole zero-trust
# tenets unit because the word "acknowledged" appeared once in its prose, 4000
# characters in. Only a section heading means the unit *is* front matter.
_FRONT_MATTER = re.compile(
    r"(?mi)^\s*acknowledge?ments?\b|^\s*table of contents\b|would like to thank")
_TOC = re.compile(r"\.{5,}")


def is_front_matter(t: str) -> bool:
    return bool(_FRONT_MATTER.search(t)) or len(_TOC.findall(t)) >= 2


def main():
    tu = pd.read_parquet(ROOT / "graphrag/output/text_units.parquet")
    docs = pd.read_parquet(ROOT / "graphrag/output/documents.parquet")
    title_of = dict(zip(docs["id"], docs["title"]))
    tu["doc_title"] = tu["document_ids"].apply(
        lambda ids: title_of.get(ids[0], "") if len(ids) else "")

    book = json.loads((Path(__file__).parent / "test_book_v3.json").read_text())
    out, failures = {}, []

    for item in book:
        qid = item["id"]
        if qid not in ANCHORS:
            continue
        ids, texts = [], []
        for doc_frag, pattern in ANCHORS[qid]:
            pool = tu[tu["doc_title"].str.contains(doc_frag, case=False, regex=False)]
            if pool.empty:
                failures.append(f"[{qid}] no document matching {doc_frag!r}")
                continue
            hits = [(len(re.findall(pattern, t, re.I)), i, t)
                    for i, t in zip(pool["id"], pool["text"])
                    if re.search(pattern, t, re.I) and not is_front_matter(t)]
            if not hits:
                failures.append(f"[{qid}] {doc_frag}: no unit matches {pattern!r}")
                continue
            # A unit repeating the anchor is discussing it; one mentioning it
            # once in passing is not.
            hits.sort(reverse=True, key=lambda h: h[0])
            for _, i, t in hits[:1]:
                if i not in ids:
                    ids.append(i)
                    texts.append(t)
        out[str(qid)] = {"tier": item["tier"], "question": item["question"],
                         "gold_ids": ids, "gold_texts": texts}

    if failures:
        print("ANCHOR FAILURES:", file=sys.stderr)
        for f in failures:
            print("  ", f, file=sys.stderr)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    for k, v in out.items():
        print(f"  [{k:>2}] {v['tier']:<16} {len(v['gold_ids'])} gold units")


if __name__ == "__main__":
    main()
