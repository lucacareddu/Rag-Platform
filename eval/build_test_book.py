"""Builds test_book_v2.json from the live corpus: gold contexts as resolvable chunk ids (not
pasted text, which re-chunking silently invalidated), with genuine multi-document questions.
"""
import json
import re
import sys
from pathlib import Path

import requests

QDRANT = "http://localhost:6333"
COLLECTION = "docs"

DOCS = {
    "SP800-40": "SP800-40r4",
    "CSWP": "CSWP_Framework",
    "SP800-184": "SP800-184",
    "SP800-61": "SP800-61r2",
}

# Each anchor: (document, regex that appears in the chunk holding the answer).
QUESTIONS = [
    {
        "id": 1, "category": "baseline-single-doc",
        "question": "What is enterprise patch management, according to NIST SP 800-40?",
        "anchors": [("SP800-40", r"patch management is the process")],
        "reference_answer": "Enterprise patch management is the process of identifying, prioritizing, acquiring, installing, and verifying the installation of patches, updates, and upgrades throughout an organization.",
    },
    {
        "id": 2, "category": "baseline-single-doc",
        "question": "What are the five Functions of the NIST Cybersecurity Framework Core?",
        "anchors": [("CSWP", r"Identify, Protect, Detect, Respond")],
        "reference_answer": "Identify, Protect, Detect, Respond, and Recover.",
    },
    {
        "id": 3, "category": "baseline-single-doc",
        "question": "What phases make up the incident response life cycle in NIST SP 800-61?",
        "anchors": [("SP800-61", r"Containment, Eradication")],
        "reference_answer": "Preparation; Detection and Analysis; Containment, Eradication, and Recovery; and Post-Incident Activity.",
    },
    {
        "id": 4, "category": "cross-document",
        "question": "How do NIST SP 800-61 and SP 800-184 each treat lessons learned after a cybersecurity incident?",
        "anchors": [("SP800-61", r"lessons learned meeting"), ("SP800-184", r"lessons learned during recovery")],
        "reference_answer": "SP 800-61 directs organizations to hold a lessons learned meeting after major incidents (mandatory for major incidents, optional otherwise) and to create a follow-up report, to improve both security measures and the incident handling process. SP 800-184 treats recovery planning as continuous rather than one-time: plans, policies, and procedures should be continually improved by addressing lessons learned during recovery efforts and by periodically validating recovery capabilities, and it warns that delaying documentation makes lessons less accurate and complete.",
    },
    {
        "id": 5, "category": "cross-document",
        "question": "How does incident containment and eradication in SP 800-61 relate to the recovery activities described in SP 800-184?",
        "anchors": [("SP800-61", r"Eradicate the incident"), ("SP800-184", r"recovery playbook|playbook is an action plan")],
        "reference_answer": "SP 800-61 places containment, eradication, and recovery in one phase of the incident response life cycle: evidence is preserved, the incident is contained, exploited vulnerabilities are identified and mitigated, and malware and other components are removed. SP 800-184 picks up the recovery portion in depth, treating it as executing a pre-planned recovery playbook for tactical restoration followed by strategic, continuous improvement of cybersecurity capability.",
    },
    {
        "id": 6, "category": "cross-document",
        "question": "How do the metrics recommended for patch management in SP 800-40 differ from the recovery metrics in SP 800-184?",
        "anchors": [("SP800-40", r"Enterprise-Level Patching Metrics"), ("SP800-184", r"recovery metrics|predefined metrics")],
        "reference_answer": "SP 800-40's patching metrics are predefined and quantitative: the percentage of assets patched by their maintenance plan deadlines, plus mean and median time to patch, broken down by asset importance and vulnerability severity. SP 800-184 notes that recovery metrics cannot always be predefined — for anomalous events there may be no well-defined recovery procedures, so it is unclear which metrics to gather and misused metrics can create a false sense of recovery; it advises deciding carefully when and how recovery metrics are used.",
    },
    {
        "id": 7, "category": "cross-document",
        "question": "How does cybersecurity event recovery in SP 800-184 map onto the Cybersecurity Framework?",
        "anchors": [("SP800-184", r"RC\.CO|Recovery Communications"), ("CSWP", r"RC\.RP|Recovery Planning \(RC")],
        "reference_answer": "SP 800-184 ties recovery to the Cybersecurity Framework's functions, aligning recovery planning with the Recover function and its Recovery Planning category, and treats continual improvement of recovery plans as reflecting the Framework's improvement subcategories. The Framework itself defines Recover as one of five concurrent, continuous functions, covering restoration of capabilities or services impaired by a cybersecurity incident.",
    },
    {
        "id": 8, "category": "cross-document",
        "question": "How does coordination and information sharing with outside parties compare between incident handling and event recovery guidance?",
        "anchors": [("SP800-61", r"sharing.{0,40}information with outside"), ("SP800-184", r"information sharing rules")],
        "reference_answer": "SP 800-61 emphasises coordinating with outside parties such as US-CERT, law enforcement, ISACs and other organisations, and sharing incident information while balancing the benefits against the sensitivity of the data. SP 800-184 emphasises communication during recovery, including notifying external stakeholders of impacts to them and reviewing milestones, goals and metrics with internal and external parties as part of the strategic recovery phase.",
    },
    {
        "id": 9, "category": "relationship",
        "question": "What is the relationship between vulnerability scanning and patch prioritization?",
        "anchors": [("SP800-40", r"Vulnerability scans"), ("SP800-40", r"prioriti")],
        "reference_answer": "Vulnerability scans and passive network monitoring contribute to asset inventory and discovery and reveal which vulnerabilities exist and are being exploited. Patches are then prioritized by how much deploying them would reduce cybersecurity risk, considering each asset's technical and mission characteristics — a patch is higher priority when it reduces more risk, and lower priority when it addresses a low-risk vulnerability on few low-importance assets.",
    },
    {
        "id": 10, "category": "negative-control",
        "question": "What migration timeline does this documentation give for adopting post-quantum cryptography?",
        "anchors": [],
        "reference_answer": "The documentation does not cover post-quantum cryptography migration timelines. An answer grounded in these documents should state that the information is not available rather than provide a timeline.",
    },
]


_FRONT_MATTER = re.compile(
    r"acknowledg|would like to thank|^\s*abstract\b|table of contents", re.I)
# Dot-leader runs are contents-page entries: they list a topic without stating it.
_TOC = re.compile(r"\.{5,}")


def _is_front_matter(text: str) -> bool:
    """Acknowledgments and similar sections name organisations and topics
    without discussing them, so they match topical anchors spuriously — the
    v2 Q8 anchor first resolved to a paragraph thanking US-CERT staff."""
    return bool(_FRONT_MATTER.search(text)) or len(_TOC.findall(text)) >= 2


def _sources() -> dict:
    from neo4j import GraphDatabase
    drv = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "localdevpassword"))
    with drv.session() as s:
        return {r["id"]: r["src"] for r in
                s.run("MATCH (d:Document) RETURN d.id AS id, d.source AS src")}


def fetch_corpus():
    points, offset = [], None
    while True:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        r = requests.post(f"{QDRANT}/collections/{COLLECTION}/points/scroll", json=body).json()["result"]
        points += r["points"]
        offset = r.get("next_page_offset")
        if not offset:
            break
    return points


def main():
    points = fetch_corpus()
    # doc_id -> short label, via any chunk that names its own publication
    by_doc = {}
    for p in points:
        by_doc.setdefault(p["payload"]["doc_id"], []).append(p)

    # Neo4j stores doc_id -> source filename, which is authoritative; sniffing
    # the text for a title is guesswork by comparison.
    label_of = {}
    for doc_id, source in _sources().items():
        for label, needle in DOCS.items():
            if needle.lower().replace("_", "") in source.lower().replace("_", ""):
                label_of[doc_id] = label
                break
    labelled = {v: k for k, v in label_of.items()}
    print("resolved documents:", {k: len(by_doc[v]) for k, v in labelled.items()})
    missing = set(DOCS) - set(labelled)
    if missing:
        sys.exit(f"ERROR: could not identify documents in corpus: {missing}")

    out, failures = [], []
    for q in QUESTIONS:
        ids = []
        for doc_label, pattern in q["anchors"]:
            pool = by_doc[labelled[doc_label]]
            hits = [c for c in pool
                    if re.search(pattern, c["payload"]["text"], re.I)
                    and not _is_front_matter(c["payload"]["text"])]
            # Scroll order is arbitrary; a chunk that states the anchor phrase
            # repeatedly is discussing it, one that mentions it once in passing
            # is not. Rank by match density so the gold set is the real answer.
            hits.sort(key=lambda c: len(re.findall(pattern, c["payload"]["text"], re.I)),
                      reverse=True)
            if not hits:
                failures.append(f"[{q['id']}] no chunk in {doc_label} matches {pattern!r}")
                continue
            ids.extend(h["id"] for h in hits[:2])
        docs_covered = sorted({label_of[c["payload"]["doc_id"]]
                               for c in points if c["id"] in set(ids)})
        if q["category"] == "cross-document" and len(docs_covered) < 2:
            failures.append(f"[{q['id']}] labelled cross-document but resolves to {docs_covered}")
        out.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "reference_answer": q["reference_answer"],
            "reference_chunk_ids": ids,
            "source_documents": docs_covered,
        })

    if failures:
        print("\nANCHOR FAILURES — fix before using this book:")
        for f in failures:
            print("  ", f)

    path = Path(__file__).parent / "test_book_v2.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path} ({len(out)} questions)")
    for r in out:
        print(f"  [{r['id']:>2}] {r['category']:<20} {len(r['reference_chunk_ids'])} chunks {r['source_documents']}")


if __name__ == "__main__":
    main()
