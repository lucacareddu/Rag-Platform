"""Arm A vs arm B: retrieval scored across two different chunkings.

The exact-id metric used in eval_retrieval.py cannot compare these arms. Arm A
reuses GraphRAG's 542 chunks; arm B's SimpleKGPipeline re-split the same
documents into 664 chunks with different boundaries, so no gold chunk id exists
in arm B's graph and every score would be 0.000 by construction -- the exact
failure mode that has already produced two false results in this experiment.

So the metric here is chunking-independent. Each gold unit was originally
DEFINED by an (document fragment, regex) anchor pair in gold_context.py; a
retrieved chunk counts as hitting that gold unit when it comes from the right
document AND matches the same regex. That is the identical criterion which
selected the gold units in the first place, so it is not a looser proxy -- it
is the definition, applied to whatever chunking the arm happens to use.

Both arms are scored by this one metric, so the comparison is fair even though
their chunk boundaries differ. Arm A is additionally scored by exact id in
eval_retrieval.py, and the two should roughly agree for it; they are reported
side by side so any divergence is visible rather than hidden.

Cost: one embedding call per query per arm. No generation, no judge.

Run: .venv-neo4j/bin/python neo4j_rag/eval_ab.py [--top-k 10]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from neo4j_graphrag.retrievers import VectorCypherRetriever

from retrievers import IDX_CHUNK, driver, embedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval/gold_context.json"
OUT = ROOT / "eval/results_neo4j_ab.json"

IDX_B = "b_chunk_embedding"

# Copied verbatim from gold_context.py, which used these to SELECT the gold
# units. Reusing them to score is therefore applying the original definition,
# not approximating it.
ANCHORS = {
    1: [("SP800-88", r"Clear applies logical techniques"),
        ("SP800-88", r"Purge applies physical or logical")],
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


def ensure_index_b(d):
    with d.session() as s:
        s.run(f"""CREATE VECTOR INDEX {IDX_B} IF NOT EXISTS
                  FOR (n:Chunk) ON (n.embedding)
                  OPTIONS {{indexConfig: {{
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'}}}}""")
        state = s.run("SHOW INDEXES YIELD name, state WHERE name = $n "
                      "RETURN state", n=IDX_B).single()
        print(f"arm B index {IDX_B}: {state['state'] if state else 'MISSING'}",
              file=sys.stderr)


# Returning the document title from the same query removes a fragile join.
# The first version matched retrieved text back to the graph with
# `WHERE c.text = $t` to find its document; that lookup silently returned no
# row, every title came back empty, and arm B scored 0/20 -- retrieval was
# fine, the scoring join was not.
Q_A = """
WITH node AS c, score
MATCH (c)-[:PART_OF]->(d:GRDocument)
RETURN c.text AS text, d.title AS title, score
ORDER BY score DESC
"""

Q_B = """
WITH node AS c, score
OPTIONAL MATCH (c)-[:FROM_DOCUMENT]->(d:Document)
RETURN c.text AS text, coalesce(d.path, '') AS title, score
ORDER BY score DESC
"""


def pairs_from(res) -> list:
    """(text, document title) per retrieved chunk, straight from the Record."""
    out = []
    for it in res.items:
        meta = getattr(it, "metadata", None) or {}
        text, title = meta.get("text"), meta.get("title")
        if text is None:
            s = str(it.content)
            m = re.search(r"text='(.*?)' title='(.*?)'", s, re.S)
            if m:
                text, title = m.group(1), m.group(2)
        if text is not None:
            out.append((text, title or ""))
    return out


def score(pairs, qid) -> tuple:
    """How many of this question's anchors are satisfied by the retrieved set."""
    anchors = ANCHORS.get(qid, [])
    hit = 0
    for doc_frag, pattern in anchors:
        for text, title in pairs:
            if doc_frag.lower() in (title or "").lower() and \
                    re.search(pattern, text, re.I):
                hit += 1
                break
    return hit, len(anchors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    gold = {int(k): v for k, v in json.loads(GOLD.read_text()).items()}
    d, emb = driver(), embedder()
    ensure_index_b(d)

    arms = {
        "A_vector": VectorCypherRetriever(d, index_name=IDX_CHUNK, embedder=emb,
                                          retrieval_query=Q_A),
        "B_vector": VectorCypherRetriever(d, index_name=IDX_B, embedder=emb,
                                          retrieval_query=Q_B),
    }

    rows = []
    for qid, g in sorted(gold.items()):
        for name, r in arms.items():
            arm = name[0]
            try:
                res = r.search(query_text=g["question"], top_k=args.top_k)
            except Exception as e:
                print(f"  [{qid}|{name}] FAILED {str(e)[:110]}", file=sys.stderr)
                continue
            pairs = pairs_from(res)
            texts = pairs
            hit, total = score(pairs, qid)
            rows.append({"id": qid, "tier": g["tier"], "arm": name,
                         "n_retrieved": len(texts), "hit": hit, "anchors": total,
                         "recall": hit / total if total else 0.0})
            print(f"[{qid:>2}|{g['tier']:<16}] {name:<10} "
                  f"retrieved={len(texts):>3} anchor_hit={hit}/{total}",
                  file=sys.stderr)
    d.close()

    OUT.write_text(json.dumps(rows, indent=2))
    report(rows, args.top_k)


def report(rows, top_k):
    print("\n" + "=" * 72)
    print(f"ARM A vs ARM B — anchor recall, chunking-independent, top_k={top_k}")
    print("=" * 72)
    print(f"  {'arm':<12}{'anchors hit':>13}{'recall':>10}{'factual':>10}{'cross-doc':>12}")
    for arm in ["A_vector", "B_vector"]:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        h, t = sum(r["hit"] for r in sub), sum(r["anchors"] for r in sub)
        f = [r["recall"] for r in sub if r["tier"] == "local"]
        c = [r["recall"] for r in sub if r["tier"] == "cross-document"]
        print(f"  {arm:<12}{f'{h}/{t}':>13}"
              f"{statistics.mean(r['recall'] for r in sub):>10.3f}"
              f"{statistics.mean(f) if f else 0:>10.3f}"
              f"{statistics.mean(c) if c else 0:>12.3f}")
    print("\n  A = GraphRAG's 542 chunks reused; B = SimpleKGPipeline's 664 chunks.")
    print("  Same embedder, same questions, same anchor definitions.")


if __name__ == "__main__":
    main()
