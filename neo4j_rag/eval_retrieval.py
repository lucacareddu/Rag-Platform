"""Retrieval-only scoring for the Neo4j arms, one embedding call per query. Chunks carry their
GraphRAG ids through the load, so hits are scored by exact id, not text overlap or a judge.
"""
import argparse
import ast
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrievers import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval/gold_context.json"
OUT = ROOT / "eval/results_neo4j_retrieval.json"

ARMS = ["vector", "entity_cypher", "entity_expand", "hybrid"]


def chunk_ids(items) -> list:
    """Pull the GraphRAG chunk id out of whichever shape the retriever returns.

    The two retriever families return different shapes and BOTH are strings, so
    this has to be explicit. Getting it wrong is silent: an earlier version
    string-split the Record repr on "chunk_id" and extracted "': '" as the id,
    which scored 0/18 gold units and looked exactly like a retrieval failure.

      VectorRetriever / HybridRetriever
        content is a repr of the return_properties dict, e.g. "{'id': ..., }".
        NOTE metadata['id'] here is Neo4j's internal element id
        ("4:841611b1-...:5"), not the chunk id -- using it silently scores zero.
      VectorCypherRetriever
        content is a neo4j Record repr, but metadata carries the map built in
        the Cypher, so metadata['chunk_id'] is authoritative.
    """
    out = []
    for it in items:
        meta = getattr(it, "metadata", None) or {}
        cid = meta.get("chunk_id")
        if not cid:
            c = it.content
            if isinstance(c, dict):
                cid = c.get("id")
            else:
                try:
                    parsed = ast.literal_eval(str(c))
                    if isinstance(parsed, dict):
                        cid = parsed.get("id")
                except (ValueError, SyntaxError):
                    cid = None
        if cid:
            out.append(cid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10,
                    help="chunks returned per arm; 10 matches basic's ~9")
    args = ap.parse_args()

    gold = json.loads(GOLD.read_text())
    book = {int(k): v for k, v in gold.items()}
    rets, driver = build(ARMS, top_k=args.top_k)

    rows = []
    for qid, g in sorted(book.items()):
        gold_ids = set(g["gold_ids"])
        for arm, r in rets.items():
            try:
                res = r.search(query_text=g["question"], top_k=args.top_k)
            except Exception as e:
                print(f"  [{qid}|{arm}] FAILED {str(e)[:100]}", file=sys.stderr)
                continue
            got = chunk_ids(res.items)
            hit = gold_ids & set(got)
            rows.append({"id": qid, "tier": g["tier"], "arm": arm,
                         "n_retrieved": len(got), "n_gold": len(gold_ids),
                         "n_hit": len(hit),
                         "recall": len(hit) / len(gold_ids) if gold_ids else 0.0,
                         "precision": len(hit) / len(got) if got else 0.0,
                         "retrieved_ids": got})
            print(f"[{qid:>2}|{g['tier']:<16}] {arm:<14} "
                  f"retrieved={len(got):>3} gold_hit={len(hit)}/{len(gold_ids)}",
                  file=sys.stderr)
    driver.close()

    OUT.write_text(json.dumps(rows, indent=2))
    report(rows, args.top_k)


def report(rows, top_k):
    print("\n" + "=" * 76)
    print(f"NEO4J RETRIEVAL — exact gold-chunk id match, top_k={top_k}")
    print("=" * 76)
    print(f"  {'arm':<16}{'gold hit':>10}{'recall':>10}{'precision':>12}{'chunks/q':>11}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        hits = sum(r["n_hit"] for r in sub)
        total = sum(r["n_gold"] for r in sub)
        print(f"  {arm:<16}{f'{hits}/{total}':>10}"
              f"{statistics.mean(r['recall'] for r in sub):>10.3f}"
              f"{statistics.mean(r['precision'] for r in sub):>12.3f}"
              f"{statistics.mean(r['n_retrieved'] for r in sub):>11.1f}")

    print("\n  by tier (recall):")
    print(f"  {'arm':<16}{'factual n=4':>14}{'cross-doc n=5':>16}")
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        f = [r["recall"] for r in sub if r["tier"] == "local"]
        c = [r["recall"] for r in sub if r["tier"] == "cross-document"]
        print(f"  {arm:<16}{statistics.mean(f) if f else 0:>14.3f}"
              f"{statistics.mean(c) if c else 0:>16.3f}")

    print("\n  GraphRAG baselines on the same 9 questions / 18 gold units:")
    print("    basic  7/18 gold, recall 0.481, precision 0.212")
    print("    local  0/18 gold, recall 0.000, precision 0.000")


if __name__ == "__main__":
    main()
