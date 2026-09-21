"""Structural fit diagnostics straight from GraphRAG parquet. No Azure calls.

Same measurements the Neo4j branch ran against the loaded graph, computed here
from the index artifacts so any GraphRAG root can be checked without a database.

The point is to test a prediction rather than describe a corpus. The cheap
screener (neo4j_rag/screen_corpus.py) read 5.6% multi-document entities on the
NIST corpus and 9.2% on the incident corpus, from samples costing ~$0.01 each,
and claimed the second would fit GraphRAG better. The NIST index later measured
7.9% on the full graph. Running the same full measurement here says whether the
screener's ordering held on a corpus it had never seen -- which is the only way
to find out if it predicts or merely describes.

Run: .venv-graphrag/bin/python eval/corpus_fit_parquet.py graphrag_incidents
"""
import statistics
import sys
from pathlib import Path

import pandas as pd


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "graphrag")
    out = root / "output"
    ent = pd.read_parquet(out / "entities.parquet")
    rel = pd.read_parquet(out / "relationships.parquet")
    tu = pd.read_parquet(out / "text_units.parquet")
    doc = pd.read_parquet(out / "documents.parquet")
    com = pd.read_parquet(out / "communities.parquet")

    print("=" * 72)
    print(f"CORPUS FIT — {root}")
    print("=" * 72)
    print(f"\n  {len(doc)} documents, {len(tu)} text units, {len(ent)} entities, "
          f"{len(rel)} relationships")

    # chunk id -> document id
    chunk_doc = {}
    for r in tu.itertuples():
        if len(r.document_ids):
            chunk_doc[r.id] = r.document_ids[0]

    # THE diagnostic: an entity confined to one document cannot bridge.
    ndocs = []
    for r in ent.itertuples():
        docs = {chunk_doc.get(c) for c in r.text_unit_ids}
        docs.discard(None)
        ndocs.append(len(docs))
    ent = ent.assign(ndocs=ndocs)

    total = len(ent)
    multi = int((ent["ndocs"] >= 2).sum())
    print("\n  Documents per entity:")
    for n, c in ent["ndocs"].value_counts().sort_index().items():
        bar = "#" * max(1, round(50 * c / total))
        print(f"    in {n:>2} doc(s): {c:>5} ({100*c/total:>5.1f}%)  {bar}")
    print(f"\n    single-document entities: {total - multi:>5} "
          f"({100*(total-multi)/total:.1f}%)")
    print(f"    MULTI-document entities:  {multi:>5} "
          f"({100*multi/total:.1f}%)   <- the bridging capacity")

    # Connectivity
    deg = {}
    for r in rel.itertuples():
        deg[r.source] = deg.get(r.source, 0) + 1
        deg[r.target] = deg.get(r.target, 0) + 1
    degrees = [deg.get(t, 0) for t in ent["title"]]
    iso = sum(1 for d in degrees if d == 0)
    one_chunk = int((ent["text_unit_ids"].apply(len) == 1).sum())
    print("\n  Connectivity:")
    print(f"    mentioned in exactly 1 chunk   {one_chunk:>6} "
          f"({100*one_chunk/total:>5.1f}%)")
    print(f"    no relationship at all         {iso:>6} ({100*iso/total:>5.1f}%)")
    print(f"    median degree                  {statistics.median(degrees):>6.1f}")
    print(f"    mean degree                    {statistics.mean(degrees):>6.1f}")

    # Relationships whose endpoints live in different documents are literal
    # cross-document bridges -- what global search exploits.
    docs_of = dict(zip(ent["title"], ent["ndocs"]))
    ent_docs = {}
    for r in ent.itertuples():
        d = {chunk_doc.get(c) for c in r.text_unit_ids}
        d.discard(None)
        ent_docs[r.title] = d
    cross = sum(1 for r in rel.itertuples()
                if ent_docs.get(r.source) and ent_docs.get(r.target)
                and ent_docs[r.source] != ent_docs[r.target]
                and not ent_docs[r.source] & ent_docs[r.target])
    print(f"\n  Cross-document relationships:  {cross:>6} of {len(rel)} "
          f"({100*cross/len(rel):.1f}%)")

    # Communities drawn from one document summarise that document, not the corpus.
    print("\n  Community spread:")
    for lvl in sorted(com["level"].unique()):
        sub = com[com["level"] == lvl]
        spreads = []
        for r in sub.itertuples():
            d = set()
            for e in r.entity_ids:
                row = ent[ent["id"] == e]
                if len(row):
                    d |= ent_docs.get(row.iloc[0]["title"], set())
            spreads.append(len(d))
        if not spreads:
            continue
        single = sum(1 for x in spreads if x <= 1)
        print(f"    level {lvl}: {len(sub):>4} communities, "
              f"mean {statistics.mean(spreads):>4.1f} documents each, "
              f"{single:>4} ({100*single/len(spreads):>5.1f}%) single-document")

    print("\n  Reference — NIST corpus, same measurement:")
    print("    multi-document entities 7.9% | no relationship 56.7% | "
          "median degree 0.0")
    print("    level 0: 26 communities, mean 8.0 docs, 3.8% single-document")


if __name__ == "__main__":
    main()
