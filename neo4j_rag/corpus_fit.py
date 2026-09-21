"""Is this corpus actually a fit for GraphRAG? Measured, not argued. No Azure calls.

GraphRAG was evaluated by Edge et al. on podcast transcripts and news articles.
Those corpora share a property this one may not: the same ENTITIES recur across
many source documents -- a politician in fifty news stories, a guest referenced
across episodes. That recurrence is the entire mechanism. Leiden clusters
entities by edge density, community reports summarise those clusters, and a
cluster only bridges documents if its entities appear in more than one.

A corpus of standards documents might behave differently. Each NIST publication
is already a self-contained topic, and its "entities" are abstract concepts
(PROCESS, ARTIFACT, CONTROL) rather than actors that appear in other people's
documents. If most entities occur in exactly one document, the graph cannot
bridge, and the measured failures follow mechanically rather than from any
defect in the implementation:

  - local search hit 0 of 18 gold units
  - entity-anchored Cypher reached only 3-4 of 18
  - cross-document recall collapsed to 0.267 for vector, 0.067-0.167 for the
    entity-anchored arms

The diagnostics below distinguish "GraphRAG does not work" from "GraphRAG had
nothing to work with here". They are structural properties of the built graph,
so they cost nothing and cannot be noisy.

Run: .venv-neo4j/bin/python neo4j_rag/corpus_fit.py
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrievers import driver  # noqa: E402


def q1(s, cypher, key, **kw):
    r = s.run(cypher, **kw).single()
    return r[key] if r else None


def main():
    d = driver()
    with d.session() as s:
        print("=" * 74)
        print("CORPUS FIT FOR GRAPHRAG — structural diagnostics on the arm A graph")
        print("=" * 74)

        n_ent = q1(s, "MATCH (e:GREntity) RETURN count(e) AS c", "c")
        n_doc = q1(s, "MATCH (d:GRDocument) RETURN count(d) AS c", "c")
        n_chunk = q1(s, "MATCH (c:GRChunk) RETURN count(c) AS c", "c")
        print(f"\n  {n_doc} documents, {n_chunk} chunks, {n_ent} entities")

        # THE diagnostic. An entity confined to one document cannot connect it
        # to another, no matter how the communities are clustered.
        print("\n  How many documents does each entity appear in?")
        dist = {r["ndocs"]: r["c"] for r in s.run("""
            MATCH (e:GREntity)-[:MENTIONED_IN]->(:GRChunk)-[:PART_OF]->(d:GRDocument)
            WITH e, count(DISTINCT d) AS ndocs
            RETURN ndocs, count(e) AS c ORDER BY ndocs
        """)}
        total = sum(dist.values())
        cum = 0
        for ndocs in sorted(dist):
            c = dist[ndocs]
            cum += c
            bar = "#" * max(1, round(60 * c / total))
            print(f"    in {ndocs:>2} doc(s): {c:>5} ({100*c/total:>5.1f}%)  {bar}")
        multi = sum(c for n, c in dist.items() if n >= 2)
        print(f"\n    single-document entities: {total - multi:>5} "
              f"({100*(total-multi)/total:.1f}%)")
        print(f"    multi-document entities:  {multi:>5} "
              f"({100*multi/total:.1f}%)   <- the only ones that can bridge")

        # An entity mentioned once is a hapax: it cannot anchor retrieval and
        # it cannot join a meaningful community.
        print("\n  Entity connectivity:")
        for label, cypher in [
            ("mentioned in exactly 1 chunk",
             "MATCH (e:GREntity) WITH e, count{(e)-[:MENTIONED_IN]->()} AS n "
             "WHERE n = 1 RETURN count(e) AS c"),
            ("no RELATED edge at all",
             "MATCH (e:GREntity) WHERE NOT (e)-[:RELATED]-() RETURN count(e) AS c"),
        ]:
            c = q1(s, cypher, "c")
            print(f"    {label:<34}{c:>6} ({100*c/n_ent:>5.1f}%)")

        degs = [r["n"] for r in s.run(
            "MATCH (e:GREntity) RETURN count{(e)-[:RELATED]-()} AS n")]
        print(f"    median RELATED degree            "
              f"{statistics.median(degs):>6.1f}")
        print(f"    mean RELATED degree              "
              f"{statistics.mean(degs):>6.1f}")

        # A relationship whose endpoints sit in different documents is a literal
        # cross-document bridge. This is what global search exploits.
        print("\n  Cross-document bridges (RELATED edges spanning two documents):")
        r = s.run("""
            MATCH (a:GREntity)-[:RELATED]-(b:GREntity)
            MATCH (a)-[:MENTIONED_IN]->(:GRChunk)-[:PART_OF]->(da:GRDocument)
            MATCH (b)-[:MENTIONED_IN]->(:GRChunk)-[:PART_OF]->(db:GRDocument)
            WITH count(DISTINCT CASE WHEN da <> db THEN [a, b] END) AS cross,
                 count(DISTINCT [a, b]) AS allpairs
            RETURN cross, allpairs
        """).single()
        if r and r["allpairs"]:
            print(f"    entity pairs linked:            {r['allpairs']:>6}")
            print(f"    spanning different documents:   {r['cross']:>6} "
                  f"({100*r['cross']/r['allpairs']:.1f}%)")

        # Communities are the unit global search summarises. A community drawn
        # from one document summarises that document, not the corpus.
        print("\n  Community spread (how many documents each community covers):")
        rows = [(r["level"], r["ndocs"], r["c"]) for r in s.run("""
            MATCH (com:GRCommunity)-[:HAS_ENTITY]->(e:GREntity)
            MATCH (e)-[:MENTIONED_IN]->(:GRChunk)-[:PART_OF]->(d:GRDocument)
            WITH com, count(DISTINCT d) AS ndocs
            RETURN com.level AS level, ndocs, count(com) AS c
            ORDER BY level, ndocs
        """)]
        for lvl in sorted({r[0] for r in rows}):
            sub = [(nd, c) for l, nd, c in rows if l == lvl]
            tot = sum(c for _, c in sub)
            single = sum(c for nd, c in sub if nd == 1)
            avg = sum(nd * c for nd, c in sub) / tot if tot else 0
            print(f"    level {lvl}: {tot:>4} communities, "
                  f"mean {avg:>4.1f} documents each, "
                  f"{single:>4} ({100*single/tot:>5.1f}%) confined to one document")

        print("\n" + "=" * 74)
        print("  Reading: GraphRAG's mechanism is entities recurring ACROSS documents.")
        print("  The more entities confined to a single document, the less there is")
        print("  for the graph to bridge, and the more the method degenerates into")
        print("  per-document summarisation.")
    d.close()


if __name__ == "__main__":
    main()
