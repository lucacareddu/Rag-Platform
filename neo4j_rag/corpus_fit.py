"""Is this corpus a fit for GraphRAG, measured not argued? GraphRAG's mechanism needs entities
recurring across documents; structural diagnostics here check whether this one has that.
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

        # THE diagnostic: an entity confined to one document cannot bridge to another.
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

        # A hapax (mentioned once) can't anchor retrieval or join a meaningful community.
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

        # Endpoints in different documents = a literal cross-document bridge.
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

        # A community drawn from one document summarises that document, not the corpus.
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
