"""Rebuild MENTIONS for every document, restoring edges dropped by the old max_mention_ratio cap.
Redone locally from data already in Neo4j, at zero API cost.
"""
import re

from .config import settings
from .neo4j_client import driver


def main():
    with driver.session(database=settings.neo4j_database) as s:
        entities = [(r["key"], [r["name"]] + (r["aliases"] or [])) for r in
                    s.run("MATCH (e:Entity) "
                          "RETURN e.key AS key, e.name AS name, e.aliases AS aliases")]
        chunks = [(r["id"], r["text"].lower()) for r in
                  s.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text")]
        print(f"entities: {len(entities)}, chunks: {len(chunks)}")

        before = s.run("MATCH ()-[m:MENTIONS]->() RETURN count(m) AS n").single()["n"]

        mentions, counts = [], {}
        for key, names in entities:
            # Any surface form counts as a mention of the same node.
            pattern = "|".join(r"(?<![A-Za-z0-9])" + re.escape(n.lower()) + r"(?![A-Za-z0-9])"
                               for n in names)
            hits = [cid for cid, text in chunks if re.search(pattern, text)]
            counts[key] = len(hits)
            mentions.extend({"chunk_id": cid, "key": key} for cid in hits)

        s.run("MATCH ()-[m:MENTIONS]->() DELETE m")
        for i in range(0, len(mentions), 5000):
            s.run("UNWIND $rows AS r "
                  "MATCH (c:Chunk {id: r.chunk_id}), (e:Entity {key: r.key}) "
                  "MERGE (c)-[:MENTIONS]->(e)", rows=mentions[i:i + 5000])
        s.run("UNWIND $rows AS r MATCH (e:Entity {key:r.key}) SET e.mention_count = r.n",
              rows=[{"key": k, "n": n} for k, n in counts.items()])

        after = s.run("MATCH ()-[m:MENTIONS]->() RETURN count(m) AS n").single()["n"]
        orphans = s.run("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() "
                        "RETURN count(e) AS n").single()["n"]
        cross = s.run(
            "MATCH (e:Entity)-[:MENTIONS]-(:Chunk)-[:HAS_CHUNK]-(d:Document) "
            "WITH e, count(DISTINCT d) AS docs WHERE docs > 1 "
            "RETURN count(e) AS n").single()["n"]
        print(f"MENTIONS: {before} -> {after}")
        print(f"entities with no mentions: {orphans}")
        print(f"cross-document entities: {cross}")


if __name__ == "__main__":
    main()
