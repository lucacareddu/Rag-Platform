"""Arm A: load the existing GraphRAG index into Neo4j. Zero Azure calls.

The GraphRAG branch already paid $1.43 to extract 6,302 entities, 3,248
relationships, 542 chunks and 550 community reports, and to embed three of
those sets at 1536 dimensions. All of it is on disk as parquet plus LanceDB,
and all of it is reusable: this script is a pandas-to-Cypher bulk load with no
model involved.

That makes it the controlled experiment. Same chunks, same entities, same
relationships, same embeddings, same generator -- only the retrieval strategy
differs between this and the GraphRAG arms, so any measured difference is
attributable to retrieval alone. Arm B (SimpleKGPipeline) rebuilds the graph
with Neo4j's own extractor and is therefore confounded with extraction
differences; it answers a different question and is kept separate.

Namespacing: Neo4j Community has no multi-database support, so arm A and arm B
cannot be separated by database. Arm A uses a GR* label prefix; arm B writes
the library's default labels. They coexist without collision.

Verified before writing a line of this (each was a silent-corruption risk):
  - entity titles are unique, 6302/6302, so matching relationships by title is
    unambiguous -- relationships.parquet stores source/target as TITLES, not ids
  - 0 relationships reference an unknown entity
  - LanceDB ids align exactly with parquet ids: 6302/6302 and 542/542
  - no nulls in entity.description, relationship.description or text_unit.text

Run: .venv-neo4j/bin/python neo4j_rag/load_graphrag.py [--wipe]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import lancedb
import pandas as pd
from neo4j import GraphDatabase

# Defaults to feat_graphrag's index root; override for a copy kept elsewhere.
ARTIFACTS = Path(os.environ.get(
    "GRAPHRAG_ARTIFACTS",
    Path(__file__).resolve().parents[1] / "graphrag" / "output"))

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
BATCH = 500

# Vector index names are referenced by the retrievers in retrievers.py.
IDX_ENTITY = "gr_entity_embedding"
IDX_CHUNK = "gr_chunk_embedding"
IDX_COMMUNITY = "gr_community_embedding"
FT_CHUNK = "gr_chunk_text"


def password() -> str:
    """Read the container's password rather than hardcoding one."""
    if os.environ.get("NEO4J_PASSWORD"):
        return os.environ["NEO4J_PASSWORD"]
    out = subprocess.run(
        ["docker", "inspect", "rag-platform-neo4j-1", "--format",
         "{{range .Config.Env}}{{println .}}{{end}}"],
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("NEO4J_AUTH="):
            return line.split("/", 1)[1].strip()
    raise RuntimeError("NEO4J_AUTH not found; set NEO4J_PASSWORD")


def vectors(db, table: str) -> dict:
    """id -> 1536-float list, from the LanceDB tables GraphRAG already wrote."""
    d = db.open_table(table).to_pandas()
    return {r.id: [float(x) for x in r.vector] for r in d.itertuples()}


def run_batched(session, cypher: str, rows: list, label: str):
    for i in range(0, len(rows), BATCH):
        session.run(cypher, rows=rows[i:i + BATCH])
    print(f"  {label:<28} {len(rows):>6}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wipe", action="store_true",
                    help="delete existing GR* nodes first (arm A only, never arm B's)")
    args = ap.parse_args()

    if not ARTIFACTS.exists():
        sys.exit(f"artifacts not found: {ARTIFACTS}")

    ent = pd.read_parquet(ARTIFACTS / "entities.parquet")
    rel = pd.read_parquet(ARTIFACTS / "relationships.parquet")
    tu = pd.read_parquet(ARTIFACTS / "text_units.parquet")
    doc = pd.read_parquet(ARTIFACTS / "documents.parquet")
    com = pd.read_parquet(ARTIFACTS / "communities.parquet")
    rep = pd.read_parquet(ARTIFACTS / "community_reports.parquet")

    db = lancedb.connect(str(ARTIFACTS / "lancedb"))
    ent_vec = vectors(db, "default-entity-description")
    tu_vec = vectors(db, "default-text_unit-text")
    com_vec = vectors(db, "default-community-full_content")
    print(f"embeddings: entity={len(ent_vec)} chunk={len(tu_vec)} "
          f"community={len(com_vec)}", file=sys.stderr)

    driver = GraphDatabase.driver(URI, auth=("neo4j", password()))
    with driver.session() as s:
        if args.wipe:
            for lbl in ["GREntity", "GRChunk", "GRDocument", "GRCommunity"]:
                s.run(f"MATCH (n:{lbl}) DETACH DELETE n")
            print("  wiped arm A labels", file=sys.stderr)

        for lbl in ["GREntity", "GRChunk", "GRDocument", "GRCommunity"]:
            s.run(f"CREATE CONSTRAINT {lbl.lower()}_id IF NOT EXISTS "
                  f"FOR (n:{lbl}) REQUIRE n.id IS UNIQUE")

        print("nodes:", file=sys.stderr)
        run_batched(s, """
            UNWIND $rows AS r
            MERGE (d:GRDocument {id: r.id})
            SET d.title = r.title
        """, [{"id": r.id, "title": r.title} for r in doc.itertuples()], "GRDocument")

        run_batched(s, """
            UNWIND $rows AS r
            MERGE (c:GRChunk {id: r.id})
            SET c.text = r.text, c.n_tokens = r.n_tokens, c.embedding = r.embedding
        """, [{"id": r.id, "text": r.text, "n_tokens": int(r.n_tokens),
               "embedding": tu_vec.get(r.id)} for r in tu.itertuples()], "GRChunk")

        run_batched(s, """
            UNWIND $rows AS r
            MERGE (e:GREntity {id: r.id})
            SET e.title = r.title, e.type = r.type, e.description = r.description,
                e.degree = r.degree, e.frequency = r.frequency,
                e.embedding = r.embedding
        """, [{"id": r.id, "title": r.title, "type": r.type,
               "description": r.description, "degree": int(r.degree),
               "frequency": int(r.frequency), "embedding": ent_vec.get(r.id)}
              for r in ent.itertuples()], "GREntity")

        reports = rep.set_index("community")
        run_batched(s, """
            UNWIND $rows AS r
            MERGE (c:GRCommunity {id: r.id})
            SET c.community = r.community, c.level = r.level, c.title = r.title,
                c.summary = r.summary, c.full_content = r.full_content,
                c.rank = r.rank, c.embedding = r.embedding
        """, [{"id": r.id, "community": int(r.community), "level": int(r.level),
               "title": r.title, "summary": r.summary,
               "full_content": r.full_content, "rank": float(r.rank),
               "embedding": com_vec.get(r.id)}
              for r in reports.reset_index().itertuples()], "GRCommunity")

        print("relationships:", file=sys.stderr)
        run_batched(s, """
            UNWIND $rows AS r
            MATCH (c:GRChunk {id: r.chunk}), (d:GRDocument {id: r.doc})
            MERGE (c)-[:PART_OF]->(d)
        """, [{"chunk": r.id, "doc": d} for r in tu.itertuples()
              for d in r.document_ids], "PART_OF")

        run_batched(s, """
            UNWIND $rows AS r
            MATCH (e:GREntity {id: r.entity}), (c:GRChunk {id: r.chunk})
            MERGE (e)-[:MENTIONED_IN]->(c)
        """, [{"entity": r.id, "chunk": c} for r in ent.itertuples()
              for c in r.text_unit_ids], "MENTIONED_IN")

        # Matched on title: relationships.parquet stores names, and titles were
        # verified unique so this cannot bind the wrong node.
        run_batched(s, """
            UNWIND $rows AS r
            MATCH (a:GREntity {title: r.source}), (b:GREntity {title: r.target})
            MERGE (a)-[x:RELATED {id: r.id}]->(b)
            SET x.description = r.description, x.weight = r.weight
        """, [{"id": r.id, "source": r.source, "target": r.target,
               "description": r.description, "weight": float(r.weight)}
              for r in rel.itertuples()], "RELATED")

        run_batched(s, """
            UNWIND $rows AS r
            MATCH (c:GRCommunity {id: r.community}), (e:GREntity {id: r.entity})
            MERGE (c)-[:HAS_ENTITY]->(e)
        """, [{"community": cid, "entity": e}
              for cid, ents in zip(rep["id"], com.set_index("community")
                                   .loc[rep["community"], "entity_ids"])
              for e in ents], "HAS_ENTITY")

        print("indexes:", file=sys.stderr)
        for name, lbl in [(IDX_ENTITY, "GREntity"), (IDX_CHUNK, "GRChunk"),
                          (IDX_COMMUNITY, "GRCommunity")]:
            s.run(f"""CREATE VECTOR INDEX {name} IF NOT EXISTS
                      FOR (n:{lbl}) ON (n.embedding)
                      OPTIONS {{indexConfig: {{
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'}}}}""")
            print(f"  {name}", file=sys.stderr)
        s.run(f"CREATE FULLTEXT INDEX {FT_CHUNK} IF NOT EXISTS "
              f"FOR (n:GRChunk) ON EACH [n.text]")
        print(f"  {FT_CHUNK}", file=sys.stderr)

        verify(s)
    driver.close()


def verify(s):
    print("\nverification:", file=sys.stderr)
    checks = {
        "GRDocument": "MATCH (n:GRDocument) RETURN count(n) AS c",
        "GRChunk": "MATCH (n:GRChunk) RETURN count(n) AS c",
        "GREntity": "MATCH (n:GREntity) RETURN count(n) AS c",
        "GRCommunity": "MATCH (n:GRCommunity) RETURN count(n) AS c",
        "PART_OF": "MATCH ()-[r:PART_OF]->() RETURN count(r) AS c",
        "MENTIONED_IN": "MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS c",
        "RELATED": "MATCH ()-[r:RELATED]->() RETURN count(r) AS c",
        "HAS_ENTITY": "MATCH ()-[r:HAS_ENTITY]->() RETURN count(r) AS c",
    }
    for k, q in checks.items():
        print(f"  {k:<16} {s.run(q).single()['c']:>7}", file=sys.stderr)

    # An entity or chunk without a vector is invisible to every vector
    # retriever, and would look like a retrieval failure rather than a load bug.
    for lbl in ["GREntity", "GRChunk", "GRCommunity"]:
        n = s.run(f"MATCH (n:{lbl}) WHERE n.embedding IS NULL "
                  f"RETURN count(n) AS c").single()["c"]
        flag = "OK" if n == 0 else "!! MISSING EMBEDDINGS"
        print(f"  {lbl} without embedding: {n}  {flag}", file=sys.stderr)


if __name__ == "__main__":
    main()
