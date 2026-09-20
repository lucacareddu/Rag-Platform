import logging
import re

from neo4j import GraphDatabase
from .config import settings
from .entity_norm import resolve

logger = logging.getLogger(__name__)

driver = GraphDatabase.driver(
    settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
)

_schema_ready = False

# entity_name powers rag-api's query-side entity linking.
SCHEMA = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
]


def ensure_schema():
    global _schema_ready
    if _schema_ready:
        return
    with driver.session(database=settings.neo4j_database) as s:
        for stmt in SCHEMA:
            s.run(stmt)
    _schema_ready = True


def write_document(doc_id: str, source: str, chunks: list[dict],
                   entities: list[dict], relations: list[dict]):
    """Writes the whole document's graph in one transaction. `chunks` ids must
    match the Qdrant point ids — that's the join key rag-api relies on."""
    ensure_schema()
    entities, relations = resolve(entities, relations)
    entities, mentions = _map_mentions(chunks, entities)
    # resolve() may drop an entity that _map_mentions then also prunes as a hub;
    # relations pointing at a pruned node would MATCH nothing, so drop them too.
    live = {e["key"] for e in entities}
    relations = [r for r in relations
                 if r["source_key"] in live and r["target_key"] in live]
    with driver.session(database=settings.neo4j_database) as s:
        s.execute_write(_write_tx, doc_id, source, chunks, entities, relations, mentions)


def _map_mentions(chunks: list[dict], entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Links entities to the chunks that mention them (whole-phrase match, not
    substring), recording how widely each is mentioned.

    Frequent entities are NOT dropped. Deleting them here was suppressing
    exactly the cross-document bridges the graph exists to provide: a
    document's central topic is the one entity another document about the same
    subject also names ("Cybersecurity Framework" covers 34% of CSWP chunks, so
    a 15% cap erased the only real link to SP 800-184). Sparse entities like
    author names survived that cap, which is why every cross-document entity
    was publication metadata.

    Hub dominance is a *ranking* problem, so it is solved at ranking time —
    rag-api discounts by mention_count rather than having the edge removed.
    """
    lowered = [(c["id"], c["text"].lower()) for c in chunks]
    kept, out = [], []
    for e in entities:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(e["name"].lower()) + r"(?![A-Za-z0-9])"
        hits = [chunk_id for chunk_id, text in lowered if re.search(pattern, text)]
        if not hits:
            continue
        kept.append({**e, "mention_count": len(hits)})
        out.extend({"chunk_id": chunk_id, "key": e["key"]} for chunk_id in hits)
    return kept, out


def _write_tx(tx, doc_id, source, chunks, entities, relations, mentions):
    tx.run(
        "MERGE (d:Document {id: $doc_id}) "
        "SET d.source = $source, d.ingested_at = datetime()",
        doc_id=doc_id, source=source,
    )

    # Prune chunks left over from a previous, longer version of this document.
    tx.run(
        "MATCH (:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
        "WHERE NOT c.id IN $chunk_ids DETACH DELETE c",
        doc_id=doc_id, chunk_ids=[c["id"] for c in chunks],
    )

    tx.run(
        "MATCH (d:Document {id: $doc_id}) "
        "UNWIND $chunks AS chunk "
        "MERGE (c:Chunk {id: chunk.id}) "
        "SET c.text = chunk.text, c.index = chunk.index "
        "MERGE (d)-[:HAS_CHUNK]->(c)",
        doc_id=doc_id, chunks=chunks,
    )

    # NEXT edges let retrieval widen to adjacent chunks.
    tx.run(
        "MATCH (:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) "
        "WITH c ORDER BY c.index "
        "WITH collect(c) AS cs "
        "UNWIND range(0, size(cs) - 2) AS i "
        "WITH cs[i] AS a, cs[i + 1] AS b "
        "MERGE (a)-[:NEXT]->(b)",
        doc_id=doc_id,
    )

    # Clear stale MENTIONS from a previous extraction.
    tx.run(
        "MATCH (:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)-[m:MENTIONS]->(:Entity) DELETE m",
        doc_id=doc_id,
    )

    if entities:
        tx.run(
            "UNWIND $entities AS entity "
            "MERGE (e:Entity {key: entity.key}) "
            "SET e.name = entity.name, e.type = entity.type, "
            "    e.mention_count = coalesce(e.mention_count, 0) + entity.mention_count",
            entities=entities,
        )

    if mentions:
        tx.run(
            "UNWIND $mentions AS mention "
            "MATCH (c:Chunk {id: mention.chunk_id}), (e:Entity {key: mention.key}) "
            "MERGE (c)-[:MENTIONS]->(e)",
            mentions=mentions,
        )

    if relations:
        # Verb kept as a property, not a dynamic relationship type (needs APOC).
        tx.run(
            "UNWIND $relations AS relation "
            "MATCH (s:Entity {key: relation.source_key}), "
            "      (t:Entity {key: relation.target_key}) "
            "MERGE (s)-[r:RELATES {type: relation.type}]->(t) "
            "SET r.doc_ids = CASE "
            "  WHEN r.doc_ids IS NULL THEN [$doc_id] "
            "  WHEN $doc_id IN r.doc_ids THEN r.doc_ids "
            "  ELSE r.doc_ids + $doc_id END",
            relations=relations, doc_id=doc_id,
        )
