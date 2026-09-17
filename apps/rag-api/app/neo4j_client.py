import re

from neo4j import GraphDatabase
from .config import settings

driver = GraphDatabase.driver(
    settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
)

# Entities mentioned by the chunks vector search already found.
_ANCHOR_ENTITIES = """
MATCH (c:Chunk) WHERE c.id IN $chunk_ids
MATCH (c)-[:MENTIONS]->(e:Entity)
RETURN DISTINCT e.key AS key, e.name AS name, e.type AS type
"""

# Query-side entity linking: the other direction into the graph, for when the
# question names an entity but no chunk phrased it similarly enough to rank.
_LINKED_ENTITIES = """
CALL db.index.fulltext.queryNodes('entity_name', $q) YIELD node, score
RETURN node.key AS key, node.name AS name, node.type AS type
ORDER BY score DESC
LIMIT $limit
"""

_NEIGHBOURHOOD = """
MATCH (e:Entity) WHERE e.key IN $keys
MATCH (e)-[r:RELATES]-(:Entity)
RETURN DISTINCT startNode(r).name AS source, r.type AS type, endNode(r).name AS target
LIMIT $limit
"""

# Ranked by entity overlap; deliberately includes chunks vector search already
# found so workflow.py's RRF fusion can sum their scores as a consensus signal.
_GRAPH_CHUNKS = """
MATCH (e:Entity) WHERE e.key IN $keys
MATCH (c:Chunk)-[:MENTIONS]->(e)
WITH c, count(DISTINCT e) AS overlap
RETURN c.text AS text, overlap
ORDER BY overlap DESC, c.index ASC
LIMIT $limit
"""


def expand(chunk_ids: list[str], question: str) -> dict:
    """Walks from the vector hits' entities into the graph, returning connected
    chunks (may overlap with vector hits) and the relevant entity/relation subgraph."""
    with driver.session(database=settings.neo4j_database) as s:
        keys = {r["key"]: r for r in s.run(_ANCHOR_ENTITIES, chunk_ids=chunk_ids)}

        lucene = _lucene_query(question)
        if lucene:
            for r in s.run(_LINKED_ENTITIES, q=lucene, limit=settings.graph_linked_entities):
                keys.setdefault(r["key"], r)

        if not keys:
            return {"entities": [], "relations": [], "chunks": []}

        key_list = list(keys)
        relations = [
            f"{r['source']} —{r['type']}→ {r['target']}"
            for r in s.run(_NEIGHBOURHOOD, keys=key_list, limit=settings.graph_max_relations)
        ]
        chunks = [
            r["text"]
            for r in s.run(_GRAPH_CHUNKS, keys=key_list, limit=settings.graph_max_chunks)
        ]

    return {
        "entities": [f"{r['name']} ({r['type']})" for r in keys.values()],
        "relations": relations,
        "chunks": chunks,
    }


def _lucene_query(question: str) -> str:
    """Quotes each term so Lucene operators (AND/OR/NOT/TO) in the question can't
    be parsed as syntax."""
    terms = [t for t in re.findall(r"\w+", question) if len(t) > 2]
    return " OR ".join(f'"{t}"' for t in terms[:16])


def stats() -> dict:
    with driver.session(database=settings.neo4j_database) as s:
        r = s.run(
            "MATCH (d:Document) WITH count(d) AS documents "
            "MATCH (c:Chunk) WITH documents, count(c) AS chunks "
            "MATCH (e:Entity) WITH documents, chunks, count(e) AS entities "
            "OPTIONAL MATCH ()-[rel:RELATES]->() "
            "RETURN documents, chunks, entities, count(rel) AS relations"
        ).single()
    return dict(r) if r else {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}
