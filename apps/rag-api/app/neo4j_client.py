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

# Query-side entity linking: for when the question names an entity no chunk phrased closely enough.
_LINKED_ENTITIES = """
CALL db.index.fulltext.queryNodes('entity_name', $q) YIELD node, score
RETURN node.key AS key, node.name AS name, node.type AS type
ORDER BY score DESC
LIMIT $limit
"""

_NEIGHBOURHOOD = """
MATCH (e:Entity) WHERE e.key IN $keys
MATCH (e)-[r:RELATES]-(o:Entity)
WITH e, r, o.key IN $keys AS mutual
ORDER BY mutual DESC
WITH e, collect(r)[0..$per_entity] AS top
UNWIND top AS r
RETURN DISTINCT startNode(r).name AS source, r.type AS type, endNode(r).name AS target
LIMIT $limit
"""

# Hop 1: chunks mentioning an anchor entity. Hop 2: chunks reached via a RELATES neighbour —
# the point of having a graph, reaching chunks with no shared vocabulary with the question.
_GRAPH_CHUNKS = """
MATCH (e:Entity) WHERE e.key IN $keys
CALL {
  WITH e
  MATCH (c:Chunk)-[:MENTIONS]->(e)
  RETURN c, 1 AS hop, e AS via
  UNION
  WITH e
  MATCH (e)-[:RELATES]-(o:Entity) WHERE NOT o.key IN $keys
  MATCH (c:Chunk)-[:MENTIONS]->(o)
  RETURN c, 2 AS hop, o AS via
}
WITH c, hop, via, count{(:Chunk)-[:MENTIONS]->(via)} AS via_freq
// Hub entities are kept as edges (they are the cross-document bridges) but
// contribute less evidence per chunk: one shared mention of a topic named in
// a third of the corpus means far less than one shared mention of a specific
// entity. 1/(1+log10(freq)) discounts smoothly instead of cutting off.
WITH c, hop, via, via_freq, 1.0 / (1.0 + log10(toFloat(via_freq))) AS weight
WITH c, min(hop) AS hop, count(DISTINCT via) AS overlap,
     sum(weight) AS evidence, min(via_freq) AS rarest
RETURN c.text AS text, hop, overlap, rarest, evidence
ORDER BY evidence DESC, hop ASC
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
            return {"entities": [], "relations": [], "chunks": [], "chunk_meta": {}}

        key_list = list(keys)
        relations = [
            f"{r['source']} —{r['type']}→ {r['target']}"
            for r in s.run(_NEIGHBOURHOOD, keys=key_list,
                            limit=settings.graph_max_relations,
                            per_entity=settings.graph_relations_per_entity)
        ]
        rows = [dict(r) for r in s.run(_GRAPH_CHUNKS, keys=key_list,
                                       limit=settings.graph_max_chunks)]

    return {
        "entities": [f"{r['name']} ({r['type']})" for r in keys.values()],
        "relations": relations,
        "chunks": [r["text"] for r in rows],
        # Per-chunk provenance for reranking; same order as `chunks`.
        "chunk_meta": {r["text"]: {"hop": r["hop"], "overlap": r["overlap"],
                                   "rarest": r["rarest"], "evidence": r["evidence"]}
                       for r in rows},
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
