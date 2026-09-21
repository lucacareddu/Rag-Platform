"""Five Neo4j retrievers over the loaded GraphRAG graph (arm A): vector (control), entity_cypher
and entity_expand (entity-anchored traversal), hybrid, and text2cypher (exact enumeration).
"""
import os
import subprocess
from pathlib import Path

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import AzureOpenAIEmbeddings
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.retrievers import (HybridRetriever, Text2CypherRetriever,
                                       VectorCypherRetriever, VectorRetriever)

ROOT = Path(__file__).resolve().parents[1]
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")

IDX_ENTITY = "gr_entity_embedding"
IDX_CHUNK = "gr_chunk_embedding"
FT_CHUNK = "gr_chunk_text"


def password() -> str:
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


def _env() -> dict:
    return dict(l.split("=", 1) for l in (ROOT / "graphrag/.env").read_text().splitlines()
                if "=" in l and not l.startswith("#"))


def embedder() -> AzureOpenAIEmbeddings:
    """Must be the same deployment that produced the stored vectors, or the
    query vector is not comparable to them and every score is meaningless."""
    v = _env()
    return AzureOpenAIEmbeddings(
        model=v["GRAPHRAG_EMBED_DEPLOYMENT"],
        azure_endpoint=v["GRAPHRAG_EMBED_API_BASE"],
        api_key=v["GRAPHRAG_EMBED_API_KEY"],
        api_version=v["GRAPHRAG_EMBED_API_VERSION"])


def llm(max_tokens: int = 3000) -> AzureOpenAILLM:
    """gpt-5-nano rejects any temperature but 1, and without
    reasoning_effort=minimal it spends the whole budget on hidden reasoning and
    returns an empty string while still billing for it."""
    v = _env()
    return AzureOpenAILLM(
        model_name=v["GRAPHRAG_DEPLOYMENT"],
        azure_endpoint=v["GRAPHRAG_API_BASE"], api_key=v["GRAPHRAG_API_KEY"],
        api_version=v["GRAPHRAG_API_VERSION"],
        model_params={"temperature": 1, "reasoning_effort": "minimal",
                      "max_completion_tokens": max_tokens})


def driver():
    return GraphDatabase.driver(URI, auth=("neo4j", password()))


# Dedup to chunks, ranked by best entity score -- multiple matching entities rank a chunk higher.
ENTITY_TO_CHUNKS = """
WITH node AS e, score
MATCH (e)-[:MENTIONED_IN]->(c:GRChunk)
WITH c, max(score) AS s, collect(DISTINCT e.title)[..5] AS via
RETURN c.text AS text, s AS score,
       {chunk_id: c.id, via_entities: via} AS metadata
ORDER BY s DESC
LIMIT %d
"""

# One RELATED hop before collecting chunks; neighbour score discounted so direct matches rank higher.
ENTITY_EXPAND_TO_CHUNKS = """
WITH node AS e, score
OPTIONAL MATCH (e)-[:RELATED]-(n:GREntity)
WHERE n IS NOT NULL
WITH e, score, collect(DISTINCT n)[..10] AS nbrs
UNWIND ([{n: e, w: 1.0}] + [x IN nbrs | {n: x, w: 0.5}]) AS hop
WITH hop.n AS ent, score * hop.w AS s
MATCH (ent)-[:MENTIONED_IN]->(c:GRChunk)
WITH c, max(s) AS s, collect(DISTINCT ent.title)[..5] AS via
RETURN c.text AS text, s AS score,
       {chunk_id: c.id, via_entities: via} AS metadata
ORDER BY s DESC
LIMIT %d
"""

# Written out rather than introspected, so the model sees only the GR* namespace, not arm B's.
SCHEMA = """
Node properties:
GRDocument {id: STRING, title: STRING}
GRChunk {id: STRING, text: STRING, n_tokens: INTEGER}
GREntity {id: STRING, title: STRING, type: STRING, description: STRING,
          degree: INTEGER, frequency: INTEGER}
GRCommunity {id: STRING, community: INTEGER, level: INTEGER, title: STRING,
             summary: STRING, full_content: STRING, rank: FLOAT}

Relationships:
(:GRChunk)-[:PART_OF]->(:GRDocument)
(:GREntity)-[:MENTIONED_IN]->(:GRChunk)
(:GREntity)-[:RELATED {description: STRING, weight: FLOAT}]->(:GREntity)
(:GRCommunity)-[:HAS_ENTITY]->(:GREntity)

IMPORTANT data facts (verified against the database):
- GREntity.type is UPPERCASE. Common values: PROCESS, ARTIFACT, PUBLICATION,
  CONTROL, FRAMEWORK, ORGANIZATION, TECHNOLOGY, THREAT, PERSON, ROLE,
  VULNERABILITY. Some rows have misspelled or empty types, so prefer matching
  on title/description text over filtering by type when the question is
  topical.
- GREntity.title is UPPERCASE, e.g. 'ZERO TRUST ARCHITECTURE'. Use
  toUpper() or case-insensitive regex (=~ '(?i).*term.*') when matching.
- GRDocument.title is a FILENAME, e.g.
  'SP800-207_Zero_Trust_Architecture.txt'. Match with CONTAINS on a fragment
  such as 'SP800-207', never equality.
- There are exactly 11 GRDocument nodes.

Query rules:
- Return literal values only. NEVER use query parameters such as $query or
  $name; inline every value into the Cypher.
- To find which publications cover a concept: match GREntity on title or
  description, then traverse MENTIONED_IN and PART_OF to GRDocument, and
  return DISTINCT d.title.
"""


def build(names=None, top_k: int = 20) -> dict:
    """Retrievers only. No generation is wired here, so calling these costs one
    embedding per query (or one chat call for text2cypher)."""
    d, emb = driver(), embedder()
    all_r = {
        "vector": lambda: VectorRetriever(
            d, index_name=IDX_CHUNK, embedder=emb,
            return_properties=["id", "text"]),
        # LIMIT bound to top_k so every arm returns the same count -- else the score reflects budget.
        "entity_cypher": lambda: VectorCypherRetriever(
            d, index_name=IDX_ENTITY, embedder=emb,
            retrieval_query=ENTITY_TO_CHUNKS % top_k),
        "entity_expand": lambda: VectorCypherRetriever(
            d, index_name=IDX_ENTITY, embedder=emb,
            retrieval_query=ENTITY_EXPAND_TO_CHUNKS % top_k),
        "hybrid": lambda: HybridRetriever(
            d, vector_index_name=IDX_CHUNK, fulltext_index_name=FT_CHUNK,
            embedder=emb, return_properties=["id", "text"]),
        "text2cypher": lambda: Text2CypherRetriever(
            d, llm=llm(1500), neo4j_schema=SCHEMA),
    }
    chosen = names or list(all_r)
    return {n: all_r[n]() for n in chosen}, d
