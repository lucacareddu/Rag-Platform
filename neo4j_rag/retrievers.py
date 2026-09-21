"""Neo4j retrievers over the loaded GraphRAG graph (arm A).

Five strategies. The first is a control, the middle three are the ones with
something to prove, and the last is a capability neither GraphRAG arm has.

  vector          VectorRetriever over GRChunk. Plain vector RAG over the same
                  542 chunks and the same 1536-d embeddings GraphRAG's basic
                  arm searched. It should roughly reproduce basic's numbers; if
                  it does not, the load or the query embedder is wrong, and
                  that is worth knowing before trusting anything else here.

  entity_cypher   VectorCypherRetriever: vector search over ENTITY DESCRIPTION
                  embeddings, then traverse MENTIONED_IN to the chunks those
                  entities appear in. This is the arm with a real hypothesis
                  behind it. GraphRAG's local search anchors on the same entity
                  embeddings and still hit 0 of 18 gold units, because its
                  traversal is fixed and its 12k budget is split across
                  entities, relationships and community reports, leaving few
                  slots for source text. Here the traversal is ours and the
                  budget goes entirely to chunks.

  entity_expand   As above plus one hop along RELATED before collecting chunks.
                  Tests whether neighbour entities pull in the second document
                  on cross-document questions, where vector recall collapsed
                  from 0.750 to 0.267.

  hybrid          HybridRetriever: vector plus the fulltext index. Cheap
                  insurance for questions that hinge on an exact term the
                  embedding smooths over.

  text2cypher     Text2CypherRetriever. Translates the question to Cypher and
                  runs it. This is the only strategy here that can ANSWER an
                  enumeration question exactly -- "how many of these
                  publications discuss X" -- because it aggregates over the
                  graph instead of sampling top-k. Vector search structurally
                  cannot do this and global search only approximates it.

Cost: retrieval is one embedding call per query (text2cypher is one chat call
instead). Generation is separate and opt-in, so retrieval quality can be scored
against gold chunks for effectively nothing.
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


# Entity hits are deduplicated to chunks and ranked by the best entity score
# that reached them, so a chunk surfaced by several matching entities ranks
# above one reached by a single weak match.
ENTITY_TO_CHUNKS = """
WITH node AS e, score
MATCH (e)-[:MENTIONED_IN]->(c:GRChunk)
WITH c, max(score) AS s, collect(DISTINCT e.title)[..5] AS via
RETURN c.text AS text, s AS score,
       {chunk_id: c.id, via_entities: via} AS metadata
ORDER BY s DESC
LIMIT %d
"""

# One hop along RELATED before collecting chunks. The neighbour's own score is
# discounted so a directly matched entity still outranks a neighbour of one.
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

# The schema handed to Text2Cypher. Written out rather than introspected so the
# model sees the GR* namespace and not arm B's labels once both are loaded.
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
        # The chunk LIMIT is bound to top_k so every arm returns the same
        # number of chunks; leaving it hardcoded made the Cypher arms return 20
        # while vector returned top_k, which would have scored budget, not
        # retrieval.
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
