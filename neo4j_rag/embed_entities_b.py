"""Embed arm B's entities so the entity-anchored path can be compared. ~$0.001.

SimpleKGPipeline embeds chunks but not entities, so arm B has no entity vector
index and the entity-anchored retrieval that arm A was tested on cannot run
against it.

An asymmetry to keep in view when reading the comparison, because it favours
arm A and is not something this script can fix: arm A's entity embeddings are
computed over GraphRAG's LLM-generated entity DESCRIPTIONS, a stage that cost
$0.17 during indexing. Arm B's entities carry only a name -- Neo4j's extractor
stores no description -- so the best available text here is "Label: name". A
bare name is a weaker retrieval signal than a paragraph of description, and
any gap in arm B's favour would be despite that, not because of it.

1,991 entities at roughly 20 tokens each is about 40k tokens, well under a cent
on text-embedding-3-small. The embedder is the same deployment used everywhere
else, so vectors stay comparable across arms.

Run: .venv-neo4j/bin/python neo4j_rag/embed_entities_b.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrievers import ROOT, driver  # noqa: E402

IDX_B_ENTITY = "b_entity_embedding"
BATCH = 256


def batch_embedder():
    """neo4j-graphrag's wrapper exposes only embed_query (one text per call),
    which would mean 1,991 sequential round-trips. The underlying Azure client
    takes a list, so use it directly -- same deployment, same vectors."""
    from openai import AzureOpenAI
    v = dict(l.split("=", 1) for l in (ROOT / "graphrag/.env").read_text().splitlines()
             if "=" in l and not l.startswith("#"))
    client = AzureOpenAI(azure_endpoint=v["GRAPHRAG_EMBED_API_BASE"],
                         api_key=v["GRAPHRAG_EMBED_API_KEY"],
                         api_version=v["GRAPHRAG_EMBED_API_VERSION"])
    deployment = v["GRAPHRAG_EMBED_DEPLOYMENT"]

    def go(texts):
        r = client.embeddings.create(model=deployment, input=texts)
        return [item.embedding for item in r.data]
    return go


def main():
    d, emb = driver(), batch_embedder()
    with d.session() as s:
        rows = [(r["id"], r["label"], r["name"]) for r in s.run("""
            MATCH (e:__Entity__)
            WHERE e.embedding IS NULL AND e.name IS NOT NULL
            RETURN elementId(e) AS id, e.name AS name,
                   [l IN labels(e) WHERE l <> '__Entity__'
                    AND l <> '__KGBuilder__'][0] AS label
        """)]
        print(f"entities to embed: {len(rows)}", file=sys.stderr)
        if not rows:
            print("nothing to do", file=sys.stderr)

        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            texts = [f"{lbl or 'Entity'}: {name}" for _, lbl, name in batch]
            vecs = emb(texts)
            s.run("""
                UNWIND $rows AS r
                MATCH (e) WHERE elementId(e) = r.id
                SET e.embedding = r.v
            """, rows=[{"id": eid, "v": v}
                       for (eid, _, _), v in zip(batch, vecs)])
            print(f"  {min(i + BATCH, len(rows))}/{len(rows)}", file=sys.stderr)

        s.run(f"""CREATE VECTOR INDEX {IDX_B_ENTITY} IF NOT EXISTS
                  FOR (n:__Entity__) ON (n.embedding)
                  OPTIONS {{indexConfig: {{
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'}}}}""")

        n = s.run("MATCH (e:__Entity__) WHERE e.embedding IS NULL "
                  "RETURN count(e) AS c").single()["c"]
        state = s.run("SHOW INDEXES YIELD name, state WHERE name = $n "
                      "RETURN state", n=IDX_B_ENTITY).single()
        print(f"\nentities still without embedding: {n}", file=sys.stderr)
        print(f"index {IDX_B_ENTITY}: {state['state'] if state else 'MISSING'}",
              file=sys.stderr)
    d.close()


if __name__ == "__main__":
    main()
