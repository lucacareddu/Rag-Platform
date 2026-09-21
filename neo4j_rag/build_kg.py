"""Arm B: rebuild the knowledge graph with Neo4j's own SimpleKGPipeline.

Arm A reused GraphRAG's graph, so it isolates retrieval. Arm B rebuilds the
graph from the same source documents with Neo4j's extractor, so it isolates
EXTRACTION -- and is therefore confounded with retrieval by construction. The
two answer different questions and are kept apart deliberately:

  A  same graph, different retriever   -> is Neo4j's retrieval better?
  B  same documents, different graph   -> is schema-guided extraction better?

The difference that makes B worth its cost: GraphRAG extracts open-endedly with
type hints, and the resulting graph has ORGANIATION and ORGANAIZATION next to
ORGANIZATION, a THREAT<VULNERABILITY, and 331 entities with an empty type.
Neo4j's extraction is schema-guided -- node types, relationship types and
allowed patterns are declared up front and the model is constrained to them --
so the type vocabulary should be clean by construction. Whether a cleaner,
smaller graph retrieves better is the actual question.

COST CONTROL. This is the only script in the Neo4j work that spends real money,
so it is built to fail cheaply:
  --limit N    process only N chunks (smoke test before the full run)
  --resume     skip documents already written, so a crash does not re-pay
Estimated full run: 542 chunks, ~1.46M input / ~0.43M output tokens, ~$0.26.
gpt-5-nano's constraints (temperature must be 1, reasoning_effort must be
minimal or output is empty while still billing) are set in retrievers.llm().

The schema below mirrors GraphRAG's entity_types so the two graphs are
comparable, minus the junk categories GraphRAG's open extraction invented.

Run: .venv-neo4j/bin/python neo4j_rag/build_kg.py --limit 3   # smoke test
     .venv-neo4j/bin/python neo4j_rag/build_kg.py             # full
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from retrievers import driver, embedder, llm

ROOT = Path(__file__).resolve().parents[1]
# The .txt files GraphRAG itself indexed, not the source PDFs. Using the same
# extracted text removes PDF parsing as a confound: arm A and arm B then differ
# only in how entities are extracted from identical characters.
DOCS = ROOT / "graphrag/input"

# Mirrors GraphRAG's extract_graph.entity_types so the graphs are comparable.
NODE_TYPES = ["Organization", "Person", "Framework", "Control", "Process",
              "Technology", "Threat", "Vulnerability", "Role", "Artifact",
              "Publication"]
RELATION_TYPES = ["DEFINES", "REQUIRES", "MITIGATES", "REFERENCES", "PART_OF",
                  "APPLIES_TO", "PUBLISHED_BY", "RELATED_TO"]

# GraphRAG used 1200 tokens / 100 overlap. FixedSizeSplitter counts characters,
# so ~4 chars per token keeps the chunking comparable rather than identical.
CHUNK_SIZE = 4800
CHUNK_OVERLAP = 400


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="process only N documents (smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip documents already present in the graph")
    args = ap.parse_args()

    files = sorted(DOCS.glob("*.txt"))
    if not files:
        sys.exit(f"no .txt documents under {DOCS}")

    d = driver()
    done = set()
    if args.resume:
        with d.session() as s:
            done = {r["p"] for r in s.run(
                "MATCH (n:Document) WHERE n.path IS NOT NULL RETURN n.path AS p")}
        print(f"resume: {len(done)} documents already built", file=sys.stderr)

    todo = [f for f in files if str(f) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"documents to process: {len(todo)} of {len(files)}", file=sys.stderr)

    pipeline = SimpleKGPipeline(
        llm=llm(3000), driver=d, embedder=embedder(),
        # additional_relationship_types MUST be True here. The field defaults to
        # len(relationship_types) == 0, so declaring a relationship vocabulary
        # silently sets it False and GraphPruning then deletes every edge whose
        # type is not in that list -- and every node orphaned by those
        # deletions. Declaring 8 invented types took a chunk that extracted
        # cleanly as 12 nodes / 21 relationships down to 1 node / 0
        # relationships, with on_error="IGNORE" hiding it.
        #
        # Constraining nodes but not edges is also the faithful analogue of the
        # arm A graph: GraphRAG constrains entity_types and leaves
        # relationships as free-text descriptions with no type vocabulary at
        # all. additional_node_types stays False, which is the schema-guided
        # cleanliness arm B exists to test.
        schema={"node_types": NODE_TYPES,
                "relationship_types": RELATION_TYPES,
                "additional_node_types": False,
                "additional_relationship_types": True},
        from_pdf=False,
        text_splitter=FixedSizeSplitter(chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVERLAP),
        perform_entity_resolution=True,
        on_error="IGNORE",
    )

    t0 = time.time()
    for i, f in enumerate(todo, 1):
        text = f.read_text(errors="ignore")
        if args.limit:
            # Smoke test: one chunk's worth, so a wiring error costs cents.
            text = text[:CHUNK_SIZE]
        t = time.time()
        try:
            await pipeline.run_async(text=text)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] {f.name}: FAILED {str(e)[:150]}",
                  file=sys.stderr)
            continue

        # Stamp the source path onto the Document nodes this run just created.
        # SimpleKGPipeline does not record it for text= input, so without this
        # --resume can never match anything and a crash halfway through costs
        # the whole run again.
        with d.session() as s:
            s.run("MATCH (n:Document) WHERE n.path IS NULL SET n.path = $p",
                  p=str(f))
        print(f"  [{i}/{len(todo)}] {f.name}  {len(text):>8,} chars  "
              f"{time.time() - t:>6.1f}s", file=sys.stderr)

    print(f"\ntotal {time.time() - t0:.1f}s", file=sys.stderr)
    stats(d)
    d.close()


def stats(d):
    print("\narm B graph:", file=sys.stderr)
    with d.session() as s:
        for q, label in [
            ("MATCH (n:Document) RETURN count(n) AS c", "Document"),
            ("MATCH (n:Chunk) RETURN count(n) AS c", "Chunk"),
            ("MATCH (n:__Entity__) RETURN count(n) AS c", "__Entity__"),
            # Scoped to arm B's own entities. An earlier version counted every
            # relationship in the database and reported 17,759 for a two-chunk
            # smoke test -- it was counting arm A's 9,188 MENTIONED_IN and
            # 3,248 RELATED, which share the graph.
            ("MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS c",
             "entity rels"),
        ]:
            try:
                print(f"  {label:<16}{s.run(q).single()['c']:>7}", file=sys.stderr)
            except Exception as e:
                print(f"  {label:<16}  ? {str(e)[:60]}", file=sys.stderr)
        print("  entity labels:", file=sys.stderr)
        for r in s.run("MATCH (n:__Entity__) UNWIND labels(n) AS l "
                       "WITH l WHERE l <> '__Entity__' AND l <> '__KGBuilder__' "
                       "RETURN l, count(*) AS c ORDER BY c DESC LIMIT 20"):
            print(f"    {r['l']:<24}{r['c']:>6}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
