"""Recover arm B's document provenance. No Azure calls.

SimpleKGPipeline sets Document.path = 'document.txt' for text= input, a
constant. build_kg.py tried to stamp the real source path afterwards with
`WHERE n.path IS NULL`, which never matched because the property was already
set to that default. All 11 Document nodes therefore carried the same name and
every chunk looked like it came from the same file.

This is not cosmetic. eval_ab.py scores a retrieved chunk as hitting a gold
anchor only when it comes from the right document, so identical titles scored
arm B 0/20 while its retrieval was in fact working -- the third time in this
experiment that a scoring defect has imitated total retrieval failure.

Recovery is exact and free: each chunk's text is a verbatim substring of
exactly one source file, so matching one chunk per Document identifies it. The
match is verified against all 11 files and refuses to guess when a chunk is
ambiguous or absent.

Run: .venv-neo4j/bin/python neo4j_rag/fix_doc_provenance.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrievers import driver  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "graphrag/input"


def main():
    sources = {f.name: f.read_text(errors="ignore") for f in sorted(DOCS.glob("*.txt"))}
    print(f"source files: {len(sources)}", file=sys.stderr)

    d = driver()
    fixed, failed = 0, []
    with d.session() as s:
        docs = [r["id"] for r in s.run(
            "MATCH (n:Document) RETURN elementId(n) AS id")]
        print(f"Document nodes: {len(docs)}", file=sys.stderr)

        for did in docs:
            # A few chunks, not one: the first chunk of a document is often
            # boilerplate (cover page, NIST disclaimer) that appears verbatim
            # in several publications and would match ambiguously.
            chunks = [r["t"] for r in s.run(
                "MATCH (n:Document)<-[:FROM_DOCUMENT]-(c:Chunk) "
                "WHERE elementId(n) = $id RETURN c.text AS t LIMIT 12", id=did)]
            votes = {}
            for t in chunks:
                probe = t[200:700] if len(t) > 900 else t[:400]
                if not probe.strip():
                    continue
                hits = [name for name, body in sources.items() if probe in body]
                if len(hits) == 1:
                    votes[hits[0]] = votes.get(hits[0], 0) + 1

            if not votes:
                failed.append(did)
                continue
            name = max(votes, key=votes.get)
            s.run("MATCH (n:Document) WHERE elementId(n) = $id SET n.path = $p",
                  id=did, p=name)
            print(f"  {name:<62} votes={votes[name]}/{len(chunks)}", file=sys.stderr)
            fixed += 1

        print(f"\nfixed {fixed}/{len(docs)}", file=sys.stderr)
        if failed:
            print(f"UNRESOLVED: {failed}", file=sys.stderr)

        distinct = s.run("MATCH (n:Document) RETURN count(DISTINCT n.path) AS c"
                         ).single()["c"]
        print(f"distinct Document.path values: {distinct} "
              f"(must be {len(docs)})", file=sys.stderr)
    d.close()


if __name__ == "__main__":
    main()
