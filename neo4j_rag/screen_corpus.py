"""Screen a corpus for GraphRAG fit before paying to index it (~$0.02): sample chunks, extract
entity names cheaply, measure cross-document recurrence. Compare scores relatively, not absolutely.
"""
import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrievers import llm  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval/results_corpus_screen.json"

CHUNK_CHARS = 4800
PER_DOC = 6

PROMPT = """Extract the named entities from this text.

Return ONLY a JSON array of strings, no other text. Each string is one entity
name: organizations, people, systems, technologies, standards, processes,
products, places. Normalise to the canonical name (expand pronouns, drop
articles). Return at most 25 entities. If there are none, return [].

TEXT:
{text}

JSON array:"""

# Would inflate recurrence for every corpus without indicating a connected graph.
_STOP = {"", "n/a", "none", "unknown", "the", "it", "this", "that", "figure",
         "table", "section", "appendix", "introduction", "conclusion"}


def normalise(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name)).strip().lower()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    n = re.sub(r"[\.,;:'\"]+$", "", n)
    return n


def sample_chunks(docs: dict, per_doc: int) -> list:
    """(doc_name, chunk_text), spread evenly through each document.

    Evenly rather than from the front: the opening chunks of these documents
    are cover pages and boilerplate, which share entities across every
    publication and would fake recurrence.
    """
    out = []
    for name, text in docs.items():
        n = max(1, len(text) // CHUNK_CHARS)
        take = min(per_doc, n)
        if take == 0:
            continue
        # Skip the first 10% and last 5%: front matter and back matter.
        lo, hi = int(len(text) * 0.10), int(len(text) * 0.95)
        span = max(1, hi - lo)
        for i in range(take):
            start = lo + (span * i) // take
            out.append((name, text[start:start + CHUNK_CHARS]))
    return out


def extract(model, item):
    name, text = item
    try:
        raw = model.invoke(PROMPT.format(text=text)).content
    except Exception as e:
        print(f"    extract failed: {str(e)[:90]}", file=sys.stderr)
        return name, []
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return name, []
    try:
        names = json.loads(m.group(0))
    except json.JSONDecodeError:
        return name, []
    ents = {normalise(x) for x in names if isinstance(x, str)}
    return name, sorted(e for e in ents if e not in _STOP and len(e) > 2)


def screen(docs: dict, label: str, workers: int = 8) -> dict:
    items = sample_chunks(docs, PER_DOC)
    print(f"\n[{label}] {len(docs)} documents, {len(items)} sampled chunks",
          file=sys.stderr)
    model = llm(1200)

    ent_docs, ent_chunks = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for doc_name, ents in pool.map(lambda it: extract(model, it), items):
            for e in ents:
                ent_docs.setdefault(e, set()).add(doc_name)
                ent_chunks[e] = ent_chunks.get(e, 0) + 1

    if not ent_docs:
        return {"label": label, "error": "no entities extracted"}

    counts = [len(v) for v in ent_docs.values()]
    multi = sum(1 for c in counts if c >= 2)
    top = sorted(ent_docs.items(), key=lambda kv: -len(kv[1]))[:12]

    r = {
        "label": label,
        "documents": len(docs),
        "sampled_chunks": len(items),
        "distinct_entities": len(ent_docs),
        "multi_doc_entities": multi,
        "multi_doc_pct": round(100 * multi / len(ent_docs), 1),
        "mean_docs_per_entity": round(statistics.mean(counts), 2),
        "max_docs_for_one_entity": max(counts),
        "entities_per_chunk": round(sum(ent_chunks.values()) / len(items), 1),
        "top_recurring": [{"entity": e, "docs": len(d)} for e, d in top],
    }
    return r


def load_dir(path: Path) -> dict:
    files = sorted([p for p in path.rglob("*") if p.suffix.lower() in
                    (".txt", ".md")])
    return {f.name: f.read_text(errors="ignore") for f in files
            if f.stat().st_size > 2000}


def load_split(path: Path, parts: int = 11) -> dict:
    """Positive control: one document cut into pseudo-documents.

    Its entities necessarily recur across the pieces, so a screener that cannot
    separate this from a genuinely disconnected corpus is measuring noise.
    """
    text = path.read_text(errors="ignore")
    size = len(text) // parts
    return {f"{path.stem}__part{i:02d}": text[i * size:(i + 1) * size]
            for i in range(parts)}


def report(results: list):
    print("\n" + "=" * 78)
    print("GRAPHRAG CORPUS FIT SCREEN")
    print("=" * 78)
    # Sample size shown alongside the score: fewer chunks scores a corpus more harshly.
    print(f"  {'corpus':<22}{'docs':>6}{'chunks':>8}{'entities':>10}"
          f"{'multi-doc':>12}{'mean docs/ent':>15}")
    for r in results:
        if r.get("error"):
            print(f"  {r['label']:<22}  {r['error']}")
            continue
        pct = f"{r['multi_doc_pct']}%"
        print(f"  {r['label']:<22}{r['documents']:>6}{r['sampled_chunks']:>8}"
              f"{r['distinct_entities']:>10}{pct:>12}"
              f"{r['mean_docs_per_entity']:>15.2f}")

    for r in results:
        if r.get("error"):
            continue
        print(f"\n  [{r['label']}] most-recurring entities:")
        for t in r["top_recurring"][:6]:
            print(f"    {t['entity'][:52]:<54} {t['docs']} docs")

    print("\n  Read the columns relative to each other, not against a fixed bar.")
    print("  A 6-chunk-per-document sample under-counts recurrence, so what is")
    print("  informative is the gap between a candidate and the references.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", help="directory of .txt/.md files")
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--split-file", help="positive control: split one file")
    ap.add_argument("--parts", type=int, default=11)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    results = json.loads(OUT.read_text()) if OUT.exists() else []

    if args.split_file:
        docs = load_split(Path(args.split_file), args.parts)
    else:
        if not args.corpus:
            sys.exit("give a corpus directory or --split-file")
        docs = load_dir(Path(args.corpus))
    if not docs:
        sys.exit("no documents found")

    r = screen(docs, args.label, args.workers)
    results = [x for x in results if x.get("label") != args.label] + [r]
    OUT.write_text(json.dumps(results, indent=2))
    report(results)


if __name__ == "__main__":
    main()
