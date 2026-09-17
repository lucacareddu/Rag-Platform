import hashlib
import logging
from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from .ocr import extract_text, chunk_text
from .embed import embed_chunks
from .graph_extraction import extract_entities_and_relations
from .neo4j_client import write_document
from .qdrant_client import upsert
from .config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Ingestion Worker")

DOCUMENTS_DIR = Path("/data")
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt", ".md"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _doc_id(path: str) -> str:
    """Content hash, not the filename, so re-ingesting the same file is idempotent."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ingest(path: str, source: str | None = None) -> dict:
    text = extract_text(path)
    chunks = chunk_text(text, settings.chunk_size)
    doc_id = _doc_id(path)
    vectors = embed_chunks(chunks)
    chunk_ids = upsert(doc_id, chunks, vectors)

    entities, relations = [], []
    if settings.graph_enabled:
        # Best-effort: a Neo4j/extraction failure must not lose the vectors already written.
        try:
            entities, relations = extract_entities_and_relations(text)
            write_document(
                doc_id,
                source or Path(path).name,
                [{"id": cid, "index": i, "text": c}
                 for i, (cid, c) in enumerate(zip(chunk_ids, chunks))],
                entities,
                relations,
            )
        except Exception as e:
            logger.warning("Graph build failed for %s (%s) — vectors ingested anyway", source or path, e)
            entities, relations = [], []

    return {"chunks": len(chunks), "entities": len(entities), "relations": len(relations)}


@app.post("/process")
def process():
    """Batch-ingest every supported file under the mounted documents volume."""
    if not DOCUMENTS_DIR.is_dir():
        raise HTTPException(404, f"{DOCUMENTS_DIR} does not exist or isn't mounted")

    results = {}
    entities = {}
    errors = {}
    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = str(path.relative_to(DOCUMENTS_DIR))
        try:
            outcome = _ingest(str(path), source=rel)
            results[rel] = outcome["chunks"]
            entities[rel] = outcome["entities"]
        except Exception as e:
            errors[rel] = str(e)

    return {
        "files_ingested": len(results),
        "chunks_per_file": results,
        "entities_per_file": entities,
        "files_failed": errors,
    }


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Ingest a file uploaded directly from the caller — no shared volume needed."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        outcome = _ingest(tmp_path, source=file.filename)
        return {
            "chunks_ingested": outcome["chunks"],
            "entities_extracted": outcome["entities"],
            "relations_extracted": outcome["relations"],
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
