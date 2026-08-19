from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from .ocr import extract_text, chunk_text
from .embed import embed_chunks
from .qdrant_client import upsert
from .config import settings

app = FastAPI(title="Ingestion Worker")

DOCUMENTS_DIR = Path("/data")
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt", ".md"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _ingest(path: str) -> int:
    text = extract_text(path)
    chunks = chunk_text(text, settings.chunk_size)
    vectors = embed_chunks(chunks)
    upsert(chunks, vectors)
    return len(chunks)


@app.post("/process")
def process():
    """Batch-ingest every supported file under the mounted documents volume
    (hostPath: /data on the pod, configurable via ingestionWorker.hostPath on the
    k3s node — see gitops/README.md). Walks subdirectories too."""
    if not DOCUMENTS_DIR.is_dir():
        raise HTTPException(404, f"{DOCUMENTS_DIR} does not exist or isn't mounted")

    results = {}
    errors = {}
    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = str(path.relative_to(DOCUMENTS_DIR))
        try:
            results[rel] = _ingest(str(path))
        except Exception as e:
            errors[rel] = str(e)

    return {
        "files_ingested": len(results),
        "chunks_per_file": results,
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
        return {"chunks_ingested": _ingest(tmp_path)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
