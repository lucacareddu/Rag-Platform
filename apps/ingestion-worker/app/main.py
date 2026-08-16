from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from .ocr import extract_text, chunk_text
from .embed import embed_chunks
from .qdrant_client import upsert
from .config import settings

app = FastAPI(title="Ingestion Worker")


class ProcessRequest(BaseModel):
    source_path: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _ingest(path: str) -> dict:
    text = extract_text(path)
    chunks = chunk_text(text, settings.chunk_size)
    vectors = embed_chunks(chunks)
    upsert(chunks, vectors)
    return {"chunks_ingested": len(chunks)}


@app.post("/process")
def process(req: ProcessRequest):
    """Ingest a file already present on this pod's filesystem (e.g. mounted volume)."""
    return _ingest(req.source_path)


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Ingest a file uploaded directly from the caller — no shared volume needed."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        return _ingest(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
