from fastapi import FastAPI
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


@app.post("/process")
def process(req: ProcessRequest):
    text = extract_text(req.source_path)
    chunks = chunk_text(text, settings.chunk_size)
    vectors = embed_chunks(chunks)
    upsert(chunks, vectors)
    return {"chunks_ingested": len(chunks)}
