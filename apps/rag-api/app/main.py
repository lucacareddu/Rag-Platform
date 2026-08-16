from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import httpx
from .graph import rag_graph
from .config import settings

app = FastAPI(title="RAG API")


class Query(BaseModel):
    question: str


class IngestRequest(BaseModel):
    source_path: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: Query):
    result = rag_graph.invoke({"question": q.question, "context": "", "answer": ""})
    return {"answer": result["answer"]}


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Ingest a file already present on the ingestion-worker's filesystem."""
    r = httpx.post(f"{settings.ingestion_url}/process", json=req.dict(), timeout=120)
    r.raise_for_status()
    return r.json()


@app.post("/ingest/upload")
def ingest_upload(file: UploadFile = File(...)):
    """Ingest a file uploaded directly by the caller (e.g. from your laptop)."""
    files = {"file": (file.filename, file.file, file.content_type)}
    r = httpx.post(f"{settings.ingestion_url}/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()
