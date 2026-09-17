from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import httpx
from .workflow import rag_workflow
from .neo4j_client import stats as graph_stats
from .config import settings

app = FastAPI(title="RAG API")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph/stats")
def graph_stats_endpoint():
    """Node/relationship counts in Neo4j, to confirm ingestion is populating the graph."""
    if not settings.graph_enabled:
        return {"enabled": False}
    try:
        return {"enabled": True, **graph_stats()}
    except Exception as e:
        return {"enabled": True, "reachable": False, "error": str(e)}


@app.post("/query")
def query(q: Query):
    result = rag_workflow.invoke(
        {"question": q.question, "chunk_ids": [], "vector_chunks": [],
         "graph_chunks": [], "graph_context": "", "context": "", "answer": ""}
    )
    return {"answer": result["answer"]}


@app.post("/ingest")
def ingest():
    """Batch-ingest every file dropped into the ingestion-worker's mounted /data volume."""
    r = httpx.post(f"{settings.ingestion_url}/process", timeout=600)
    r.raise_for_status()
    return r.json()


@app.post("/ingest/upload")
def ingest_upload(file: UploadFile = File(...)):
    """Ingest a file uploaded directly by the caller (e.g. from your laptop)."""
    files = {"file": (file.filename, file.file, file.content_type)}
    r = httpx.post(f"{settings.ingestion_url}/upload", files=files, timeout=120)
    r.raise_for_status()
    return r.json()
