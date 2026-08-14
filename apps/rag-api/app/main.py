from fastapi import FastAPI
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
    r = httpx.post(f"{settings.ingestion_url}/process", json=req.dict(), timeout=120)
    r.raise_for_status()
    return r.json()
