import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from .config import settings

client = QdrantClient(url=settings.qdrant_url)


def chunk_id(doc_id: str, index: int) -> str:
    """Deterministic id — the join key shared with Neo4j's Chunk nodes."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{index}"))


def ensure_collection(dim: int):
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert(doc_id: str, chunks: list[str], vectors: list[list[float]]) -> list[str]:
    """Replaces this document's points rather than appending — deterministic ids
    make re-ingest idempotent."""
    ensure_collection(len(vectors[0]))
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )
    ids = [chunk_id(doc_id, i) for i in range(len(chunks))]
    points = [
        PointStruct(
            id=pid,
            vector=v,
            payload={"text": c, "doc_id": doc_id, "chunk_index": i},
        )
        for i, (pid, c, v) in enumerate(zip(ids, chunks, vectors))
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return ids
