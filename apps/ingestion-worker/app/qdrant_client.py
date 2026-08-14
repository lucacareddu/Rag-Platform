import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from .config import settings

client = QdrantClient(url=settings.qdrant_url)


def ensure_collection(dim: int):
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert(chunks: list[str], vectors: list[list[float]]):
    ensure_collection(len(vectors[0]))
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v, payload={"text": c})
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
