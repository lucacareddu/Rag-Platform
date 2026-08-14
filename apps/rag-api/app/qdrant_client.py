from qdrant_client import QdrantClient
from .config import settings

client = QdrantClient(url=settings.qdrant_url)

def search(vector, top_k: int = 5):
    return client.search(collection_name=settings.qdrant_collection, query_vector=vector, limit=top_k)
