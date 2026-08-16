from openai import OpenAI
from .config import settings

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=settings.embed_model, input=chunks)
    return [d.embedding for d in resp.data]
