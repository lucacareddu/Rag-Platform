from openai import OpenAI
from .config import settings

client = OpenAI(base_url=settings.github_models_base_url, api_key=settings.github_token)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=settings.embed_model, input=chunks)
    return [d.embedding for d in resp.data]
