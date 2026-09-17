from openai import OpenAI
from .config import settings

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embeds in batches — the API caps inputs per request."""
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), settings.embed_batch_size):
        batch = chunks[start:start + settings.embed_batch_size]
        resp = client.embeddings.create(model=settings.embed_model, input=batch)
        # Response order isn't guaranteed to match input order.
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors
