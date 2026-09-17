from openai import OpenAI
from .config import settings

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embeds in batches — the API caps inputs per request."""
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), settings.embed_batch_size):
        batch = chunks[start:start + settings.embed_batch_size]
        resp = client.embeddings.create(model=settings.embed_model, input=batch)
        # Response order isn't guaranteed to match input order, but Gemini's
        # endpoint returns .index as None — fall back to position when so.
        indexed = [(d.index if d.index is not None else i, d.embedding) for i, d in enumerate(resp.data)]
        vectors.extend(e for _, e in sorted(indexed, key=lambda p: p[0]))
    return vectors
