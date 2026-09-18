import time

from openai import OpenAI, RateLimitError
from .config import settings

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), settings.embed_batch_size):
        batch = chunks[start:start + settings.embed_batch_size]
        vectors.extend(_embed_batch(batch))
    return vectors


def _embed_batch(batch: list[str]) -> list[list[float]]:
    for attempt in range(settings.embed_max_retries + 1):
        try:
            resp = client.embeddings.create(model=settings.embed_model, input=batch)
            indexed = [(d.index if d.index is not None else i, d.embedding) for i, d in enumerate(resp.data)]
            return [e for _, e in sorted(indexed, key=lambda p: p[0])]
        except RateLimitError as e:
            if attempt == settings.embed_max_retries:
                raise
            time.sleep(_retry_delay(e))


def _retry_delay(e: RateLimitError) -> float:
    try:
        details = e.body["error"]["details"]
        delay = next(d["retryDelay"] for d in details if d.get("@type", "").endswith("RetryInfo"))
        return float(delay.rstrip("s")) + 1
    except Exception:
        return 5.0
