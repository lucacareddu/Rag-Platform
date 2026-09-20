import time

from openai import OpenAI, RateLimitError
from .config import settings

client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)


_recent: list[float] = []


def _throttle(n: int):
    """The free-tier embedding quota counts each input item, not each HTTP
    request, so a batched call of 32 spends 32 units. Without pacing a
    150-chunk document blows the per-minute budget mid-document and the
    retries then compete with the very requests that exhausted it."""
    global _recent
    limit = settings.embed_requests_per_minute
    while True:
        now = time.monotonic()
        _recent = [t for t in _recent if now - t < 60]
        if len(_recent) + n <= limit:
            break
        time.sleep(max(1.0, 60 - (now - _recent[0]) + 1))
    _recent.extend([time.monotonic()] * n)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), settings.embed_batch_size):
        batch = chunks[start:start + settings.embed_batch_size]
        _throttle(len(batch))
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
