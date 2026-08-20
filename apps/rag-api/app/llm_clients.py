import logging

from langsmith import traceable
from openai import OpenAI
from .config import settings

logger = logging.getLogger(__name__)

gemini_client = OpenAI(base_url=settings.gemini_base_url, api_key=settings.gemini_api_key)
ollama_client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")  # Ollama ignores the key


@traceable(name="embed", run_type="embedding")
def embed(texts: list[str]) -> list[list[float]]:
    """Always Gemini — switching embedding models would change vector dimensions
    and break similarity search against existing Qdrant data."""
    resp = gemini_client.embeddings.create(model=settings.embed_model, input=texts)
    return [d.embedding for d in resp.data]


@traceable(name="chat-gemini", run_type="llm")
def _chat_gemini(messages: list[dict]) -> str:
    resp = gemini_client.chat.completions.create(model=settings.llm_model, messages=messages)
    return resp.choices[0].message.content


@traceable(name="chat-ollama-fallback", run_type="llm")
def _chat_ollama(messages: list[dict]) -> str:
    resp = ollama_client.chat.completions.create(model=settings.ollama_model, messages=messages)
    return resp.choices[0].message.content


@traceable(name="chat", run_type="chain")
def chat(messages: list[dict]) -> str:
    """Gemini first; on any Gemini API error (rate limit, 5xx, timeout, etc.),
    fall back to the local Ollama model so the RAG API stays available."""
    try:
        return _chat_gemini(messages)
    except Exception as e:
        logger.warning("Gemini chat call failed (%s), falling back to Ollama (%s)", e, settings.ollama_model)
        return _chat_ollama(messages)
