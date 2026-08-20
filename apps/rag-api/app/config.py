import os

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    gemini_api_key: str
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.5-flash"
    embed_model: str = "gemini-embedding-2-preview"
    ingestion_url: str = "http://ingestion-worker:8001"

    # Local LLM fallback (used only for chat/generation when Gemini errors —
    # embeddings always stay on Gemini so Qdrant vector dimensions stay consistent)
    ollama_base_url: str = "http://ollama:11434/v1"
    ollama_model: str = "phi4-mini"

    # LangSmith tracing (optional — no-ops if langsmith_api_key is unset)
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-platform"
    langsmith_tracing: bool = True

    class Config:
        env_file = ".env"

settings = Settings()

# langsmith's `traceable` decorator reads these env vars at import/call time, so
# they must be set before any module using @traceable gets imported.
if settings.langsmith_api_key and settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
