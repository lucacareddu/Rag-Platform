from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    gemini_api_key: str
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.5-flash"
    embed_model: str = "gemini-embedding-2-preview"
    ingestion_url: str = "http://ingestion-worker:8001"

    class Config:
        env_file = ".env"

settings = Settings()
