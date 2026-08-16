from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    gemini_api_key: str
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embed_model: str = "gemini-embedding-2-preview"
    chunk_size: int = 800

    class Config:
        env_file = ".env"

settings = Settings()
