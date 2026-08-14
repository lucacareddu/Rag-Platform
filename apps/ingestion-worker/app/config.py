from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    github_token: str
    github_models_base_url: str = "https://models.github.ai/inference"
    embed_model: str = "openai/text-embedding-3-small"
    chunk_size: int = 800

    class Config:
        env_file = ".env"

settings = Settings()
