from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "docs"
    gemini_api_key: str
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embed_model: str = "gemini-embedding-2-preview"
    chunk_size: int = 800
    embed_batch_size: int = 32
    embed_max_retries: int = 5
    embed_requests_per_minute: int = 90   # free tier caps at 100 inputs/min; leave headroom

    # graph_enabled=false skips extraction, ingest stays vectors-only.
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    graph_enabled: bool = True
    max_mention_ratio: float = 0.15   # drop entities mentioned in more chunks than this (doc title/topic self-references)
    extract_model: str = "gemini-3.5-flash"
    extract_window_chars: int = 60000   # covers the whole document across windows, not just the start
    extract_window_overlap: int = 2000   # so a relation straddling a boundary is still seen whole
    extract_max_windows: int = 40        # cost ceiling: ~2.3M chars of a single document

    class Config:
        env_file = ".env"

settings = Settings()
