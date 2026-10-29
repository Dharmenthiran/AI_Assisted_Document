"""Application Configuration Module.

Loads environment variables with robust defaults for local development.
Integrates Pydantic BaseSettings for type validation and documentation.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for Document AI Assistant."""

    APP_NAME: str = "Document AI Assistant"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Ollama LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT_SECONDS: float = 300.0

    # Embeddings & Vector Store
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    COLLECTION_NAME: str = "document_rag_collection"

    # Chunking & Retrieval Parameters
    # Note: CHUNK_SIZE & CHUNK_OVERLAP use character-based approximation
    # (~4 characters per token: 1000 chars ≈ 250 tokens; 150 chars ≈ 37 tokens)
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 4

    # Upload Constraints
    MAX_UPLOAD_MB: int = 15
    UPLOADS_DIR: str = str(BASE_DIR / "data" / "uploads")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure runtime directories exist immediately
Path(settings.UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
