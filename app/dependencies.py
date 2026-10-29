"""FastAPI Dependency Injection Providers.

Constructs and provides shared singletons and configured service instances
to API route handlers, enabling clean decoupling and easy test mocking.
"""

from functools import lru_cache
from fastapi import Depends

from app.config import Settings, settings
from app.services.pdf_service import PDFService
from app.services.chunking_service import RecursiveTextChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


@lru_cache()
def get_settings() -> Settings:
    """Provides cached application settings."""
    return settings


def get_pdf_service(
    cfg: Settings = Depends(get_settings),
) -> PDFService:
    """Provides configured PDFService."""
    return PDFService(
        uploads_dir=cfg.UPLOADS_DIR,
        max_upload_mb=cfg.MAX_UPLOAD_MB,
    )


def get_chunker(
    cfg: Settings = Depends(get_settings),
) -> RecursiveTextChunker:
    """Provides configured RecursiveTextChunker."""
    return RecursiveTextChunker(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )


def get_embedding_service(
    cfg: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Provides singleton EmbeddingService instance."""
    return EmbeddingService.get_instance(model_name=cfg.EMBEDDING_MODEL)


def get_vector_store(
    cfg: Settings = Depends(get_settings),
) -> VectorStoreService:
    """Provides singleton VectorStoreService instance."""
    return VectorStoreService.get_instance(
        persist_dir=cfg.CHROMA_PERSIST_DIR,
        collection_name=cfg.COLLECTION_NAME,
    )


def get_retrieval_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    cfg: Settings = Depends(get_settings),
) -> RetrievalService:
    """Provides configured RetrievalService."""
    return RetrievalService(
        embedding_service=embedding_service,
        vector_store_service=vector_store,
        default_top_k=cfg.TOP_K,
    )


def get_generation_service(
    cfg: Settings = Depends(get_settings),
) -> GenerationService:
    """Provides configured GenerationService."""
    return GenerationService(
        base_url=cfg.OLLAMA_BASE_URL,
        model=cfg.OLLAMA_MODEL,
        timeout=cfg.OLLAMA_TIMEOUT_SECONDS,
    )
