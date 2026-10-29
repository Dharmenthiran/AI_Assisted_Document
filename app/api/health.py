"""Health and dependency diagnostic endpoints."""

from fastapi import APIRouter, Depends
from app.config import Settings
from app.dependencies import (
    get_settings,
    get_generation_service,
    get_vector_store,
    get_embedding_service,
)
from app.schemas.health import HealthResponse, DependencyStatus
from app.services.generation_service import GenerationService
from app.services.vector_store_service import VectorStoreService
from app.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def check_health(
    settings: Settings = Depends(get_settings),
    gen_service: GenerationService = Depends(get_generation_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    emb_service: EmbeddingService = Depends(get_embedding_service),
) -> HealthResponse:
    """Check health and connectivity of application dependencies."""
    ollama_status = await gen_service.check_availability()
    vector_stats = vector_store.get_stats()

    embedding_status = {
        "model_name": emb_service.model_name,
        "loaded_in_memory": emb_service.is_loaded(),
        "dimension": emb_service.dimension if emb_service.is_loaded() else 384,
    }

    # Overall health determination
    is_healthy = ollama_status.get("online", False) and ollama_status.get("model_present", False)
    status_label = "healthy" if is_healthy else "degraded"

    return HealthResponse(
        status=status_label,
        app_name=settings.APP_NAME,
        dependencies=DependencyStatus(
            ollama=ollama_status,
            vector_store=vector_stats,
            embedding_model=embedding_status,
        ),
    )
