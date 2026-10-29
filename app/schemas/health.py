"""Pydantic models for application health diagnostics."""

from typing import Dict, Any, List
from pydantic import BaseModel


class DependencyStatus(BaseModel):
    """Status details for system dependencies."""

    ollama: Dict[str, Any]
    vector_store: Dict[str, Any]
    embedding_model: Dict[str, Any]


class HealthResponse(BaseModel):
    """Application health and readiness status."""

    status: str
    app_name: str
    dependencies: DependencyStatus
