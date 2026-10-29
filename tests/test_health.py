"""Integration tests for application health diagnostic endpoint."""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


def test_health_endpoint_healthy(client: TestClient):
    """Verify health endpoint returns healthy when Ollama is online."""
    mock_availability = {
        "online": True,
        "model_present": True,
        "configured_model": "llama3.2:3b",
        "available_models": ["llama3.2:3b"],
        "error": None,
    }

    with patch(
        "app.services.generation_service.GenerationService.check_availability",
        new_callable=AsyncMock,
        return_value=mock_availability,
    ):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "dependencies" in data
        assert data["dependencies"]["ollama"]["online"] is True
        assert data["dependencies"]["ollama"]["model_present"] is True


def test_health_endpoint_degraded(client: TestClient):
    """Verify health endpoint returns degraded status when Ollama is offline."""
    mock_availability = {
        "online": False,
        "model_present": False,
        "configured_model": "llama3.2:3b",
        "available_models": [],
        "error": "Connection refused",
    }

    with patch(
        "app.services.generation_service.GenerationService.check_availability",
        new_callable=AsyncMock,
        return_value=mock_availability,
    ):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["dependencies"]["ollama"]["online"] is False
