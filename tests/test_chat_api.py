"""Integration tests for RAG Chat API endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.dependencies import get_embedding_service, get_vector_store
from app.schemas.chat import SourceCitation
from app.services.chunking_service import DocumentChunk
from app.services.generation_service import LLMUnavailableError, LLMModelNotFoundError
from app.services.vector_store_service import VectorStoreService


def test_chat_no_documents_uploaded(client: TestClient):
    """Verify chat handles empty vector store cleanly."""
    response = client.post("/api/chat", json={"question": "What is the policy?"})
    assert response.status_code == 200
    data = response.json()
    assert "No documents have been uploaded or indexed yet" in data["answer"]
    assert data["context_used"] is False
    assert data["sources"] == []


def test_chat_validation_errors(client: TestClient):
    """Verify invalid questions or out-of-range top_k are rejected by Pydantic."""
    # Empty question
    res1 = client.post("/api/chat", json={"question": ""})
    assert res1.status_code == 422

    # Negative or 0 top_k
    res2 = client.post("/api/chat", json={"question": "Valid question?", "top_k": 0})
    assert res2.status_code == 422


def test_chat_with_grounded_context_and_sources(client: TestClient, test_temp_dir):
    """Verify end-to-end chat returns answer and source citations."""
    store = VectorStoreService(
        persist_dir=str(test_temp_dir / "chroma"),
        collection_name="test_chat_col",
    )
    store.register_document("doc_hr", "hr_policy.pdf", 2048, 1, 1)

    chunk = DocumentChunk(
        chunk_id="doc_hr_p1_c0",
        document_id="doc_hr",
        filename="hr_policy.pdf",
        page=1,
        text="Employees receive 15 days of paid annual vacation.",
        chunk_index=0,
        char_length=52,
    )
    store.add_chunks([chunk], [[1.0, 0.0, 0.0, 0.0]])

    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]

    from app.main import app
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb

    mock_llm_answer = "Employees receive 15 days of paid annual vacation (hr_policy.pdf, Page 1)."

    try:
        with patch(
            "app.services.generation_service.GenerationService.generate_answer",
            new_callable=AsyncMock,
            return_value=mock_llm_answer,
        ):
            response = client.post(
                "/api/chat",
                json={"question": "How many vacation days do employees receive?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == mock_llm_answer
            assert data["context_used"] is True
            assert len(data["sources"]) == 1
            citation = data["sources"][0]
            assert citation["filename"] == "hr_policy.pdf"
            assert citation["page"] == 1
            assert "15 days of paid annual vacation" in citation["snippet"]
    finally:
        app.dependency_overrides.pop(get_vector_store, None)
        app.dependency_overrides.pop(get_embedding_service, None)


def test_chat_ollama_offline_error_handling(client: TestClient, test_temp_dir):
    """Verify 503 is returned when Ollama is offline."""
    store = VectorStoreService(
        persist_dir=str(test_temp_dir / "chroma"),
        collection_name="test_chat_offline",
    )
    store.register_document("doc_offline", "policy.pdf", 100, 1, 1)
    chunk = DocumentChunk(
        chunk_id="doc_off_p1_c0",
        document_id="doc_offline",
        filename="policy.pdf",
        page=1,
        text="Some policy text.",
        chunk_index=0,
        char_length=17,
    )
    store.add_chunks([chunk], [[1.0, 0.0, 0.0, 0.0]])

    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]

    from app.main import app
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb

    try:
        with patch(
            "app.services.generation_service.GenerationService.generate_answer",
            new_callable=AsyncMock,
            side_effect=LLMUnavailableError("Connection refused"),
        ):
            response = client.post("/api/chat", json={"question": "What is the policy?"})
            assert response.status_code == 503
            assert "offline or unreachable" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_vector_store, None)
        app.dependency_overrides.pop(get_embedding_service, None)
