"""Integration tests for document upload, listing, and deletion API routes."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.dependencies import get_embedding_service


def test_document_upload_success(client: TestClient, sample_pdf_bytes):
    """Verify uploading a valid PDF indexes chunks and returns 201."""
    # Mock embedding service to avoid heavy model download during tests
    mock_emb = MagicMock()
    mock_emb.embed_texts.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
    mock_emb.is_loaded.return_value = True
    mock_emb.dimension = 384

    from app.main import app
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb

    try:
        files = {"file": ("test_policy.pdf", sample_pdf_bytes, "application/pdf")}
        response = client.post("/api/documents/upload", files=files)
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test_policy.pdf"
        assert data["status"] == "indexed"
        assert data["page_count"] == 2
        assert data["chunk_count"] >= 2
        doc_id = data["document_id"]

        # Verify GET /api/documents lists the new doc
        list_res = client.get("/api/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total_count"] >= 1
        assert any(d["document_id"] == doc_id for d in list_data["documents"])

        # Verify DELETE /api/documents/{doc_id}
        del_res = client.delete(f"/api/documents/{doc_id}")
        assert del_res.status_code == 200
        del_data = del_res.json()
        assert del_data["document_id"] == doc_id
        assert del_data["deleted_chunks"] >= 2

        # Verify deleting again returns 404
        del_res_again = client.delete(f"/api/documents/{doc_id}")
        assert del_res_again.status_code == 404

    finally:
        app.dependency_overrides.pop(get_embedding_service, None)


def test_document_upload_invalid_extension(client: TestClient):
    """Verify non-PDF file upload returns 400 Bad Request."""
    files = {"file": ("document.txt", b"plain text content", "text/plain")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "Only .pdf files are accepted" in response.json()["detail"]


def test_document_upload_empty_file(client: TestClient):
    """Verify 0-byte file upload returns 400 Bad Request."""
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "uploaded file is empty" in response.json()["detail"]


def test_document_upload_scanned_pdf(client: TestClient, empty_pdf_bytes):
    """Verify scanned/empty-text PDF returns 422 Unprocessable Entity."""
    files = {"file": ("scanned.pdf", empty_pdf_bytes, "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 422
    assert "No extractable text was found" in response.json()["detail"]
