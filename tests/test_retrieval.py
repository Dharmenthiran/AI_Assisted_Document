"""Unit tests for RAG retrieval service independent of LLM generation."""

from unittest.mock import MagicMock
from app.services.chunking_service import DocumentChunk
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService


def test_retrieval_empty_vector_store(test_temp_dir):
    """Verify retrieval handles empty collection gracefully."""
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    store = VectorStoreService(
        persist_dir=str(test_temp_dir / "chroma"),
        collection_name="test_retrieval_empty",
    )
    retrieval_service = RetrievalService(
        embedding_service=mock_emb,
        vector_store_service=store,
        default_top_k=2,
    )

    res = retrieval_service.retrieve(query="What is the policy?")
    assert res.has_context is False
    assert res.citations == []
    assert res.context_text == ""


def test_retrieval_with_chunks_assembles_context(test_temp_dir):
    """Verify retrieval returns structured context and source citations."""
    mock_emb = MagicMock()
    mock_emb.embed_query.return_value = [1.0, 0.0, 0.0, 0.0]

    store = VectorStoreService(
        persist_dir=str(test_temp_dir / "chroma"),
        collection_name="test_retrieval_chunks",
    )
    store.register_document("doc_1", "security_policy.pdf", 500, 1, 1)

    chunk = DocumentChunk(
        chunk_id="doc_1_p1_c0",
        document_id="doc_1",
        filename="security_policy.pdf",
        page=1,
        text="All laptops must have full-disk encryption enabled.",
        chunk_index=0,
        char_length=51,
    )
    store.add_chunks([chunk], [[1.0, 0.0, 0.0, 0.0]])

    retrieval_service = RetrievalService(
        embedding_service=mock_emb,
        vector_store_service=store,
        default_top_k=2,
    )

    res = retrieval_service.retrieve(query="Are laptops encrypted?")
    assert res.has_context is True
    assert len(res.citations) == 1
    assert res.citations[0].filename == "security_policy.pdf"
    assert res.citations[0].page == 1
    assert "[Source: security_policy.pdf | Page: 1]" in res.context_text
    assert "All laptops must have full-disk encryption enabled." in res.context_text
