"""Unit tests for ChromaDB vector store and document catalog."""

from app.services.chunking_service import DocumentChunk
from app.services.vector_store_service import VectorStoreService


def test_vector_store_add_query_delete(test_temp_dir):
    """Verify end-to-end vector storage lifecycle."""
    store = VectorStoreService(
        persist_dir=str(test_temp_dir / "chroma"),
        collection_name="test_vs",
    )

    # 1. Register document in catalog
    store.register_document(
        document_id="doc_abc",
        filename="handbook.pdf",
        file_size_bytes=1024,
        page_count=2,
        chunk_count=2,
    )
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].document_id == "doc_abc"

    # 2. Add sample chunks with mock embeddings (dimension 4)
    chunks = [
        DocumentChunk(
            chunk_id="doc_abc_p1_c0",
            document_id="doc_abc",
            filename="handbook.pdf",
            page=1,
            text="The standard workday begins at 9:00 AM.",
            chunk_index=0,
            char_length=39,
        ),
        DocumentChunk(
            chunk_id="doc_abc_p2_c1",
            document_id="doc_abc",
            filename="handbook.pdf",
            page=2,
            text="Remote work policy permits two days work from home.",
            chunk_index=1,
            char_length=51,
        ),
    ]
    # Unit vectors
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    added = store.add_chunks(chunks, embeddings)
    assert added == 2
    assert store.collection.count() == 2

    # 3. Query similar chunks
    # Query vector closest to chunk 0
    citations = store.query_similar(query_embedding=[0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(citations) == 2
    assert citations[0].document_id == "doc_abc"
    assert citations[0].page == 1
    assert "9:00 AM" in citations[0].snippet
    assert citations[0].score is not None

    # 4. Query with document_id filter
    filtered = store.query_similar(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        document_id="doc_abc",
    )
    assert len(filtered) == 2

    non_matching = store.query_similar(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        document_id="non_existent",
    )
    assert len(non_matching) == 0

    # 5. Delete document
    deleted = store.delete_document("doc_abc")
    assert deleted == 2
    assert store.collection.count() == 0
    assert len(store.list_documents()) == 0
