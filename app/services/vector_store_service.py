"""ChromaDB Persistent Vector Store and Document Catalog Service.

Manages persistent embeddings storage, HNSW cosine index, chunk retrieval with metadata,
and document-level registration.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.schemas.chat import SourceCitation
from app.schemas.document import DocumentMetadata
from app.services.chunking_service import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Encapsulates persistent ChromaDB client, collection lifecycle, and document metadata registry."""

    _instance: Optional["VectorStoreService"] = None

    def __init__(self, persist_dir: str, collection_name: str = "document_rag_collection"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.catalog_path = self.persist_dir / "documents_catalog.json"

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        # Cosine distance collection: distance = 1 - cosine_similarity
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_catalog()

    @classmethod
    def get_instance(
        cls, persist_dir: str, collection_name: str = "document_rag_collection"
    ) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = cls(persist_dir=persist_dir, collection_name=collection_name)
        return cls._instance

    # -------------------------------------------------------------------------
    # Catalog Management (Document Metadata Persistence)
    # -------------------------------------------------------------------------

    def _ensure_catalog(self) -> None:
        """Ensure document catalog file exists."""
        if not self.catalog_path.exists():
            self._save_catalog({})

    def _load_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load catalog from disk."""
        try:
            if self.catalog_path.exists():
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading document catalog: {e}")
        return {}

    def _save_catalog(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Persist catalog to disk atomically."""
        tmp_path = self.catalog_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.catalog_path)

    def register_document(
        self,
        document_id: str,
        filename: str,
        file_size_bytes: int,
        page_count: int,
        chunk_count: int,
        status: str = "indexed",
        error_message: Optional[str] = None,
    ) -> DocumentMetadata:
        """Register a new document in the metadata catalog."""
        catalog = self._load_catalog()
        doc_meta = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_size_bytes=file_size_bytes,
            page_count=page_count,
            chunk_count=chunk_count,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            error_message=error_message,
        )
        catalog[document_id] = doc_meta.model_dump()
        self._save_catalog(catalog)
        return doc_meta

    def list_documents(self) -> List[DocumentMetadata]:
        """List all registered documents ordered by creation time descending."""
        catalog = self._load_catalog()
        docs = [DocumentMetadata(**item) for item in catalog.values()]
        docs.sort(key=lambda d: d.created_at, reverse=True)
        return docs

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieve document metadata by document ID."""
        catalog = self._load_catalog()
        if document_id in catalog:
            return DocumentMetadata(**catalog[document_id])
        return None

    # -------------------------------------------------------------------------
    # Vector Chunk Operations
    # -------------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Insert document chunks and their dense embeddings into ChromaDB."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "char_length": chunk.char_length,
            }
            for chunk in chunks
        ]

        # Insert in batches of 100 to avoid any RPC or memory limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(f"Upserted {len(chunks)} chunks into collection '{self.collection_name}'")
        return len(chunks)

    def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        document_id: Optional[str] = None,
    ) -> List[SourceCitation]:
        """Query top_k most similar chunks, converting cosine distance to similarity score."""
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return []

        # Bound top_k to available items
        k = min(top_k, total_chunks)
        where_filter = {"document_id": document_id} if document_id else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        citations: List[SourceCitation] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return citations

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            text_content = documents[idx] if idx < len(documents) else ""
            dist = distances[idx] if idx < len(distances) else 1.0

            # For cosine distance: cosine_sim = 1 - distance
            similarity = round(max(0.0, 1.0 - dist), 4)

            snippet = text_content[:280] + "..." if len(text_content) > 280 else text_content

            citations.append(
                SourceCitation(
                    document_id=meta.get("document_id", "unknown"),
                    filename=meta.get("filename", "unknown.pdf"),
                    page=int(meta.get("page", 1)),
                    chunk_id=chunk_id,
                    snippet=snippet,
                    score=similarity,
                )
            )

        return citations

    def get_chunk_by_id(self, chunk_id: str) -> Optional[str]:
        """Fetch full text of a chunk by ID."""
        res = self.collection.get(ids=[chunk_id], include=["documents"])
        if res and res["documents"]:
            return res["documents"][0]
        return None

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document from ChromaDB and remove from catalog."""
        # Check count before deletion
        initial_count = self.collection.count()
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as e:
            logger.warning(f"Error during collection delete for {document_id}: {e}")

        deleted_chunks = max(0, initial_count - self.collection.count())

        # Update catalog
        catalog = self._load_catalog()
        if document_id in catalog:
            del catalog[document_id]
            self._save_catalog(catalog)

        return deleted_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Return collection diagnostics."""
        catalog = self._load_catalog()
        return {
            "total_documents": len(catalog),
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
        }
