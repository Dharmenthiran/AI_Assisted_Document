"""Retrieval Service for the RAG pipeline.

Coordinates query embedding, vector similarity search, and context assembly
independent of the LLM generation phase.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from app.schemas.chat import SourceCitation
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService


@dataclass
class RetrievalResult:
    """Encapsulates retrieved context and citations."""
    context_text: str
    citations: List[SourceCitation]
    has_context: bool


class RetrievalService:
    """Manages chunk search and context packaging."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        default_top_k: int = 4,
    ):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Embeds query, searches vector store, and formats grounded context text.
        
        Returns RetrievalResult with formatted context and citations.
        """
        k = top_k if top_k is not None and top_k > 0 else self.default_top_k

        # 1. Embed query
        query_vector = self.embedding_service.embed_query(query)

        # 2. Retrieve top-k similar chunks
        citations = self.vector_store_service.query_similar(
            query_embedding=query_vector,
            top_k=k,
            document_id=document_id,
        )

        if not citations:
            return RetrievalResult(
                context_text="",
                citations=[],
                has_context=False,
            )

        # 3. Assemble structured context block for LLM
        # We retrieve full chunk texts to give the LLM complete context
        context_blocks = []
        for citation in citations:
            full_text = self.vector_store_service.get_chunk_by_id(citation.chunk_id)
            text_to_use = full_text if full_text else citation.snippet
            context_blocks.append(
                f"[Source: {citation.filename} | Page: {citation.page}]\n{text_to_use}"
            )

        assembled_context = "\n\n---\n\n".join(context_blocks)

        return RetrievalResult(
            context_text=assembled_context,
            citations=citations,
            has_context=True,
        )
