"""RAG Chat endpoint for grounded question answering with citations."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings
from app.dependencies import (
    get_settings,
    get_retrieval_service,
    get_generation_service,
    get_vector_store,
)
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.generation_service import (
    GenerationService,
    LLMError,
    LLMModelNotFoundError,
    LLMUnavailableError,
)
from app.services.retrieval_service import RetrievalService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question grounded in uploaded documents",
)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> ChatResponse:
    """Execute the end-to-end RAG pipeline for the given question.
    
    1. Check for indexed documents.
    2. Embed question and retrieve Top-K relevant chunks with metadata.
    3. Construct grounded prompt with untrusted document context.
    4. Query local Ollama model.
    5. Return generated answer and source citations.
    """
    # 1. Validate that documents exist in database
    documents = vector_store.list_documents()
    if not documents:
        return ChatResponse(
            question=request.question,
            answer="No documents have been uploaded or indexed yet. Please upload a PDF document first before asking questions.",
            sources=[],
            model=generation_service.model,
            context_used=False,
            warning="No documents available in vector store.",
        )

    # 2. Retrieve relevant context
    retrieval_result = retrieval_service.retrieve(
        query=request.question,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    if not retrieval_result.has_context:
        return ChatResponse(
            question=request.question,
            answer="Based on the uploaded documents, I could not find relevant information to answer your question.",
            sources=[],
            model=generation_service.model,
            context_used=False,
            warning="No chunks matched the search query.",
        )

    # 3. Generate answer using Ollama LLM
    try:
        answer = await generation_service.generate_answer(
            question=request.question,
            context=retrieval_result.context_text,
            citations=retrieval_result.citations,
        )
    except LLMUnavailableError as e:
        logger.error(f"LLM unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Ollama local LLM service is offline or unreachable at {generation_service.base_url}. "
                "Please start Ollama in your terminal ('ollama serve') and retry."
            ),
        )
    except LLMModelNotFoundError as e:
        logger.error(f"LLM model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except LLMError as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer from local LLM: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during answer generation.",
        )

    return ChatResponse(
        question=request.question,
        answer=answer,
        sources=retrieval_result.citations,
        model=generation_service.model,
        context_used=True,
    )
