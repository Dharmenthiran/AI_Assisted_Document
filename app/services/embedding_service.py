"""Embedding Service using SentenceTransformers.

Generates dense vector representations for document chunks and user queries
using a locally cached, open-source model (default: all-MiniLM-L6-v2).
Operates reliably on CPU or GPU without cloud fees or external APIs.
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages local embedding model lifecycle and vector generation."""

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingService":
        """Singleton accessor to ensure model weights are kept in memory only once."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _load_model(self) -> SentenceTransformer:
        """Loads SentenceTransformer weights lazily on first access."""
        if self._model is None:
            logger.info(f"Loading local embedding model: '{self.model_name}'...")
            try:
                self._model = SentenceTransformer(self.model_name)
                # Compute dimension using a dummy sample
                test_vec = self._model.encode("test", convert_to_numpy=True)
                self._dimension = len(test_vec)
                logger.info(
                    f"Model '{self.model_name}' loaded successfully. Vector dimension: {self._dimension}"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                raise RuntimeError(
                    f"Could not initialize embedding model '{self.model_name}'. "
                    f"Please verify internet connectivity for initial download or check model name. Details: {e}"
                )
        return self._model

    @property
    def dimension(self) -> int:
        """Returns the vector dimensionality."""
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def is_loaded(self) -> bool:
        """Check if model weights are loaded in RAM."""
        return self._model is not None

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a list of text strings into vector floats."""
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalized for cosine similarity
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single user query string."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")
        model = self._load_model()
        vector = model.encode(
            query.strip(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()
