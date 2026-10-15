"""Embedder module wrapping SentenceTransformer with device detection and token limit warnings."""

import logging
from typing import Sequence
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """SentenceTransformer embedding wrapper with device management and token length checks."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.resolved_device

        logger.info("Initializing Embedder '%s' on device '%s'...", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.max_seq_length: int = getattr(self.model, "max_seq_length", 128)

        # Handle sentence-transformers API differences
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
            self.embedding_dim: int = dim if dim is not None else 384
        else:
            self.embedding_dim = 384

        # Log, never print: stdout is the protocol channel when served over MCP stdio.
        logger.info(
            "Loaded Embedder '%s' on '%s' (max_seq_length=%d, dim=%d)",
            self.model_name,
            self.device,
            self.max_seq_length,
            self.embedding_dim,
        )

    def check_token_length(self, text: str) -> int:
        """Calculate token count and issue warning if text exceeds max_seq_length."""
        if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "encode"):
            token_count = len(self.model.tokenizer.encode(text, truncation=False))
            if token_count > self.max_seq_length:
                warnings.warn(
                    f"Text with {token_count} tokens exceeds max_seq_length ({self.max_seq_length}). "
                    "Trailing text will be truncated by embedding model.",
                    UserWarning,
                    stacklevel=2,
                )
            return token_count
        return len(text.split())

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        warn_on_truncation: bool = False,
    ) -> np.ndarray:
        """Generate normalized vector embeddings for a list of texts.

        Args:
            texts: Sequence of strings to encode.
            batch_size: Batch size for model inference.
            show_progress_bar: Whether to display a progress bar.
            warn_on_truncation: If True, check each text for truncation warnings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        if warn_on_truncation:
            for t in texts:
                self.check_token_length(t)

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a single query."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return list(embedding)
