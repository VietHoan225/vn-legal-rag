"""Reranker module using Cross-Encoder models."""

import logging
from typing import List
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

class Reranker:
    """Re-ranks a list of search results using a Cross-Encoder."""

    def __init__(self, model_name: str | None = None):
        """Initialize the reranker model."""
        self.model_name = model_name or settings.reranker_model
        logger.info("Loading reranker model: %s on %s", self.model_name, settings.resolved_device)
        self.model = CrossEncoder(
            self.model_name, 
            device=settings.resolved_device,
            max_length=512
        )

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """
        Re-rank the search results based on the query.
        
        Args:
            query: The user's question.
            results: The list of initial SearchResults from the vector store.
            top_k: Number of top results to return after reranking.
            
        Returns:
            A sorted list of SearchResults (highest score first) truncated to top_k.
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, res.text] for res in results]
        
        # Predict scores
        scores = self.model.predict(pairs)  # type: ignore
        
        # Update scores in results
        for res, score in zip(results, scores):
            res.score = float(score)  # Replace vector distance with cross-encoder score
            
        # Sort by new score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
