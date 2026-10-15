"""Hybrid Search module using Qdrant (Dense) and BM25 (Sparse) with RRF."""

import logging
import time
from typing import List
from pathlib import Path
import bm25s

from src.config import settings
from src.retrieval.vector_store import QdrantVectorStore, SearchResult
from src.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines Dense Search and Sparse Search using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, vector_store: QdrantVectorStore, embedder: Embedder, bm25_dir: str = "data/processed/bm25_index"):
        self.vector_store = vector_store
        self.embedder = embedder
        
        bm25_path = Path(bm25_dir)
        if bm25_path.exists():
            logger.info("Loading BM25 index from %s", bm25_dir)
            self.bm25_retriever = bm25s.BM25.load(bm25_dir, load_corpus=True)
        else:
            logger.warning("BM25 index not found at %s. Hybrid search will fallback to Dense only.", bm25_dir)
            self.bm25_retriever = None

    def _rrf_combine(
        self, 
        dense_results: List[SearchResult], 
        sparse_results: List[SearchResult], 
        k: int = 60
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion."""
        rrf_scores = {}
        # We need a way to uniquely identify chunks. chunk_id is perfect.
        chunks_map = {}

        # Process dense results
        for rank, res in enumerate(dense_results):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunks_map[res.chunk_id] = res

        # Process sparse results
        for rank, res in enumerate(sparse_results):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunks_map[res.chunk_id] = res

        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final list
        final_results = []
        for chunk_id, score in sorted_chunks:
            res = chunks_map[chunk_id]
            res.score = score  # Assign RRF score to result
            final_results.append(res)
            
        return final_results

    def search(
        self, 
        query: str, 
        limit: int = 5,
        dense_weight: float = 1.0, 
        sparse_weight: float = 1.0
    ) -> List[SearchResult]:
        """Perform hybrid search and return fused results."""
        # 1. Dense Search
        t0 = time.perf_counter()
        query_vec = self.embedder.embed_query(query)
        t_embed = time.perf_counter()
        dense_results = self.vector_store.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vec,
            limit=limit
        )
        t_dense = time.perf_counter()

        # 2. Sparse Search
        if not self.bm25_retriever:
            return dense_results
            
        query_tokens = bm25s.tokenize(query)
        
        # bm25s returns (results, scores), where results is a dict or object containing the corpus docs
        sparse_docs, sparse_scores = self.bm25_retriever.retrieve(query_tokens, k=limit)
        
        # Convert sparse docs to SearchResult
        sparse_results = []
        # bm25s retrieve returns numpy arrays of shape (num_queries, k)
        for i in range(len(sparse_docs[0])):
            doc_dict = sparse_docs[0][i]
            score = float(sparse_scores[0][i])
            
            sparse_results.append(SearchResult(
                chunk_id=str(doc_dict.get("chunk_id", "")),
                doc_id=int(doc_dict.get("doc_id", 0)),
                article_id=str(doc_dict.get("article_id", "")),
                article_number=int(doc_dict.get("article_number", 0)),
                title=str(doc_dict.get("title", "")),
                category=str(doc_dict.get("category", "")),
                text=str(doc_dict.get("text", "")),
                score=score,
                clause_number=doc_dict.get("clause_number"),
            ))

        t_sparse = time.perf_counter()

        # 3. Combine with RRF
        fused = self._rrf_combine(dense_results, sparse_results)
        logger.info(
            "Hybrid search: embed %.0fms, qdrant %.0fms, bm25 %.0fms",
            (t_embed - t0) * 1000, (t_dense - t_embed) * 1000, (t_sparse - t_dense) * 1000,
        )
        return fused
