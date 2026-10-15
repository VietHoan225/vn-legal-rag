"""Tools for the AI Agent."""

from typing import List, Optional
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore, SearchResult
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.config import settings


class SearchError(Exception):
    """The legal search itself failed (database down, model error), as opposed to finding nothing."""


class LegalSearchTool:
    """Tool to search for legal documents in the vector database."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[Reranker] = None,
        top_k: int = 5,
        candidates: int | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = max(candidates or settings.rerank_candidates, top_k)
        
    def get_tool_schema(self) -> dict:
        """Returns the JSON schema for this tool to pass to LLMs (Function Calling)."""
        return {
            "type": "function",
            "function": {
                "name": "search_legal_docs",
                "description": "Tìm kiếm các điều khoản pháp luật Việt Nam liên quan đến câu hỏi. Sử dụng công cụ này khi cần tra cứu luật để có căn cứ trả lời.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Câu truy vấn để tìm kiếm. Nên là một câu hỏi rõ ý hoặc từ khóa pháp lý (ví dụ: 'mức phạt không đội mũ bảo hiểm').",
                        }
                    },
                    "required": ["query"],
                },
            }
        }
        
    def search(self, query: str) -> List[SearchResult]:
        """Retrieve legal chunks for a query; with a reranker, fetch `candidates` hits and keep the best top_k.

        Raises SearchError if retrieval fails. An empty list means nothing relevant was found.
        """
        try:
            if self.reranker is None:
                return self.retriever.search(query=query, limit=self.top_k)
            candidates = self.retriever.search(query=query, limit=self.candidates)
            return self.reranker.rerank(query, candidates, top_k=self.top_k)
        except Exception as e:
            raise SearchError(f"Legal search failed: {e}") from e

    def execute(self, query: str) -> str:
        """Execute the search and return formatted string."""
        return self.format_results(self.search(query))

    def format_results(self, search_results: List[SearchResult]) -> str:
        """Combine retrieved search results into a single context string."""
        if not search_results:
            return "Không tìm thấy tài liệu pháp lý nào phù hợp với truy vấn này trong cơ sở dữ liệu."
        context_parts = []
        for rank, res in enumerate(search_results, start=1):
            source = f"[{rank}] {res.title} ({res.location()})" if res.article_number else f"[{rank}] {res.title}"
            text = f"Nguồn: {source}\nNội dung: {res.text}"
            context_parts.append(text)
            
        return "\n\n---\n\n".join(context_parts)
