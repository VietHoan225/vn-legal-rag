"""Tools for the AI Agent."""

from typing import List
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore, SearchResult
from src.retrieval.hybrid_search import HybridRetriever
from src.config import settings

class LegalSearchTool:
    """Tool to search for legal documents in the vector database."""
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        
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
        
    def execute(self, query: str) -> str:
        """Execute the search and return formatted string."""
        results = self.retriever.search(
            query=query,
            limit=5
        )
        
        if not results:
            return "Không tìm thấy tài liệu pháp lý nào phù hợp với truy vấn này trong cơ sở dữ liệu."
            
        return self._format_results(results)
        
    def _format_results(self, search_results: List[SearchResult]) -> str:
        """Combine retrieved search results into a single context string."""
        context_parts = []
        for rank, res in enumerate(search_results, start=1):
            source = f"[{rank}] {res.title} (Điều {res.article_number})" if res.article_number else f"[{rank}] {res.title}"
            text = f"Nguồn: {source}\nNội dung: {res.text}"
            context_parts.append(text)
            
        return "\n\n---\n\n".join(context_parts)
