from fastapi import FastAPI, HTTPException
from typing import List

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.generator import GroqGenerator
from src.api.models import AskRequest, AskResponse, SourceDocument

app = FastAPI(
    title="VN Legal RAG API",
    description="API for Vietnamese Legal Question Answering using RAG",
    version="0.1.0"
)

# Global variables for lazy loading
embedder = None
vector_store = None
reranker = None
hybrid_retriever = None
generator = None

def get_embedder():
    global embedder
    if embedder is None:
        embedder = Embedder()
    return embedder

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = QdrantVectorStore()
    return vector_store

def get_hybrid_retriever():
    global hybrid_retriever
    if hybrid_retriever is None:
        hybrid_retriever = HybridRetriever(get_vector_store(), get_embedder())
    return hybrid_retriever

def get_reranker():
    global reranker
    if reranker is None:
        reranker = Reranker()
    return reranker

def get_generator():
    global generator
    if generator is None:
        generator = GroqGenerator()
    return generator

@app.get("/health")
def health_check():
    """Check if the API is running."""
    return {"status": "ok", "message": "API is running"}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Ask a legal question and get an answer generated from retrieved documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # 1. Embed the query
        query_vec = get_embedder().embed_query(request.question)
        
        # 2. Retrieve relevant documents (fetch more for reranking, use hybrid)
        initial_limit = request.limit * 3
        search_results = get_hybrid_retriever().search(
            query=request.question,
            limit=initial_limit
        )

        if not search_results:
            return AskResponse(
                answer="Xin lỗi, tôi không tìm thấy tài liệu pháp lý nào liên quan đến câu hỏi của bạn.",
                sources=[]
            )

        # 3. Rerank results
        search_results = get_reranker().rerank(
            query=request.question,
            results=search_results,
            top_k=request.limit
        )

        # 4. Generate answer
        generation_result = get_generator().generate_answer(request.question, search_results)
        
        # 4. Format response
        sources = [
            SourceDocument(
                id=res.chunk_id,
                content=res.text,
                score=res.score,
                metadata={
                    "doc_id": res.doc_id,
                    "article_id": res.article_id,
                    "article_number": res.article_number,
                    "title": res.title,
                    "category": res.category
                }
            )
            for res in search_results
        ]

        return AskResponse(
            answer=generation_result.answer,
            sources=sources
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
