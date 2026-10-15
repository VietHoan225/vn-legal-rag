from collections import OrderedDict
import json
import logging
import time
from threading import Lock

import groq
import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore, SearchResult
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridRetriever
from src.generation.generator import GroqGenerator
from src.api.models import AskRequest, AskResponse, ChatRequest, ChatResponse, SourceDocument
from src.agent.agent import LegalAgent
from src.agent.tools import LegalSearchTool, SearchError

# Show INFO logs (timings) from project code only; libraries stay at WARNING
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

BUSY_MESSAGE = "Hệ thống đang quá tải (LLM giới hạn lượt gọi). Vui lòng thử lại sau khoảng 1 phút."
SEARCH_UNAVAILABLE_MESSAGE = "Không tra cứu được cơ sở dữ liệu pháp luật lúc này nên tôi không trả lời để tránh thiếu căn cứ. Vui lòng thử lại sau."
UNAVAILABLE_MESSAGE = "Dịch vụ LLM tạm thời gián đoạn. Vui lòng thử lại sau ít phút."
MAX_SESSIONS = 100

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

def to_source_document(res: SearchResult) -> SourceDocument:
    """Convert a retrieval result into the API source format."""
    return SourceDocument(
        id=res.chunk_id,
        content=res.text,
        score=res.score,
        metadata={
            "doc_id": res.doc_id,
            "article_id": res.article_id,
            "article_number": res.article_number,
            "clause_number": res.clause_number,
            "location": res.location(),
            "title": res.title,
            "category": res.category,
        },
    )


# One LegalAgent per chat session (memory lives inside the agent).
# Least recently used sessions are dropped once MAX_SESSIONS is reached.
sessions: "OrderedDict[str, LegalAgent]" = OrderedDict()
sessions_lock = Lock()


def create_agent() -> LegalAgent:
    """Build a new agent that shares the loaded retriever and reranker."""
    return LegalAgent(search_tool=LegalSearchTool(get_hybrid_retriever(), get_reranker()))


def get_session_agent(session_id: str) -> LegalAgent:
    """Return the agent for a session, creating it on first use."""
    with sessions_lock:
        agent = sessions.get(session_id)
        if agent is None:
            agent = create_agent()
            sessions[session_id] = agent
            if len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        sessions.move_to_end(session_id)
        return agent


def unique_sources(agent: LegalAgent) -> list[SourceDocument]:
    """Sources the agent retrieved for its latest message, each chunk once."""
    # The agent may search several times
    unique = {res.chunk_id: res for res in agent.last_sources}.values()
    return [to_source_document(res) for res in unique]


def llm_http_error(exc: Exception) -> HTTPException:
    """Map an LLM client failure to the HTTP error the caller should see."""
    if isinstance(exc, SearchError):
        logger.error("Legal search unavailable: %s", exc)
        return HTTPException(status_code=503, detail=SEARCH_UNAVAILABLE_MESSAGE)
    if isinstance(exc, openai.RateLimitError):
        logger.warning("LLM rate limit: %s", exc)
        return HTTPException(status_code=429, detail=BUSY_MESSAGE)
    if isinstance(exc, openai.APITimeoutError):
        return HTTPException(status_code=504, detail="LLM phản hồi quá lâu. Vui lòng thử lại.")
    if isinstance(exc, openai.APIError):  # provider-side failure, e.g. "Service temporarily overloaded"
        logger.warning("LLM provider error: %s", exc)
        return HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE)
    logger.error("Chat failed: %s", exc, exc_info=exc)
    return HTTPException(status_code=500, detail=str(exc))


def ndjson(event: dict) -> str:
    """One streaming event as a JSON line."""
    return json.dumps(event, ensure_ascii=False) + "\n"


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
        
        # 5. Format response
        sources = [to_source_document(res) for res in search_results]

        return AskResponse(
            answer=generation_result.answer,
            sources=sources
        )

    except groq.RateLimitError as e:
        logger.warning("Groq rate limit: %s", e)
        raise HTTPException(status_code=429, detail=BUSY_MESSAGE)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Multi-turn chat with the legal agent; memory is kept per session_id."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    agent = get_session_agent(request.session_id)
    try:
        answer = agent.chat(request.message)
    except Exception as e:
        raise llm_http_error(e)

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        sources=unique_sources(agent),
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Like /chat, but streams the answer as newline-delimited JSON events.

    Events: {"type": "token", "text": ...} repeated, then {"type": "sources", "sources": [...]},
    or {"type": "error", "detail": ...} if the LLM fails after streaming has started.
    Failures before the first token still return a normal HTTP error (429, 504, 500).
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    agent = get_session_agent(request.session_id)
    tokens = agent.chat_stream(request.message)
    t0 = time.perf_counter()

    # Pull the first token here so early failures become proper HTTP errors
    try:
        first = next(tokens, None)
    except Exception as e:
        raise llm_http_error(e)
    logger.info("Chat stream: first token after %.1fs", time.perf_counter() - t0)

    def events():
        try:
            if first is not None:
                yield ndjson({"type": "token", "text": first})
            for text in tokens:
                yield ndjson({"type": "token", "text": text})
        except Exception as e:
            yield ndjson({"type": "error", "detail": llm_http_error(e).detail})
            return
        yield ndjson({"type": "sources", "sources": [src.model_dump() for src in unique_sources(agent)]})

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    """Forget a chat session's memory."""
    with sessions_lock:
        removed = sessions.pop(session_id, None) is not None
    return {"session_id": session_id, "reset": removed}
