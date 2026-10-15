import json

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app
from src.retrieval.vector_store import SearchResult
from src.generation.generator import GenerationResult

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}

@patch("src.api.main.get_embedder")
@patch("src.api.main.get_hybrid_retriever")
@patch("src.api.main.get_reranker")
@patch("src.api.main.get_generator")
def test_ask_question_success(mock_get_generator, mock_get_reranker, mock_get_hybrid, mock_get_embedder):
    # Setup mocks
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_get_embedder.return_value = mock_embed
    
    mock_search = MagicMock()
    mock_search.search.return_value = [
        SearchResult(
            chunk_id="1", 
            text="Điều 1: Luật này...", 
            score=0.9, 
            doc_id=100,
            article_id="art_1",
            article_number=1,
            title="Điều 1",
            category="general",
            clause_number=2,
        )
    ]
    mock_get_hybrid.return_value = mock_search

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = mock_search.search.return_value
    mock_get_reranker.return_value = mock_reranker

    mock_generate = MagicMock()
    mock_generate.generate_answer.return_value = GenerationResult(
        answer="Theo Điều 1, luật này...",
        prompt_used="test prompt"
    )
    mock_get_generator.return_value = mock_generate

    # Make request
    response = client.post("/ask", json={"question": "Luật này nói gì?"})

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Theo Điều 1, luật này..."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == "1"
    assert data["sources"][0]["content"] == "Điều 1: Luật này..."
    assert data["sources"][0]["metadata"]["clause_number"] == 2
    assert data["sources"][0]["metadata"]["location"] == "Điều 1, Khoản 2"

@patch("src.api.main.get_embedder")
@patch("src.api.main.get_hybrid_retriever")
def test_ask_question_no_results(mock_get_hybrid, mock_get_embedder):
    # Setup mocks
    mock_embed = MagicMock()
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_get_embedder.return_value = mock_embed
    
    mock_search = MagicMock()
    mock_search.search.return_value = []
    mock_get_hybrid.return_value = mock_search

    # Make request
    response = client.post("/ask", json={"question": "Câu hỏi không có trong luật?"})

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "không tìm thấy" in data["answer"].lower()
    assert len(data["sources"]) == 0

def test_ask_question_empty_query():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()


class FakeAgent:
    """Stands in for LegalAgent: echoes messages and remembers how many it got."""

    def __init__(self, sources=None, error: Exception | None = None, fail_after: int = 0):
        self.received: list[str] = []
        self.last_sources = sources or []
        self.error = error
        self.fail_after = fail_after  # chat_stream: tokens yielded before the error is raised

    def chat(self, message: str) -> str:
        if self.error:
            raise self.error
        self.received.append(message)
        return f"trả lời {len(self.received)}"

    def chat_stream(self, message: str):
        for i, token in enumerate(["Xin ", "chào"]):
            if self.error and i == self.fail_after:
                raise self.error
            yield token


@pytest.fixture
def clear_sessions():
    import src.api.main as api_main
    api_main.sessions.clear()
    yield api_main.sessions
    api_main.sessions.clear()


def make_chat_result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, text="Phạt tiền", score=0.03, doc_id=39, article_id="100/2019/nđ-cp#6",
        article_number=6, title="Văn bản số 100/2019/NĐ-CP", category="nđ-cp", clause_number=2,
    )


@patch("src.api.main.create_agent")
def test_chat_reuses_agent_per_session(mock_create_agent, clear_sessions):
    mock_create_agent.side_effect = lambda: FakeAgent()

    r1 = client.post("/chat", json={"session_id": "a", "message": "xin chào"})
    r2 = client.post("/chat", json={"session_id": "a", "message": "câu 2"})
    r3 = client.post("/chat", json={"session_id": "b", "message": "xin chào"})

    assert r1.json()["answer"] == "trả lời 1"
    assert r2.json()["answer"] == "trả lời 2"  # same session keeps memory
    assert r3.json()["answer"] == "trả lời 1"  # new session starts fresh
    assert mock_create_agent.call_count == 2


@patch("src.api.main.create_agent")
def test_chat_returns_unique_sources(mock_create_agent, clear_sessions):
    sources = [make_chat_result("39_1"), make_chat_result("39_1"), make_chat_result("39_2")]
    mock_create_agent.return_value = FakeAgent(sources=sources)

    data = client.post("/chat", json={"session_id": "s", "message": "mũ bảo hiểm"}).json()
    assert [s["id"] for s in data["sources"]] == ["39_1", "39_2"]
    assert data["sources"][0]["metadata"]["location"] == "Điều 6, Khoản 2"


@patch("src.api.main.create_agent")
def test_chat_rate_limit_returns_429(mock_create_agent, clear_sessions):
    import httpx
    import openai

    response = httpx.Response(429, request=httpx.Request("POST", "https://openrouter.ai/api/v1"))
    mock_create_agent.return_value = FakeAgent(error=openai.RateLimitError("limit", response=response, body=None))

    r = client.post("/chat", json={"session_id": "s", "message": "hỏi"})
    assert r.status_code == 429
    assert "quá tải" in r.json()["detail"]


def test_chat_empty_message():
    r = client.post("/chat", json={"session_id": "s", "message": "  "})
    assert r.status_code == 400


@patch("src.api.main.create_agent")
def test_reset_chat_forgets_session(mock_create_agent, clear_sessions):
    mock_create_agent.side_effect = lambda: FakeAgent()
    client.post("/chat", json={"session_id": "a", "message": "câu 1"})

    assert client.delete("/chat/a").json()["reset"] is True
    r = client.post("/chat", json={"session_id": "a", "message": "câu 1 lại"})
    assert r.json()["answer"] == "trả lời 1"


@patch("src.api.main.create_agent")
def test_sessions_are_capped(mock_create_agent, clear_sessions, monkeypatch):
    import src.api.main as api_main
    monkeypatch.setattr(api_main, "MAX_SESSIONS", 2)
    mock_create_agent.side_effect = lambda: FakeAgent()
    for sid in ["a", "b", "c"]:
        client.post("/chat", json={"session_id": sid, "message": "hi"})
    assert list(clear_sessions) == ["b", "c"]


def rate_limit_error():
    import httpx
    import openai

    response = httpx.Response(429, request=httpx.Request("POST", "https://openrouter.ai/api/v1"))
    return openai.RateLimitError("limit", response=response, body=None)


def read_events(response) -> list[dict]:
    """Parse a newline-delimited JSON streaming response."""
    return [json.loads(line) for line in response.text.splitlines() if line]


@patch("src.api.main.create_agent")
def test_chat_stream_sends_tokens_then_sources(mock_create_agent, clear_sessions):
    mock_create_agent.return_value = FakeAgent(sources=[make_chat_result("39_1"), make_chat_result("39_1")])

    r = client.post("/chat/stream", json={"session_id": "s", "message": "mũ bảo hiểm"})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = read_events(r)
    assert [e["type"] for e in events] == ["token", "token", "sources"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Xin chào"
    assert [src["id"] for src in events[-1]["sources"]] == ["39_1"]
    assert events[-1]["sources"][0]["metadata"]["location"] == "Điều 6, Khoản 2"


@patch("src.api.main.create_agent")
def test_chat_stream_error_before_first_token_is_http_error(mock_create_agent, clear_sessions):
    mock_create_agent.return_value = FakeAgent(error=rate_limit_error(), fail_after=0)

    r = client.post("/chat/stream", json={"session_id": "s", "message": "hỏi"})
    assert r.status_code == 429
    assert "quá tải" in r.json()["detail"]


@patch("src.api.main.create_agent")
def test_chat_stream_error_mid_stream_is_an_event(mock_create_agent, clear_sessions):
    mock_create_agent.return_value = FakeAgent(error=rate_limit_error(), fail_after=1)

    r = client.post("/chat/stream", json={"session_id": "s", "message": "hỏi"})
    assert r.status_code == 200  # headers were already sent with the first token
    events = read_events(r)
    assert [e["type"] for e in events] == ["token", "error"]
    assert "quá tải" in events[1]["detail"]


def test_chat_stream_empty_message():
    r = client.post("/chat/stream", json={"session_id": "s", "message": "  "})
    assert r.status_code == 400


@patch("src.api.main.create_agent")
def test_chat_stream_provider_error_returns_503(mock_create_agent, clear_sessions):
    import httpx
    import openai

    error = openai.APIError("Upstream error: Service temporarily overloaded", request=httpx.Request("POST", "https://x"), body=None)
    mock_create_agent.return_value = FakeAgent(error=error, fail_after=0)

    r = client.post("/chat/stream", json={"session_id": "s", "message": "hỏi"})
    assert r.status_code == 503
    assert "gián đoạn" in r.json()["detail"]


@patch("src.api.main.create_agent")
def test_chat_stream_search_failure_returns_503(mock_create_agent, clear_sessions):
    from src.agent.tools import SearchError

    mock_create_agent.return_value = FakeAgent(error=SearchError("qdrant down"), fail_after=0)

    r = client.post("/chat/stream", json={"session_id": "s", "message": "hỏi"})
    assert r.status_code == 503
    assert "cơ sở dữ liệu pháp luật" in r.json()["detail"]


@patch("src.api.main.create_agent")
def test_chat_search_failure_returns_503(mock_create_agent, clear_sessions):
    from src.agent.tools import SearchError

    mock_create_agent.return_value = FakeAgent(error=SearchError("qdrant down"))
    r = client.post("/chat", json={"session_id": "s", "message": "hỏi"})
    assert r.status_code == 503
