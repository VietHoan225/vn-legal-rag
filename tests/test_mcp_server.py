"""Unit tests for the MCP server (fake search backend, no models or Qdrant)."""

import asyncio
import json
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.agent.tools import SearchError
from src.mcp_server.server import DATA_NOTE, MAX_TOP_K, LazyValue, create_server
from src.retrieval.vector_store import SearchResult


def make_hit(article: int, clause: int | None = None, score: float = 1.23456) -> SearchResult:
    return SearchResult(
        chunk_id=f"c{article}",
        doc_id=1,
        article_id=f"100/2019/nđ-cp#{article}",
        article_number=article,
        title="Văn bản số 100/2019/NĐ-CP",
        category="nđ-cp",
        text=f"Nội dung Điều {article}",
        score=score,
        clause_number=clause,
    )


class FakeSearch:
    """Search backend stub: records calls, returns `hits` or raises `error`."""

    def __init__(self, hits: list[SearchResult] | None = None, error: Exception | None = None):
        self.hits = hits or []
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        return self.hits


def call(server: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool through the MCP server and return its structured result."""
    content = asyncio.run(server.call_tool(name, args))
    if isinstance(content, tuple):  # (unstructured content, structured result)
        return content[1]
    return json.loads(content[0].text)


def test_server_lists_search_law_with_guidance_in_description():
    server = create_server(lambda: FakeSearch())
    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools] == ["search_law"]
    desc = tools[0].description
    assert "đèn tín hiệu giao thông" in desc  # query-writing hint
    assert "lỗi thời" in desc  # outdated-data warning
    assert set(tools[0].inputSchema["properties"]) == {"query", "top_k"}
    assert tools[0].inputSchema["required"] == ["query"]


def test_search_law_returns_compact_structured_hits():
    fake = FakeSearch([make_hit(6, 4), make_hit(47)])
    out = call(create_server(lambda: fake), "search_law", {"query": "vượt đèn đỏ xe máy", "top_k": 2})
    assert out["count"] == 2
    first = out["results"][0]
    assert first == {
        "rank": 1,
        "law": "100/2019/NĐ-CP",
        "article": 6,
        "clause": 4,
        "citation": "100/2019/NĐ-CP, Điều 6, Khoản 4",
        "relevance_score": 1.235,
        "text": "Nội dung Điều 6",
    }
    assert out["results"][1]["clause"] is None
    assert out["results"][1]["citation"] == "100/2019/NĐ-CP, Điều 47"
    assert out["note"] == DATA_NOTE
    assert fake.calls == [("vượt đèn đỏ xe máy", 2)]


@pytest.mark.parametrize("requested, expected", [(0, 1), (-3, 1), (3, 3), (500, MAX_TOP_K)])
def test_search_law_clamps_top_k(requested: int, expected: int):
    fake = FakeSearch()
    call(create_server(lambda: fake), "search_law", {"query": "q", "top_k": requested})
    assert fake.calls == [("q", expected)]


def test_search_law_default_top_k_is_five():
    fake = FakeSearch()
    call(create_server(lambda: fake), "search_law", {"query": "q"})
    assert fake.calls == [("q", 5)]


def test_search_law_no_hits_is_an_empty_result_not_an_error():
    out = call(create_server(lambda: FakeSearch([])), "search_law", {"query": "q"})
    assert out["count"] == 0 and out["results"] == []
    assert out["note"] == DATA_NOTE


def test_search_law_rejects_blank_query():
    with pytest.raises(ToolError, match="must not be empty"):
        call(create_server(lambda: FakeSearch()), "search_law", {"query": "   "})


def test_search_failure_is_a_tool_error_not_an_empty_result():
    fake = FakeSearch(error=SearchError("qdrant down"))
    with pytest.raises(ToolError, match="không có nghĩa là không có quy định"):
        call(create_server(lambda: fake), "search_law", {"query": "q"})


def test_lazy_value_builds_once_and_retries_after_failure():
    attempts = []

    def factory() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("not ready")
        return "ready"

    lazy = LazyValue(factory)
    with pytest.raises(ConnectionError):
        lazy.get()
    assert lazy.get() == "ready"
    assert lazy.get() == "ready"
    assert len(attempts) == 2  # one failed build, one successful build, then cached
