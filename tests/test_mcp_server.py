"""Unit tests for the MCP server (fake search backend, no models or Qdrant)."""

import asyncio
import json
from typing import Any

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from src.agent.tools import SearchError
from src.mcp_server.law_catalog import LawCatalog
from src.mcp_server.server import DATA_NOTE, MAX_ARTICLE_CHARS, MAX_TOP_K, LazyValue, create_server
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


LONG_LINE = "Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với hành vi vi phạm quy định."
CATALOG = LawCatalog([
    {
        "doc_id": 1,
        "title": "Văn bản số 100/2019/NĐ-CP",
        "category": "nđ-cp",
        "content": "Điều 1. Ngắn\nNội dung ngắn.\nĐiều 2. Dài\n" + "\n".join(f"{i}. {LONG_LINE}" for i in range(1, 200)),
    },
    {"doc_id": 2, "title": "Văn bản số 01/2020/TT-BTC", "category": "tt-btc", "content": "Điều 1. Phạm vi\nNội dung."},
])


def make_server(fake: "FakeSearch | None" = None):
    fake = fake or FakeSearch()
    return create_server(lambda: fake, lambda: CATALOG)


def call(server: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool through the MCP server and return its structured result."""
    content = asyncio.run(server.call_tool(name, args))
    if isinstance(content, tuple):  # (unstructured content, structured result)
        return content[1]
    return json.loads(content[0].text)


def test_server_lists_three_tools_with_guidance_in_descriptions():
    server = make_server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) == {"search_law", "get_article", "list_laws"}
    for tool in tools.values():
        assert "lỗi thời" in tool.description  # outdated-data warning on every tool
    assert set(tools["get_article"].inputSchema["properties"]) == {"law_name", "article_number"}
    assert tools["list_laws"].inputSchema.get("properties", {}) == {}
    search = tools["search_law"]
    desc = search.description
    assert "đèn tín hiệu giao thông" in desc  # query-writing hint
    assert set(search.inputSchema["properties"]) == {"query", "top_k"}
    assert search.inputSchema["required"] == ["query"]


def test_search_law_returns_compact_structured_hits():
    fake = FakeSearch([make_hit(6, 4), make_hit(47)])
    out = call(make_server(fake), "search_law", {"query": "vượt đèn đỏ xe máy", "top_k": 2})
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
    call(make_server(fake), "search_law", {"query": "q", "top_k": requested})
    assert fake.calls == [("q", expected)]


def test_search_law_default_top_k_is_five():
    fake = FakeSearch()
    call(make_server(fake), "search_law", {"query": "q"})
    assert fake.calls == [("q", 5)]


def test_search_law_no_hits_is_an_empty_result_not_an_error():
    out = call(make_server(FakeSearch([])), "search_law", {"query": "q"})
    assert out["count"] == 0 and out["results"] == []
    assert out["note"] == DATA_NOTE


def test_search_law_rejects_blank_query():
    with pytest.raises(ToolError, match="must not be empty"):
        call(make_server(), "search_law", {"query": "   "})


def test_search_failure_is_a_tool_error_not_an_empty_result():
    fake = FakeSearch(error=SearchError("qdrant down"))
    with pytest.raises(ToolError, match="không có nghĩa là không có quy định"):
        call(make_server(fake), "search_law", {"query": "q"})


def test_get_article_returns_full_short_article():
    out = call(make_server(), "get_article", {"law_name": "100/2019/nd-cp", "article_number": 1})
    assert out["law"] == "100/2019/NĐ-CP"
    assert out["text"] == "Điều 1. Ngắn\nNội dung ngắn."
    assert out["truncated"] is False and "truncation_note" not in out
    assert out["note"] == DATA_NOTE


def test_get_article_truncates_long_article_at_a_line_break():
    out = call(make_server(), "get_article", {"law_name": "100/2019/NĐ-CP", "article_number": 2})
    assert out["truncated"] is True
    assert len(out["text"]) <= MAX_ARTICLE_CHARS
    assert out["total_chars"] > MAX_ARTICLE_CHARS
    assert out["text"].endswith(LONG_LINE)  # cut between lines, not mid-sentence
    assert "search_law" in out["truncation_note"]


def test_get_article_unknown_law_and_article_are_tool_errors():
    server = make_server()
    with pytest.raises(ToolError, match="Không tìm thấy văn bản"):
        call(server, "get_article", {"law_name": "999/2099/XX", "article_number": 1})
    with pytest.raises(ToolError, match="không có Điều 7"):
        call(server, "get_article", {"law_name": "01/2020/TT-BTC", "article_number": 7})


def test_list_laws_groups_numbers_by_category():
    out = call(make_server(), "list_laws", {})
    assert out["total"] == 2
    assert out["categories"] == {"nđ-cp": ["100/2019/NĐ-CP"], "tt-btc": ["01/2020/TT-BTC"]}
    assert out["note"] == DATA_NOTE


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
