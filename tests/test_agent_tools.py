"""Unit tests for agent tools (no model or LLM calls)."""

import pytest

from src.agent.tools import LegalSearchTool, SearchError
from src.retrieval.vector_store import SearchResult


def make_result(article_number: int, clause_number: int | None) -> SearchResult:
    """Build a minimal SearchResult."""
    return SearchResult(
        chunk_id="3_0",
        doc_id=3,
        article_id=f"01/2099/nđ-cp#{article_number}",
        article_number=article_number,
        title="Văn bản số 01/2099/NĐ-CP",
        category="nđ-cp",
        text="Nội dung",
        score=1.0,
        clause_number=clause_number,
    )


class FakeRetriever:
    """Retriever stub returning fixed results."""

    def __init__(self, results: list[SearchResult]):
        self.results = results

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.results


def test_location_with_and_without_clause():
    """location() includes the clause only when known."""
    assert make_result(6, 2).location() == "Điều 6, Khoản 2"
    assert make_result(6, None).location() == "Điều 6"


def test_tool_output_cites_clause():
    """Tool output exposes article and clause so the agent can cite precisely."""
    tool = LegalSearchTool(FakeRetriever([make_result(6, 2), make_result(1, None)]))  # type: ignore[arg-type]
    out = tool.execute("mũ bảo hiểm")
    assert "[1] Văn bản số 01/2099/NĐ-CP (Điều 6, Khoản 2)" in out
    assert "[2] Văn bản số 01/2099/NĐ-CP (Điều 1)" in out


def test_tool_empty_results():
    """Tool returns a not-found message when retrieval is empty."""
    tool = LegalSearchTool(FakeRetriever([]))  # type: ignore[arg-type]
    assert "Không tìm thấy" in tool.execute("abc")


class FakeReranker:
    """Reranker stub: reverses the candidates and keeps top_k."""

    def __init__(self):
        self.calls: list[tuple[str, int, int]] = []

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, len(results), top_k))
        return list(reversed(results))[:top_k]


class LimitRetriever:
    """Retriever stub that records the limit it was asked for and returns `limit` results."""

    def __init__(self):
        self.limits: list[int] = []

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.limits.append(limit)
        return [make_result(i, None) for i in range(1, limit + 1)]


class BrokenRetriever:
    def search(self, query: str, limit: int = 5):
        raise ConnectionError("qdrant down")


def test_tool_without_reranker_fetches_top_k():
    retriever = LimitRetriever()
    tool = LegalSearchTool(retriever, top_k=4)  # type: ignore[arg-type]
    assert len(tool.search("q")) == 4
    assert retriever.limits == [4]


def test_tool_with_reranker_fetches_more_then_keeps_top_k():
    retriever, reranker = LimitRetriever(), FakeReranker()
    tool = LegalSearchTool(retriever, reranker, top_k=5)  # type: ignore[arg-type]
    results = tool.search("mũ bảo hiểm")
    assert retriever.limits == [15]
    assert reranker.calls == [("mũ bảo hiểm", 15, 5)]
    assert [r.article_number for r in results] == [15, 14, 13, 12, 11]  # reranker's order is kept


def test_tool_wraps_retrieval_failure_in_search_error():
    tool = LegalSearchTool(BrokenRetriever())  # type: ignore[arg-type]
    with pytest.raises(SearchError, match="qdrant down"):
        tool.search("q")
