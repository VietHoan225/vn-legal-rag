"""Unit tests for agent tools (no model or LLM calls)."""

from src.agent.tools import LegalSearchTool
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
