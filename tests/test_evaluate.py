"""Unit tests for retrieval eval metrics (fake retriever and reranker, no model or Qdrant)."""

from eval.evaluate import compute_metrics
from src.retrieval.vector_store import SearchResult


def make_result(article_number: int) -> SearchResult:
    """SearchResult whose doc_id equals its article number, for easy expectations."""
    return SearchResult(
        chunk_id=str(article_number),
        doc_id=article_number,
        article_id=f"x#{article_number}",
        article_number=article_number,
        title="Văn bản số X",
        category="x",
        text="Nội dung",
        score=0.0,
    )


class FakeHybrid:
    """Returns articles 1..limit in order and records each requested limit."""

    def __init__(self):
        self.limits: list[int] = []

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.limits.append(limit)
        return [make_result(i) for i in range(1, limit + 1)]


class ReverseReranker:
    """Puts the last candidate first, so a deep answer can surface after reranking."""

    def __init__(self):
        self.pool_sizes: list[int] = []

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        self.pool_sizes.append(len(results))
        return list(reversed(results))[:top_k]


def run(expected_article: int, reranker=None, candidates: int = 30, recall_at=None):
    hybrid = FakeHybrid()
    item = {"question": "q", "expected_doc_ids": [expected_article], "expected_articles": [f"x#{expected_article}"]}
    metrics = compute_metrics(
        [item], None, None, "col", k_list=[1, 2],
        reranker=reranker, hybrid_retriever=hybrid,  # type: ignore[arg-type]
        candidates=candidates, recall_at=recall_at,
    )
    return metrics, hybrid


def test_reranker_gets_exactly_candidates():
    """Every reranked configuration scores the same fixed pool."""
    reranker = ReverseReranker()
    run(1, reranker=reranker, candidates=30)
    assert reranker.pool_sizes == [30]


def test_answer_deep_in_pool_is_found_by_reranker():
    """Answer at position 30 only shows up because the reranker saw all 30 candidates."""
    metrics, _ = run(30, reranker=ReverseReranker(), candidates=30)
    assert metrics["article_hit"][1] == 1.0
    assert metrics["article_mrr"] == 1.0
    assert metrics["rerank_ms_median"] is not None


def test_recall_at_cutoffs():
    """Answer at position 25 is inside the top 30 and 50 but not the top 10 or 20."""
    metrics, hybrid = run(25, recall_at=[10, 20, 30, 50])
    assert metrics["article_recall"] == {10: 0.0, 20: 0.0, 30: 1.0, 50: 1.0}
    assert metrics["doc_recall"][30] == 1.0
    assert hybrid.limits[:4] == [10, 20, 30, 50]


def test_without_reranker_ranks_only_top_k():
    """No reranker: the final ranking is just the retriever's top max_k."""
    metrics, hybrid = run(2)
    assert hybrid.limits == [2]
    assert metrics["article_hit"] == {1: 0.0, 2: 1.0}
    assert metrics["article_mrr"] == 0.5
    assert metrics["candidates"] is None
