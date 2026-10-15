"""Unit tests for HybridRetriever fusion (fake stores, no model or Qdrant)."""

from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.vector_store import SearchResult


def make_result(chunk_id: str) -> SearchResult:
    """Build a minimal SearchResult identified by chunk_id."""
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=1,
        article_id=f"01/2099/nđ-cp#{chunk_id}",
        article_number=1,
        title="Văn bản số 01/2099/NĐ-CP",
        category="nđ-cp",
        text=f"Nội dung {chunk_id}",
        score=0.0,
    )


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.0]


class FakeVectorStore:
    """Dense branch: returns the first `limit` of a fixed ranking."""

    def __init__(self, chunk_ids: list[str]):
        self.chunk_ids = chunk_ids

    def search(self, collection_name: str, query_vector, limit: int = 4, category=None) -> list[SearchResult]:
        return [make_result(c) for c in self.chunk_ids[:limit]]


class FakeBM25:
    """Sparse branch: mimics bm25s.retrieve, returning (docs, scores) of shape (1, k)."""

    def __init__(self, chunk_ids: list[str]):
        self.chunk_ids = chunk_ids

    def retrieve(self, query_tokens, k: int):
        ids = self.chunk_ids[:k]
        return [[make_result(c).to_dict() for c in ids]], [[1.0] * len(ids)]


def make_retriever(tmp_path, dense: list[str], sparse: list[str] | None) -> HybridRetriever:
    """HybridRetriever with fake branches; a missing bm25_dir means dense only."""
    retriever = HybridRetriever(FakeVectorStore(dense), FakeEmbedder(), bm25_dir=tmp_path / "missing")  # type: ignore[arg-type]
    if sparse is not None:
        retriever.bm25_retriever = FakeBM25(sparse)  # type: ignore[assignment]
    return retriever


def test_search_returns_at_most_limit(tmp_path):
    """Two disjoint branches of 5 hits fuse to 10, but only `limit` come back."""
    retriever = make_retriever(tmp_path, dense=list("ABCDE"), sparse=list("FGHIJ"))
    assert len(retriever.search("q", limit=5)) == 5


def test_chunk_found_by_both_branches_is_kept_once_and_ranked_first(tmp_path):
    """RRF adds both branch scores, so a shared chunk outranks single-branch top hits."""
    retriever = make_retriever(tmp_path, dense=list("ABCDE"), sparse=list("CFGHI"))
    ids = [r.chunk_id for r in retriever.search("q", limit=5)]
    assert ids[0] == "C"
    assert ids.count("C") == 1


def test_dense_only_when_bm25_missing(tmp_path):
    """Without a BM25 index the retriever falls back to the dense ranking."""
    retriever = make_retriever(tmp_path, dense=list("ABCDE"), sparse=None)
    assert retriever.bm25_retriever is None
    assert [r.chunk_id for r in retriever.search("q", limit=3)] == ["A", "B", "C"]
