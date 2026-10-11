"""Unit tests for legal text chunking."""

from src.ingest.chunking import (
    chunk_document,
    chunk_documents,
    normalize_title_to_code,
    split_clauses,
    split_points,
    truncate_at_word,
)
from tests.fixtures.sample_docs import (
    SAMPLE_DOC_WITH_NO_ARTICLES,
    SAMPLE_LEGAL_DOC_1,
    SAMPLE_PENALTY_DOC,
)


def test_normalize_title_to_code():
    """Test extracting normalized law code from various titles."""
    assert normalize_title_to_code("Văn bản số 99/2021/NĐ-CP") == "99/2021/nđ-cp"
    assert normalize_title_to_code("Văn bản số 01/2013/TT-BVHTTDL") == "01/2013/tt-bvhttdl"
    assert normalize_title_to_code("41/2019/QH14") == "41/2019/qh14"


def test_prefix_present_on_all_chunks():
    """Test that all chunks contain the required prefix '<title> - Điều N'."""
    chunks = chunk_document(SAMPLE_LEGAL_DOC_1, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 0
    for chunk in chunks:
        expected_prefix = f"{chunk.title} - Điều {chunk.article_number}"
        assert chunk.text.startswith(expected_prefix), (
            f"Chunk {chunk.chunk_id} missing prefix: {expected_prefix}"
        )


def test_no_chunk_exceeds_limit():
    """Test that no chunk exceeds chunk_size limit for multiple threshold settings."""
    for chunk_size in [350, 450, 600, 800]:
        chunks = chunk_document(
            SAMPLE_LEGAL_DOC_1,
            chunk_size=chunk_size,
            chunk_overlap=50,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size, (
                f"Chunk {chunk.chunk_id} length {len(chunk.text)} exceeds {chunk_size}"
            )


def test_no_chunk_starts_mid_sentence():
    """Test that no chunk body starts mid-sentence with cut words or lowercase fragments."""
    chunks = chunk_document(SAMPLE_LEGAL_DOC_1, chunk_size=400, chunk_overlap=60)
    for chunk in chunks:
        # Strip prefix line ('<title> - Điều N' or '<title> - Điều N - Khoản K')
        body = chunk.text.split("\n", 1)[1].strip()
        first_char = body[0]
        # Body must start with an uppercase letter, a digit (e.g. clause '1.'), or 'Điều'
        assert first_char.isupper() or first_char.isdigit(), (
            f"Chunk {chunk.chunk_id} appears to start mid-sentence: '{body[:30]}...'"
        )


def test_inline_references_not_split():
    """Test that inline references like 'theo Điều 5 của Luật này' are not split as separate articles."""
    chunks = chunk_document(SAMPLE_LEGAL_DOC_1, chunk_size=600, chunk_overlap=80)
    article_numbers = {c.article_number for c in chunks}
    # Only Điều 1, 2, 3 should exist. Inline Điều 5, 8, 12 must not be treated as separate articles.
    assert article_numbers == {1, 2, 3}
    assert 5 not in article_numbers
    assert 12 not in article_numbers
    assert 8 not in article_numbers


def test_reasonable_chunk_count():
    """Test that number of chunks is reasonable depending on chunk_size."""
    # With very large chunk_size, each of the 3 articles is exactly 1 chunk
    large_chunks = chunk_document(SAMPLE_LEGAL_DOC_1, chunk_size=5000, chunk_overlap=80)
    assert len(large_chunks) == 3

    # With small chunk_size (300 chars), long Article 3 splits into multiple chunks
    small_chunks = chunk_document(SAMPLE_LEGAL_DOC_1, chunk_size=300, chunk_overlap=50)
    assert len(small_chunks) > 3


def test_doc_with_no_articles_returns_empty():
    """Test graceful handling of document without 'Điều N.' headers."""
    chunks = chunk_document(SAMPLE_DOC_WITH_NO_ARTICLES)
    assert chunks == []


def test_chunk_documents_batch():
    """Test chunk_documents with multiple docs."""
    chunks = chunk_documents([SAMPLE_LEGAL_DOC_1, SAMPLE_DOC_WITH_NO_ARTICLES])
    assert len(chunks) >= 3
    assert all(c.doc_id == 1 for c in chunks)


def test_points_keep_clause_penalty():
    """A point must stay with its own clause's penalty range, never a neighbour's."""
    chunks = chunk_document(SAMPLE_PENALTY_DOC, chunk_size=400, chunk_overlap=60)
    helmet = [c for c in chunks if "mũ bảo hiểm" in c.text]
    assert len(helmet) == 1
    chunk = helmet[0]
    assert chunk.article_number == 2
    assert chunk.clause_number == 2
    assert "200.000 đồng đến 300.000 đồng" in chunk.text
    assert "100.000 đồng đến 200.000 đồng" not in chunk.text


def test_long_clause_split_repeats_lead_in_and_heading():
    """Every sub-chunk of a split clause repeats the article heading and clause lead-in."""
    chunks = chunk_document(SAMPLE_PENALTY_DOC, chunk_size=400, chunk_overlap=60)
    clause1 = [c for c in chunks if c.article_number == 2 and c.clause_number == 1]
    assert len(clause1) > 1
    for c in clause1:
        assert c.text.startswith("Văn bản số 01/2099/NĐ-CP - Điều 2 - Khoản 1\n")
        assert "Điều 2. Xử phạt người điều khiển xe mô tô" in c.text
        assert "Phạt tiền từ 100.000 đồng đến 200.000 đồng" in c.text
        assert len(c.text) <= 400


def test_points_not_cut():
    """Each point line appears whole in exactly one chunk when points fit the limit."""
    chunks = chunk_document(SAMPLE_PENALTY_DOC, chunk_size=400, chunk_overlap=60)
    _, clauses = split_clauses(SAMPLE_PENALTY_DOC["content"].split("Điều 3.")[0])
    for _, clause_text in clauses:
        _, points = split_points(clause_text)
        for pt in points:
            assert sum(pt in c.text for c in chunks) == 1, pt


def test_split_clauses_and_points():
    """Clause and point splitting return lead-in, numbers and point texts."""
    art = "Điều 9. Tên điều\n1. Phạt tiền từ 1 đồng đến 2 đồng đối với:\na) Hành vi A;\nđ) Hành vi B.\n2. Khoản không có điểm."
    preamble, clauses = split_clauses(art)
    assert preamble == "Điều 9. Tên điều"
    assert [n for n, _ in clauses] == [1, 2]
    lead_in, points = split_points(clauses[0][1])
    assert lead_in == "1. Phạt tiền từ 1 đồng đến 2 đồng đối với:"
    assert points == ["a) Hành vi A;", "đ) Hành vi B."]
    assert split_points(clauses[1][1]) == ("2. Khoản không có điểm.", [])


def test_short_article_and_no_clause_fallback():
    """Short articles stay whole; long articles without clauses use sentence windows."""
    chunks = chunk_document(SAMPLE_PENALTY_DOC, chunk_size=400, chunk_overlap=60)
    art1 = [c for c in chunks if c.article_number == 1]
    assert len(art1) == 1 and art1[0].clause_number is None
    art3 = [c for c in chunks if c.article_number == 3]
    assert len(art3) > 1
    assert all(c.clause_number is None and len(c.text) <= 400 for c in art3)


def test_truncate_at_word():
    """Truncation respects the limit and does not cut words."""
    assert truncate_at_word("ngắn", 10) == "ngắn"
    out = truncate_at_word("Xử phạt người điều khiển xe mô tô", 15)
    assert len(out) <= 15 and out.endswith("…")
    assert out[:-1] in "Xử phạt người điều khiển xe mô tô" and not out[:-1].endswith(" ")
