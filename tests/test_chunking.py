"""Unit tests for legal text chunking."""

from src.ingest.chunking import (
    chunk_document,
    chunk_documents,
    normalize_title_to_code,
)
from tests.fixtures.sample_docs import (
    SAMPLE_DOC_WITH_NO_ARTICLES,
    SAMPLE_LEGAL_DOC_1,
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
        # Strip prefix and newline
        prefix = f"{chunk.title} - Điều {chunk.article_number}"
        body = chunk.text[len(prefix) :].strip()
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
