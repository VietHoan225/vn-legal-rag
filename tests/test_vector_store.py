"""Unit tests for QdrantVectorStore using mock QdrantClient."""

from unittest.mock import MagicMock
import numpy as np
import pytest
from src.retrieval.vector_store import QdrantVectorStore, SearchResult


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient instance."""
    client = MagicMock()
    # Mock get_collections
    collections_res = MagicMock()
    collections_res.collections = []
    client.get_collections.return_value = collections_res
    return client


def test_chunk_id_to_uuid_deterministic():
    """Test deterministic UUID generation from chunk_id."""
    uuid1 = QdrantVectorStore.chunk_id_to_uuid("0_1")
    uuid2 = QdrantVectorStore.chunk_id_to_uuid("0_1")
    uuid3 = QdrantVectorStore.chunk_id_to_uuid("0_2")
    assert uuid1 == uuid2
    assert uuid1 != uuid3


def test_create_collection(mock_qdrant_client):
    """Test create_collection calls client.create_collection with VectorParams."""
    store = QdrantVectorStore(client=mock_qdrant_client)
    store.create_collection("test_col", vector_size=384, recreate=False)
    mock_qdrant_client.create_collection.assert_called_once()


def test_upsert_chunks(mock_qdrant_client):
    """Test batch upsert into Qdrant."""
    store = QdrantVectorStore(client=mock_qdrant_client)
    chunks = [
        {
            "chunk_id": "1_0",
            "doc_id": 1,
            "article_id": "99/2021/nđ-cp#1",
            "article_number": 1,
            "title": "Văn bản số 99/2021/NĐ-CP",
            "category": "nđ-cp",
            "text": "Nội dung Điều 1",
        },
        {
            "chunk_id": "1_1",
            "doc_id": 1,
            "article_id": "99/2021/nđ-cp#2",
            "article_number": 2,
            "title": "Văn bản số 99/2021/NĐ-CP",
            "category": "nđ-cp",
            "text": "Nội dung Điều 2",
        },
    ]
    vectors = np.ones((2, 384), dtype=np.float32)

    upserted = store.upsert_chunks("test_col", chunks, vectors, batch_size=1)
    assert upserted == 2
    assert mock_qdrant_client.upsert.call_count == 2


def test_search(mock_qdrant_client):
    """Test search transforms Qdrant hits to SearchResult objects."""
    hit = MagicMock()
    hit.score = 0.88
    hit.payload = {
        "chunk_id": "1_0",
        "doc_id": 1,
        "article_id": "99/2021/nđ-cp#1",
        "article_number": 1,
        "title": "Văn bản số 99/2021/NĐ-CP",
        "category": "nđ-cp",
        "text": "Nội dung Điều 1",
    }
    mock_qdrant_client.search.return_value = [hit]

    store = QdrantVectorStore(client=mock_qdrant_client)
    results = store.search("test_col", query_vector=[0.1] * 384, limit=1)

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].doc_id == 1
    assert results[0].article_number == 1
    assert results[0].score == 0.88
