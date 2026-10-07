"""Unit tests for Embedder using mocks without downloading models."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.retrieval.embedder import Embedder


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid loading weights or downloading during test."""
    with patch("src.retrieval.embedder.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.max_seq_length = 128
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        # Mock tokenizer encode
        mock_tokenizer = MagicMock()
        mock_instance.tokenizer = mock_tokenizer

        # Mock encode method
        def fake_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.ones(384, dtype=np.float32)
            return np.ones((len(texts), 384), dtype=np.float32)

        mock_instance.encode.side_effect = fake_encode
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


def test_embedder_initialization(mock_sentence_transformer):
    """Test embedder initialization using mocked model."""
    mock_cls, mock_instance = mock_sentence_transformer
    embedder = Embedder(model_name="custom/model", device="cpu")

    mock_cls.assert_called_once_with("custom/model", device="cpu")
    assert embedder.max_seq_length == 128
    assert embedder.embedding_dim == 384


def test_embedder_warning_on_token_limit(mock_sentence_transformer):
    """Test warning issued when tokens exceed max_seq_length."""
    _, mock_instance = mock_sentence_transformer
    # Simulate text returning 200 tokens (limit is 128)
    mock_instance.tokenizer.encode.return_value = [1] * 200

    embedder = Embedder(model_name="test-model", device="cpu")
    with pytest.warns(UserWarning, match="exceeds max_seq_length"):
        embedder.check_token_length("Đoạn văn bản pháp luật rất dài...")


def test_embed_texts_and_query(mock_sentence_transformer):
    """Test embed_texts and embed_query return correct shapes and types."""
    _, mock_instance = mock_sentence_transformer
    embedder = Embedder(model_name="test-model", device="cpu")

    # Embed batch
    texts = ["Văn bản 1", "Văn bản 2", "Văn bản 3"]
    vectors = embedder.embed_texts(texts, batch_size=2)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 384)

    # Embed single query
    query_vec = embedder.embed_query("Quy định về an toàn giao thông?")
    assert isinstance(query_vec, list)
    assert len(query_vec) == 384
