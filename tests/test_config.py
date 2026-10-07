"""Unit tests for configuration loading."""

from unittest.mock import patch
from src.config import Settings


def test_default_settings():
    """Test default configuration values without .env."""
    cfg = Settings(_env_file=None)
    assert cfg.qdrant_url == "http://localhost:6333"
    assert "MiniLM" in cfg.embedding_model
    assert cfg.chunk_size == 600
    assert cfg.chunk_overlap == 80


def test_custom_settings():
    """Test custom settings passing arguments."""
    cfg = Settings(
        _env_file=None,
        qdrant_url="http://custom-qdrant:6333",
        chunk_size=1000,
        chunk_overlap=150,
        device="cpu",
    )
    assert cfg.qdrant_url == "http://custom-qdrant:6333"
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 150
    assert cfg.resolved_device == "cpu"


def test_resolved_device_cuda_auto():
    """Test auto-detection when cuda is available."""
    with patch("torch.cuda.is_available", return_value=True):
        cfg = Settings(_env_file=None, device=None)
        assert cfg.resolved_device == "cuda"


def test_resolved_device_cpu_fallback():
    """Test auto-detection fallback when cuda is not available."""
    with patch("torch.cuda.is_available", return_value=False):
        cfg = Settings(_env_file=None, device=None)
        assert cfg.resolved_device == "cpu"
