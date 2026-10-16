"""Unit tests for configuration loading."""

from unittest.mock import patch
from src.config import PROJECT_ROOT, Settings


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


def test_env_file_is_anchored_at_project_root():
    """.env is looked up in the repo, not in whatever directory the process starts from."""
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_bm25_dir_independent_of_cwd(tmp_path, monkeypatch):
    """Starting from another directory (as an MCP client may) still finds the repo's BM25 index path."""
    monkeypatch.chdir(tmp_path)
    cfg = Settings(_env_file=None)
    assert cfg.bm25_dir == PROJECT_ROOT / "data/processed/bm25_index"


def test_bm25_dir_relative_and_absolute_overrides(tmp_path):
    """A relative override resolves against the repo root; an absolute one is kept."""
    assert Settings(_env_file=None, bm25_dir="custom/bm25").bm25_dir == PROJECT_ROOT / "custom/bm25"
    assert Settings(_env_file=None, bm25_dir=tmp_path).bm25_dir == tmp_path


def test_raw_data_path_resolved_against_project_root(tmp_path, monkeypatch):
    """The raw data path, like bm25_dir, does not depend on the working directory."""
    monkeypatch.chdir(tmp_path)
    assert Settings(_env_file=None).raw_data_path == PROJECT_ROOT / "data/raw/van-ban-phap-luat.json"
    assert Settings(_env_file=None, raw_data_path=tmp_path / "x.json").raw_data_path == tmp_path / "x.json"
