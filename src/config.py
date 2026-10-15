"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Literal
import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, so .env and data paths do not depend on the working directory
# (an MCP client may start the server from anywhere).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core service settings
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    qdrant_collection_name: str = "legal_chunks"

    # BM25 index directory; a relative path is resolved against PROJECT_ROOT
    bm25_dir: Path = Path("data/processed/bm25_index")

    # Candidates the retriever hands to the reranker, which keeps the final top_k
    rerank_candidates: int = 30

    # Ingest / Chunking settings
    chunk_size: int = 600
    chunk_overlap: int = 80

    # Device selection: auto-detect if None
    device: Literal["cuda", "cpu"] | None = None

    @field_validator("bm25_dir")
    @classmethod
    def _resolve_against_root(cls, v: Path) -> Path:
        """Anchor relative paths at the repo root instead of the working directory."""
        return v if v.is_absolute() else PROJECT_ROOT / v

    @property
    def resolved_device(self) -> str:
        """Return explicitly configured device or auto-detect cuda/cpu."""
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
