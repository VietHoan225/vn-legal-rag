"""Configuration management using pydantic-settings."""

from typing import Literal
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core service settings
    groq_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    qdrant_collection_name: str = "legal_chunks"

    # Ingest / Chunking settings
    chunk_size: int = 600
    chunk_overlap: int = 80

    # Device selection: auto-detect if None
    device: Literal["cuda", "cpu"] | None = None

    @property
    def resolved_device(self) -> str:
        """Return explicitly configured device or auto-detect cuda/cpu."""
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
