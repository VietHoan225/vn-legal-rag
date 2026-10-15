"""Ingest script: embed processed chunks with caching and upsert to Qdrant idempotently."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load JSONL chunks from disk."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Processed chunks not found at: {chunks_path}. Run scripts/prepare_data.py first."
        )

    logger.info("Loading chunks from: %s", chunks_path)
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks.", len(chunks))
    return chunks


def get_or_compute_embeddings(
    chunks: list[dict[str, Any]],
    cache_path: Path,
    force_recompute: bool = False,
    batch_size: int = 256,
) -> np.ndarray:
    """Load cached embeddings or compute and cache them."""
    if cache_path.exists() and not force_recompute:
        logger.info("Found cached embeddings at: %s. Loading...", cache_path)
        embeddings = np.load(cache_path)
        if len(embeddings) == len(chunks):
            logger.info("Successfully loaded cached embeddings of shape %s", embeddings.shape)
            return embeddings
        logger.warning(
            "Cache size mismatch (cache has %d, chunks have %d). Recomputing...",
            len(embeddings),
            len(chunks),
        )

    logger.info("Computing embeddings for %d chunks with Embedder...", len(chunks))
    embedder = Embedder()
    texts = [c["text"] for c in chunks]

    start_time = time.time()
    embeddings = embedder.embed_texts(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    elapsed = time.time() - start_time
    logger.info("Embedding computed in %.2f seconds (shape %s).", elapsed, embeddings.shape)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info("Cached embeddings saved to: %s", cache_path)
    return embeddings


def ingest(
    chunks_path: Path,
    cache_path: Path,
    collection_name: str,
    recreate: bool = False,
    force_reembed: bool = False,
    batch_size: int = 500,
    embed_batch_size: int = 256,
) -> int:
    """Ingest chunks and embeddings into Qdrant."""
    chunks = load_chunks(chunks_path)
    embeddings = get_or_compute_embeddings(
        chunks=chunks,
        cache_path=cache_path,
        force_recompute=force_reembed,
        batch_size=embed_batch_size,
    )

    vector_store = QdrantVectorStore()
    vector_dim = embeddings.shape[1]

    # Ensure collection exists
    vector_store.create_collection(
        collection_name=collection_name,
        vector_size=vector_dim,
        recreate=recreate,
    )

    logger.info("Upserting %d points into collection '%s'...", len(chunks), collection_name)
    start_time = time.time()
    upserted = vector_store.upsert_chunks(
        collection_name=collection_name,
        chunks=chunks,
        vectors=embeddings,
        batch_size=batch_size,
    )
    elapsed = time.time() - start_time
    logger.info("Upsert completed in %.2f seconds.", elapsed)

    total_count = vector_store.count(collection_name)
    logger.info("Total points currently in collection '%s': %d", collection_name, total_count)
    return total_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest chunks into Qdrant vector store.")
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to processed chunks JSONL file.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/processed/embeddings.npy"),
        help="Path to embeddings cache .npy file.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=settings.qdrant_collection_name,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if it already exists.",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Ignore cached embeddings and recompute.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Qdrant upserts.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Batch size for embedding; lower it for large models (e.g. 32 for BAAI/bge-m3 on a 6GB GPU).",
    )

    args = parser.parse_args()
    ingest(
        chunks_path=args.chunks_file,
        cache_path=args.cache_file,
        collection_name=args.collection_name,
        recreate=args.recreate,
        force_reembed=args.force_reembed,
        batch_size=args.batch_size,
        embed_batch_size=args.embed_batch_size,
    )


if __name__ == "__main__":
    main()
