"""Prepare data script: clean and chunk raw legal documents to JSONL."""

import argparse
import json
import logging
import sys
import time
from typing import Any
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingest.chunking import chunk_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    raw_path: Path,
    output_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Load raw legal documents, chunk them, and write to JSONL file.

    Args:
        raw_path: Path to raw van-ban-phap-luat.json.
        output_path: Path to write processed chunks.jsonl.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Target overlap size in characters.

    Returns:
        Total number of chunks produced.
    """
    logger.info("Loading raw legal documents from: %s", raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        docs: list[dict[str, Any]] = json.load(f)

    logger.info("Loaded %d documents. Beginning chunking...", len(docs))
    start_time = time.time()

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elapsed = time.time() - start_time

    logger.info(
        "Chunking complete in %.2f seconds. Total chunks generated: %d",
        elapsed,
        len(chunks),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    logger.info("Saved %d chunks to: %s", len(chunks), output_path)

    if chunks:
        lens = [len(c.text) for c in chunks]
        logger.info(
            "Chunk character stats: min=%d, max=%d, avg=%d",
            min(lens),
            max(lens),
            sum(lens) // len(lens),
        )

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and chunk legal data into JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/van-ban-phap-luat.json"),
        help="Path to raw legal documents JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to output chunks JSONL file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Maximum chunk character size.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Chunk character overlap.",
    )

    args = parser.parse_args()
    prepare_data(
        raw_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
