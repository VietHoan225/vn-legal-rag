"""Build BM25 index for Hybrid Search using bm25s."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
import bm25s

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_bm25(chunks_path: Path, output_dir: Path) -> None:
    """Load chunks and build BM25 index."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found at {chunks_path}")

    logger.info("Loading chunks from %s", chunks_path)
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    logger.info("Loaded %d chunks. Tokenizing texts...", len(chunks))
    
    # Extract text from chunks
    texts = [c["text"] for c in chunks]
    
    # Tokenize (bm25s tokenize method handles basic splitting and lowercasing)
    corpus_tokens = bm25s.tokenize(texts)
    
    logger.info("Building BM25 model...")
    start_time = time.time()
    retriever = bm25s.BM25(corpus=chunks) # We store the original chunk dict in corpus
    retriever.index(corpus_tokens)
    elapsed = time.time() - start_time
    logger.info("BM25 index built in %.2f seconds.", elapsed)

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output_dir), corpus=chunks)
    logger.info("BM25 index saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 Index from chunks.")
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
        help="Path to processed chunks JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/bm25_index"),
        help="Directory to save the BM25 index.",
    )

    args = parser.parse_args()
    build_bm25(args.chunks_file, args.output_dir)


if __name__ == "__main__":
    main()
