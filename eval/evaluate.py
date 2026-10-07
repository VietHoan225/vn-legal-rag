"""Evaluation script: measure Hit@k and MRR at document and article levels."""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    eval_items: list[dict[str, Any]],
    vector_store: QdrantVectorStore,
    embedder: Embedder,
    collection_name: str,
    k_list: list[int],
) -> dict[str, Any]:
    """Evaluate retrieval performance across test queries."""
    max_k = max(k_list)
    doc_hits: dict[int, int] = {k: 0 for k in k_list}
    article_hits: dict[int, int] = {k: 0 for k in k_list}
    doc_rr_list: list[float] = []
    article_rr_list: list[float] = []

    total_queries = len(eval_items)
    logger.info("Evaluating %d test queries with max_k=%d...", total_queries, max_k)

    for item in eval_items:
        question = item["question"]
        expected_doc_ids = set(item.get("expected_doc_ids", []))
        expected_articles = set(item.get("expected_articles", []))

        # Query embedding and search
        query_vec = embedder.embed_query(question)
        results = vector_store.search(
            collection_name=collection_name,
            query_vector=query_vec,
            limit=max_k,
        )

        retrieved_docs = [r.doc_id for r in results]
        retrieved_articles = [r.article_id for r in results]

        # Calculate Hit@k
        for k in k_list:
            top_docs = set(retrieved_docs[:k])
            top_arts = set(retrieved_articles[:k])

            if expected_doc_ids and not expected_doc_ids.isdisjoint(top_docs):
                doc_hits[k] += 1
            if expected_articles and not expected_articles.isdisjoint(top_arts):
                article_hits[k] += 1

        # Calculate MRR for documents
        doc_rr = 0.0
        for rank, d_id in enumerate(retrieved_docs, start=1):
            if d_id in expected_doc_ids:
                doc_rr = 1.0 / rank
                break
        doc_rr_list.append(doc_rr)

        # Calculate MRR for articles
        art_rr = 0.0
        for rank, a_id in enumerate(retrieved_articles, start=1):
            if a_id in expected_articles:
                art_rr = 1.0 / rank
                break
        article_rr_list.append(art_rr)

    metrics = {
        "total_queries": total_queries,
        "doc_hit": {k: doc_hits[k] / total_queries for k in k_list},
        "article_hit": {k: article_hits[k] / total_queries for k in k_list},
        "doc_mrr": sum(doc_rr_list) / total_queries if total_queries else 0.0,
        "article_mrr": sum(article_rr_list) / total_queries if total_queries else 0.0,
    }
    return metrics


def format_results_markdown(
    metrics: dict[str, Any],
    k_list: list[int],
    config_info: dict[str, Any],
) -> str:
    """Format evaluation metrics as markdown table and report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    table_rows = []
    for k in k_list:
        doc_h = metrics["doc_hit"][k] * 100
        art_h = metrics["article_hit"][k] * 100
        table_rows.append(f"| Hit@{k} | {doc_h:.2f}% | {art_h:.2f}% |")

    doc_mrr = metrics["doc_mrr"]
    art_mrr = metrics["article_mrr"]

    report = f"""
## Kết quả đánh giá Retrieval - {timestamp}

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `{config_info['embedding_model']}`
- **Device:** `{config_info['device']}`
- **Vector Store:** Qdrant (`{config_info['qdrant_url']}`), Collection: `{config_info['collection_name']}`
- **Chunking:** `chunk_size={config_info['chunk_size']}`, `chunk_overlap={config_info['chunk_overlap']}` (prefix: `<tên văn bản> - Điều N`)
- **Số câu hỏi đánh giá:** {metrics['total_queries']} câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
{chr(10).join(table_rows)}
| **MRR** | **{doc_mrr:.4f}** | **{art_mrr:.4f}** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = {metrics['doc_hit'].get(2, 0)*100:.1f}%, Hit@4 = {metrics['doc_hit'].get(4, 0)*100:.1f}%, Hit@6 = {metrics['doc_hit'].get(6, 0)*100:.1f}%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = {art_mrr:.4f}).
"""
    return report.strip()


def run_evaluation(
    questions_path: Path,
    output_results_path: Path,
    collection_name: str,
    k_list: list[int],
) -> dict[str, Any]:
    """Load test dataset, run retrieval evaluation, and append results to markdown."""
    logger.info("Loading test questions from: %s", questions_path)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)

    embedder = Embedder()
    vector_store = QdrantVectorStore()

    start_time = time.time()
    metrics = compute_metrics(
        eval_items=questions,
        vector_store=vector_store,
        embedder=embedder,
        collection_name=collection_name,
        k_list=k_list,
    )
    elapsed = time.time() - start_time
    logger.info("Evaluation completed in %.2f seconds.", elapsed)

    config_info = {
        "embedding_model": settings.embedding_model,
        "device": settings.resolved_device,
        "qdrant_url": settings.qdrant_url,
        "collection_name": collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }

    report_md = format_results_markdown(metrics, k_list, config_info)
    print("\n" + report_md + "\n")

    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    existing_content = ""
    if output_results_path.exists():
        with open(output_results_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

    new_content = (
        "# Nhật ký đánh giá & Thử nghiệm (Evaluation Results)\n\n" + report_md
        if not existing_content
        else existing_content + "\n\n---\n\n" + report_md
    )

    with open(output_results_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info("Results successfully appended to: %s", output_results_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legal RAG retrieval performance.")
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("data/raw/cau-hoi-phap-luat-test.json"),
        help="Path to evaluation questions JSON file.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("eval/results.md"),
        help="Path to results markdown file.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=settings.qdrant_collection_name,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[2, 4, 6],
        help="List of k values for Hit@k.",
    )

    args = parser.parse_args()
    run_evaluation(
        questions_path=args.questions_file,
        output_results_path=args.output_md,
        collection_name=args.collection_name,
        k_list=args.top_k,
    )


if __name__ == "__main__":
    main()
