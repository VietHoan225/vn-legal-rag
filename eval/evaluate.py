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
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _hits(retrieved: list[Any], expected: set[Any]) -> bool:
    """True if any expected id appears in the retrieved ids."""
    return bool(expected) and not expected.isdisjoint(retrieved)


def _reciprocal_rank(retrieved: list[Any], expected: set[Any]) -> float:
    """1/rank of the first expected id, or 0.0 if none is retrieved."""
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in expected:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    eval_items: list[dict[str, Any]],
    vector_store: QdrantVectorStore | None,
    embedder: Embedder | None,
    collection_name: str,
    k_list: list[int],
    reranker: Reranker | None = None,
    hybrid_retriever: HybridRetriever | None = None,
    candidates: int = 30,
    recall_at: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval across test queries.

    Hit@k and MRR are measured on the final ranking (after reranking, if any),
    which reranks `candidates` retrieved hits for every configuration.
    Recall@N is measured before reranking: is the answer among the first N
    retrieved hits? It shows whether the candidate pool is wide enough.
    """
    max_k = max(k_list)
    candidates = max(candidates, max_k)
    recall_at = sorted(recall_at or [])

    def retrieve(question: str, limit: int) -> list[Any]:
        if hybrid_retriever:
            return hybrid_retriever.search(query=question, limit=limit)
        assert vector_store is not None and embedder is not None
        return vector_store.search(
            collection_name=collection_name,
            query_vector=embedder.embed_query(question),
            limit=limit,
        )

    doc_hits: dict[int, int] = {k: 0 for k in k_list}
    article_hits: dict[int, int] = {k: 0 for k in k_list}
    doc_recall: dict[int, int] = {n: 0 for n in recall_at}
    article_recall: dict[int, int] = {n: 0 for n in recall_at}
    doc_rr_list: list[float] = []
    article_rr_list: list[float] = []
    rerank_ms: list[float] = []

    total_queries = len(eval_items)
    logger.info(
        "Evaluating %d test queries with max_k=%d, candidates=%d, recall_at=%s...",
        total_queries, max_k, candidates, recall_at,
    )

    for item in eval_items:
        question = item["question"]
        expected_doc_ids = set(item.get("expected_doc_ids", []))
        expected_articles = set(item.get("expected_articles", []))

        # Recall of the candidate pool, before reranking
        for n in recall_at:
            pool = retrieve(question, n)
            doc_recall[n] += _hits([r.doc_id for r in pool], expected_doc_ids)
            article_recall[n] += _hits([r.article_id for r in pool], expected_articles)

        # Final ranking
        if reranker:
            pool = retrieve(question, candidates)
            t0 = time.perf_counter()
            results = reranker.rerank(question, pool, top_k=max_k)
            rerank_ms.append((time.perf_counter() - t0) * 1000)
        else:
            results = retrieve(question, max_k)

        retrieved_docs = [r.doc_id for r in results]
        retrieved_articles = [r.article_id for r in results]
        for k in k_list:
            doc_hits[k] += _hits(retrieved_docs[:k], expected_doc_ids)
            article_hits[k] += _hits(retrieved_articles[:k], expected_articles)
        doc_rr_list.append(_reciprocal_rank(retrieved_docs, expected_doc_ids))
        article_rr_list.append(_reciprocal_rank(retrieved_articles, expected_articles))

    def rate(counts: dict[int, int]) -> dict[int, float]:
        return {k: v / total_queries if total_queries else 0.0 for k, v in counts.items()}

    rerank_ms.sort()
    metrics = {
        "total_queries": total_queries,
        "candidates": candidates if reranker else None,
        "doc_hit": rate(doc_hits),
        "article_hit": rate(article_hits),
        "doc_recall": rate(doc_recall),
        "article_recall": rate(article_recall),
        "doc_mrr": sum(doc_rr_list) / total_queries if total_queries else 0.0,
        "article_mrr": sum(article_rr_list) / total_queries if total_queries else 0.0,
        "rerank_ms_median": rerank_ms[len(rerank_ms) // 2] if rerank_ms else None,
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

    recall_rows = [
        f"| Recall@{n} | {metrics['doc_recall'][n] * 100:.2f}% | {metrics['article_recall'][n] * 100:.2f}% |"
        for n in metrics["doc_recall"]
    ]
    recall_section = (
        "\n\n**Recall trước rerank** (đáp án có nằm trong N kết quả đầu của retriever không):\n\n"
        "| Chỉ số | Mức Văn bản | Mức Điều |\n|---|---|---|\n" + "\n".join(recall_rows)
        if recall_rows else ""
    )
    candidates_line = (
        f"\n- **Ứng viên đưa vào rerank:** {metrics['candidates']} "
        f"(thời gian rerank trung vị: {metrics['rerank_ms_median']:.0f} ms/câu)"
        if metrics.get("candidates") else ""
    )

    report = f"""
## Kết quả đánh giá Retrieval - {timestamp}

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `{config_info['embedding_model']}`
- **Reranker Model:** `{config_info.get('reranker_model', 'None')}`
- **Hybrid Search (BM25):** `{'Enabled' if config_info.get('use_hybrid') else 'Disabled'}`{candidates_line}
- **Device:** `{config_info['device']}`
- **Vector Store:** Qdrant (`{config_info['qdrant_url']}`), Collection: `{config_info['collection_name']}`
- **Chunking:** `chunk_size={config_info['chunk_size']}`, `chunk_overlap={config_info['chunk_overlap']}` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** {metrics['total_queries']} câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
{chr(10).join(table_rows)}
| **MRR** | **{doc_mrr:.4f}** | **{art_mrr:.4f}** | N/A |{recall_section}

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
    use_reranker: bool = False,
    use_hybrid: bool = False,
    candidates: int = settings.rerank_candidates,
    recall_at: list[int] | None = None,
) -> dict[str, Any]:
    """Load test dataset, run retrieval evaluation, and append results to markdown."""
    logger.info("Loading test questions from: %s", questions_path)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)

    embedder = Embedder()
    vector_store = QdrantVectorStore()
    reranker = Reranker() if use_reranker else None
    hybrid_retriever = HybridRetriever(vector_store, embedder) if use_hybrid else None

    start_time = time.time()
    metrics = compute_metrics(
        eval_items=questions,
        vector_store=vector_store,
        embedder=embedder,
        collection_name=collection_name,
        k_list=k_list,
        reranker=reranker,
        hybrid_retriever=hybrid_retriever,
        candidates=candidates,
        recall_at=recall_at,
    )
    elapsed = time.time() - start_time
    logger.info("Evaluation completed in %.2f seconds.", elapsed)

    config_info = {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model if use_reranker else "None",
        "use_hybrid": use_hybrid,
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

    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable Cross-Encoder reranking.",
    )

    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Enable Hybrid Search (Dense + Sparse BM25).",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=settings.rerank_candidates,
        help="Retrieved hits handed to the reranker (same for every configuration).",
    )
    parser.add_argument(
        "--recall-at",
        nargs="*",
        type=int,
        default=[10, 20, 30, 50],
        help="Pool sizes N for Recall@N before reranking (pass none to skip).",
    )

    args = parser.parse_args()
    run_evaluation(
        questions_path=args.questions_file,
        output_results_path=args.output_md,
        collection_name=args.collection_name,
        k_list=args.top_k,
        use_reranker=args.use_reranker,
        use_hybrid=args.use_hybrid,
        candidates=args.candidates,
        recall_at=args.recall_at,
    )


if __name__ == "__main__":
    main()
