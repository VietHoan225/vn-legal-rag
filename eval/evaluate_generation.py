"""Evaluation script: score generated answers on citations, refusals and key facts."""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

from groq import RateLimitError

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.generation.generator import GroqGenerator
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import QdrantVectorStore, SearchResult
from eval.generation_metrics import aggregate, score_answer, strip_thinking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRIC_LABELS = {
    "doc_cited": "Trích đúng văn bản",
    "article_cited": "Trích đúng Điều",
    "clause_cited": "Trích đúng Khoản",
    "facts_correct": "Đúng mức phạt / dữ kiện",
    "refusal": "Tỉ lệ từ chối",
}


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of evaluation questions."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retry_wait_seconds(exc: RateLimitError, attempt: int) -> float:
    """Seconds to wait after a 429: use the retry-after header, else exponential backoff."""
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(header) + 1.0
    except (TypeError, ValueError):
        return min(10.0 * 2**attempt, 120.0)


def generate_with_retry(
    generator: GroqGenerator,
    question: str,
    results: list[SearchResult],
    max_attempts: int = 5,
) -> str:
    """Call the generator, waiting and retrying when Groq returns a rate-limit error."""
    for attempt in range(max_attempts):
        try:
            return generator.generate_answer(question, results).answer
        except RateLimitError as exc:
            if attempt == max_attempts - 1:
                raise
            wait = retry_wait_seconds(exc, attempt)
            logger.warning("Rate limited (attempt %d/%d), waiting %.0fs: %s", attempt + 1, max_attempts, wait, exc)
            time.sleep(wait)
    raise AssertionError("unreachable")


def run_set(
    name: str,
    questions: list[dict[str, Any]],
    retriever: HybridRetriever,
    reranker: Reranker,
    generator: GroqGenerator,
    top_k: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    """Retrieve, generate and score every question in one evaluation set."""
    records = []
    for i, item in enumerate(questions, start=1):
        question = item["question"]
        results = retriever.search(query=question, limit=max(settings.rerank_candidates, top_k))
        results = reranker.rerank(question, results, top_k=top_k)

        error = None
        try:
            answer = generate_with_retry(generator, question, results)
        except Exception as exc:  # keep going so one API failure does not kill the run
            logger.warning("[%s %d/%d] Generation failed: %s", name, i, len(questions), exc)
            answer, error = "", str(exc)

        scores = score_answer(answer, item) if error is None else {}
        records.append({
            "set": name,
            "query_id": item.get("query_id"),
            "question": question,
            "expected_articles": item.get("expected_articles", []),
            "retrieved_articles": [r.article_id for r in results],
            "answer": strip_thinking(answer),
            "error": error,
            "scores": scores,
        })
        logger.info("[%s %d/%d] %s", name, i, len(questions), scores or "ERROR")
        time.sleep(sleep_seconds)
    return records


def format_results_markdown(
    summaries: dict[str, dict[str, Any]],
    config_info: dict[str, Any],
) -> str:
    """Format generation metrics per evaluation set as a markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_names = list(summaries)
    header = "| Chỉ số | " + " | ".join(
        f"{n} ({summaries[n]['scored']}/{summaries[n]['total']} câu)" for n in set_names
    ) + " |"
    rows = [header, "|---|" + "---|" * len(set_names)]
    for key, label in METRIC_LABELS.items():
        cells = []
        for n in set_names:
            value = summaries[n]["metrics"].get(key)
            cells.append(f"{value * 100:.1f}%" if value is not None else "-")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    return f"""
## Kết quả đánh giá Generation - {timestamp}

### 1. Cấu hình thử nghiệm
- **Generator:** `GroqGenerator` (`{config_info['llm_model']}`), max_tokens={config_info['max_tokens']}
- **Retrieval:** Hybrid (Dense + BM25) + Rerank (`{config_info['reranker_model']}`) trên {config_info['candidates']} ứng viên, top_k={config_info['top_k']} đoạn làm context
- **Embedding Model:** `{config_info['embedding_model']}`, device `{config_info['device']}`
- **Chunking:** `chunk_size={config_info['chunk_size']}`, `chunk_overlap={config_info['chunk_overlap']}` (Điều → Khoản → nhóm Điểm)
- **Câu trả lời chi tiết:** `{config_info['output_file']}`

### 2. Bảng chỉ số đo lường

{chr(10).join(rows)}

Ghi chú: chấm bằng luật (regex), không dùng LLM. Câu từ chối không được tính điểm trích dẫn.
"Trích đúng Điều" cần có cả số hiệu văn bản và "Điều N";
"Trích đúng Khoản" cần thêm "Khoản K"; "Đúng mức phạt" cần có đủ mọi con số trong `expected_facts`.
Câu lỗi API không được tính vào tỉ lệ.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate legal RAG answer generation.")
    parser.add_argument(
        "--retrieval-questions",
        type=Path,
        default=Path("data/raw/cau-hoi-phap-luat-test.json"),
        help="Zalo Legal test questions (citation metrics only).",
    )
    parser.add_argument(
        "--gold-questions",
        type=Path,
        default=Path("eval/generation_questions.json"),
        help="Hand-checked questions with expected clause and facts.",
    )
    parser.add_argument("--output-md", type=Path, default=Path("eval/results.md"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/outputs"))
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks passed to the LLM.")
    parser.add_argument("--limit", type=int, default=10, help="Use the first N questions per set (0 = all).")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds between LLM calls (rate limits).")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Cap on answer length per LLM call.")
    args = parser.parse_args()

    sets = {
        "Gold (mức phạt)": load_questions(args.gold_questions),
        "Zalo test": load_questions(args.retrieval_questions),
    }
    if args.limit:
        sets = {name: qs[: args.limit] for name, qs in sets.items()}

    embedder = Embedder()
    vector_store = QdrantVectorStore()
    retriever = HybridRetriever(vector_store, embedder)
    reranker = Reranker()
    generator = GroqGenerator(max_tokens=args.max_tokens)

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, questions in sets.items():
        records = run_set(name, questions, retriever, reranker, generator, args.top_k, args.sleep)
        scored = [r["scores"] for r in records if r["error"] is None]
        summaries[name] = {"total": len(records), "scored": len(scored), "metrics": aggregate(scored)}
        all_records.extend(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f"generation_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d answers to %s", len(all_records), output_file)

    config_info = {
        "llm_model": generator.model_name,
        "reranker_model": settings.reranker_model,
        "embedding_model": settings.embedding_model,
        "device": settings.resolved_device,
        "top_k": args.top_k,
        "candidates": max(settings.rerank_candidates, args.top_k),
        "max_tokens": args.max_tokens,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "output_file": output_file.as_posix(),
    }
    report_md = format_results_markdown(summaries, config_info)
    print("\n" + report_md + "\n")

    with open(args.output_md, "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n" + report_md + "\n")
    logger.info("Results appended to: %s", args.output_md)


if __name__ == "__main__":
    main()
