"""Test script to ask a legal question from terminal."""

import argparse
import sys
from pathlib import Path
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import QdrantVectorStore
from src.generation.generator import GroqGenerator

logging.basicConfig(level=logging.WARNING)  # Hide verbose info logs for clean output

def main():
    parser = argparse.ArgumentParser(description="Ask a legal question.")
    parser.add_argument("question", type=str, help="Câu hỏi pháp lý cần tư vấn")
    args = parser.parse_args()

    print(f"\n🔍 Đang tìm kiếm câu trả lời cho: '{args.question}'...")
    
    # 1. Khởi tạo các công cụ
    embedder = Embedder()
    vector_store = QdrantVectorStore()
    generator = GroqGenerator()

    # 2. Retrieval (Tìm kiếm)
    query_vec = embedder.embed_query(args.question)
    search_results = vector_store.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vec,
        limit=3  # Lấy 3 đoạn luật giống nhất
    )

    if not search_results:
        print("❌ Không tìm thấy đoạn luật nào trong Qdrant. Bạn đã chạy scripts/ingest.py chưa?")
        return

    print("📚 Đã tìm thấy các tài liệu liên quan. Đang nhờ AI phân tích...")

    # 3. Generation (Sinh câu trả lời)
    result = generator.generate_answer(args.question, search_results)

    # In kết quả
    print("\n" + "="*50)
    print("⚖️ KẾT QUẢ TƯ VẤN:")
    print("="*50)
    print(result.answer)
    print("="*50)


if __name__ == "__main__":
    main()
