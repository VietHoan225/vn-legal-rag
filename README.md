# VN Legal RAG

Hệ thống **Retrieval-Augmented Generation (RAG)** chuyên biệt để tra cứu và giải đáp các văn bản pháp luật Việt Nam. Hệ thống kết hợp giữa tìm kiếm Dense (Vector) và Sparse (BM25) cùng Cross-encoder Reranking để mang lại độ chính xác cực cao (Hit Rate 96%).

## Dữ liệu (Dataset)
**Văn bản pháp luật thật** (luật, nghị định, thông tư) trích từ dataset [Zalo AI Challenge 2021 — Legal Text Retrieval](https://huggingface.co/datasets/minhnguyent546/zalo-ai-legal-text-retrieval-2021): **249 văn bản, 10.000 điều luật, ~14,3 triệu ký tự**. Kèm **60 câu hỏi tình huống thật có ground-truth** để chấm điểm retrieval khách quan.

## Kiến trúc Hệ thống

- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Sparse Retrieval:** BM25 (thư viện `bm25s`)
- **Vector Database:** Qdrant
- **LLM Generator:** Groq (tốc độ cao)
- **Backend API:** FastAPI
- **Frontend UI:** Streamlit

## Kết quả Đánh giá (Evaluation)

Hệ thống sử dụng bộ 60 câu hỏi pháp luật phức tạp (test set) để đánh giá khả năng trích xuất đúng Điều luật liên quan:

| Phiên bản | Hit@4 (Văn bản) | Hit@4 (Điều luật) | MRR (Điều luật) |
|---|---|---|---|
| Vector thuần | 78.33% | 48.33% | 0.3825 |
| Vector + Reranker | 85.00% | 61.67% | 0.5708 |
| **Hybrid (BM25) + Reranker** | **96.67%** | **90.00%** | **0.7978** |

*(Xem chi tiết trong file `eval/results.md`)*

## Hướng dẫn Chạy ứng dụng bằng Docker (Khuyên dùng)

Cách nhanh nhất để chạy toàn bộ hệ thống (Database + API + Giao diện) là dùng Docker Compose.

1. Đảm bảo bạn đã cài đặt [Docker](https://docs.docker.com/get-docker/) và Docker Compose.
2. Sao chép file `.env.example` thành `.env` và điền `GROQ_API_KEY`:
   ```bash
   cp .env.example .env
   ```
3. Khởi chạy toàn bộ hệ thống:
   ```bash
   docker-compose up -d --build
   ```
4. Truy cập giao diện Chatbot tại: [http://localhost:8501](http://localhost:8501)
5. Truy cập API Docs (Swagger) tại: [http://localhost:8000/docs](http://localhost:8000/docs)

## Cấu trúc thư mục chính

- `src/ingest/`: Các script làm sạch, cắt đoạn (chunking) dữ liệu pháp luật.
- `src/retrieval/`: Chứa các module tương tác với Qdrant, thuật toán Hybrid Search và Reranker.
- `src/generation/`: Xử lý gọi API tới Groq để sinh câu trả lời.
- `src/api/`: Khai báo FastAPI Backend.
- `src/ui/`: Giao diện Web bằng Streamlit.
- `eval/`: Mã nguồn để chạy bộ đánh giá tự động (Evaluation).
- `scripts/`: Chứa lệnh crawl dữ liệu, lập chỉ mục BM25 và đẩy vector lên Qdrant.

## Hướng phát triển (Tương lai)
- Bổ sung bộ nhớ lịch sử hội thoại (Chat memory).
- Chuyển hóa luồng RAG tĩnh thành AI Agent (sử dụng Groq Function Calling) để LLM tự quyết định khi nào cần tra cứu luật, khi nào chỉ cần giao tiếp thông thường.
