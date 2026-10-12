# VN Legal RAG

Hệ thống **Retrieval-Augmented Generation (RAG)** chuyên biệt để tra cứu và giải đáp các văn bản pháp luật Việt Nam. Hệ thống kết hợp giữa tìm kiếm Dense (Vector) và Sparse (BM25) cùng Cross-encoder Reranking để mang lại độ chính xác cao (Hit@4 mức văn bản 95%).

## Dữ liệu (Dataset)
**Văn bản pháp luật thật** (luật, nghị định, thông tư) trích từ dataset [Zalo AI Challenge 2021 — Legal Text Retrieval](https://huggingface.co/datasets/minhnguyent546/zalo-ai-legal-text-retrieval-2021): **249 văn bản, 10.000 điều luật, ~14,3 triệu ký tự**. Kèm **60 câu hỏi tình huống thật có ground-truth** để chấm điểm retrieval khách quan.

## Kiến trúc Hệ thống

- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Sparse Retrieval:** BM25 (thư viện `bm25s`)
- **Vector Database:** Qdrant
- **LLM Generator:** Hỗ trợ OpenRouter (Llama-3, Gemma-2, v.v. thông qua Function Calling) và Groq.
- **Backend API:** FastAPI
- **Frontend UI:** Streamlit
- **AI Agent (New!):** Tích hợp Conversation Memory và Tool-use tự động tra cứu luật dựa trên ngữ cảnh hội thoại.

## Kết quả Đánh giá (Evaluation)

Hệ thống sử dụng bộ 60 câu hỏi pháp luật phức tạp (test set) để đánh giá khả năng trích xuất đúng Điều luật liên quan:

| Phiên bản | Hit@4 (Văn bản) | Hit@4 (Điều luật) | MRR (Điều luật) |
|---|---|---|---|
| Vector thuần | 78.33% | 45.00% | 0.4039 |
| Vector + Reranker | 85.00% | 63.33% | 0.6083 |
| **Hybrid (BM25) + Reranker** | **95.00%** | **85.00%** | **0.7811** |

Số liệu đo ngày 2026-10-11 với chunking theo cấu trúc Điều → Khoản (52.263 chunk).

**Generation:** trên bộ 10 câu hỏi mức phạt giao thông có đáp án chuẩn (`eval/generation_questions.json`), câu trả lời đúng mức phạt 10/10 câu và trích đúng Điều, Khoản 9/10 câu.

*(Xem chi tiết và so sánh với chunking cũ trong file `eval/results.md`)*

## Hướng dẫn Chạy ứng dụng bằng Docker (Khuyên dùng)

Docker Compose chạy 3 service: **Qdrant** (vector DB), **API** (FastAPI, `/ask` và `/chat`) và **UI** (Streamlit). Mặc định API chạy trên **CPU**, không cần GPU.

### 1. Chuẩn bị
- Cài [Docker](https://docs.docker.com/get-docker/) (Docker Compose v2).
- Tạo file `.env` từ mẫu và điền API key:
  ```bash
  cp .env.example .env
  # OPENROUTER_API_KEY: dùng cho chatbot (/chat)
  # GROQ_API_KEY: dùng cho /ask và eval generation
  ```

### 2. Chuẩn bị dữ liệu (chỉ làm một lần)
Index không nằm trong image. API đọc BM25 index từ `data/processed/bm25_index`, còn vector nằm trong `qdrant_storage/`. Cả hai được mount từ máy vào container. Dữ liệu thô đặt trong `data/raw/`.

```bash
pip install torch            # chọn bản CPU hoặc CUDA phù hợp với máy
pip install -r requirements.txt
docker compose up -d qdrant  # bật Qdrant trước để ingest
python scripts/prepare_data.py
python scripts/ingest.py --recreate
python scripts/build_bm25.py
```

### 3. Chạy toàn bộ hệ thống
```bash
docker compose up -d --build
```
- Chatbot: [http://localhost:8501](http://localhost:8501)
- API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

Lần đầu API sẽ tải embedding model về `~/.cache/huggingface` (được mount vào container để dùng lại). UI chỉ khởi động sau khi API báo healthy.

### Chạy bằng GPU (tùy chọn)
Cần GPU NVIDIA và [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Cấu trúc thư mục chính

- `src/ingest/`: Các script làm sạch, cắt đoạn (chunking) dữ liệu pháp luật.
- `src/retrieval/`: Chứa các module tương tác với Qdrant, thuật toán Hybrid Search và Reranker.
- `src/generation/`: Xử lý gọi API tới Groq để sinh câu trả lời.
- `src/api/`: Khai báo FastAPI Backend.
- `src/ui/`: Giao diện Web bằng Streamlit.
- `eval/`: Mã nguồn để chạy bộ đánh giá tự động (Evaluation).
- `scripts/`: Chứa lệnh crawl dữ liệu, lập chỉ mục BM25 và đẩy vector lên Qdrant.

- **Thư mục `src/agent/` (Mới):** Cấu trúc lõi của AI Agent bao gồm:
  - `agent.py`: Vòng lặp suy luận (Agent Loop), gọi LLM và kích hoạt tool.
  - `memory.py`: Quản lý bộ nhớ hội thoại.
  - `tools.py`: Công cụ kết nối Hybrid Search cho Agent.

## Hướng phát triển (Tương lai)
- Thử embedding model mạnh hơn (`BAAI/bge-m3`, `intfloat/multilingual-e5-base`): model hiện tại chỉ nhận 128 token nên chunk dài có thể bị cắt.
- Thêm reranker vào tool tra cứu của Agent (hiện tool chỉ dùng Hybrid Search).
- Mở rộng bộ eval generation (`eval/generation_questions.json`) ra ngoài lĩnh vực giao thông.
