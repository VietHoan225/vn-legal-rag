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

## Luồng hoạt động

Hệ thống gồm 3 chương trình chạy riêng, mỗi chương trình nghe ở một **port** (số cửa) khác nhau:

| Thành phần | Port | Vai trò | Code |
|---|---|---|---|
| **Streamlit** | 8501 | Giao diện chat trên trình duyệt | `src/ui/app.py` |
| **FastAPI** chạy bằng **uvicorn** | 8000 | Nhận câu hỏi, giữ mỗi cuộc trò chuyện một agent riêng, trả câu trả lời | `src/api/main.py` |
| **Qdrant** | 6333 | Cơ sở dữ liệu vector chứa các đoạn luật | (Docker image `qdrant/qdrant`) |

FastAPI chỉ là bộ khung gồm các hàm xử lý. **uvicorn** là chương trình lắng nghe mạng, nhận yêu cầu rồi gọi đúng hàm FastAPI. Tách UI và API để mô hình embedding và reranker (nặng) chỉ phải nạp một lần ở API.

### Một tin nhắn đi qua những đâu

```
Trình duyệt ─ gõ câu hỏi ─▶ Streamlit (:8501)                       src/ui/app.py
                              │  HTTP POST /chat/stream
                              ▼
                          FastAPI (:8000)                          src/api/main.py
                              │  mỗi session_id có một LegalAgent riêng (có memory)
                              ▼
                          LegalAgent                               src/agent/agent.py
                              │ 1. hỏi LLM: có cần tra luật không?  ───────────▶ OpenRouter
                              │ 2. LLM yêu cầu gọi tool search_legal_docs
                              ▼
                          HybridRetriever                          src/retrieval/hybrid_search.py
                              │  embed câu hỏi → Qdrant (:6333) + BM25 → gộp bằng RRF
                              ▼
                              │ 3. đưa đoạn luật cho LLM, LLM viết câu trả lời ─▶ OpenRouter
                              ▼
        câu trả lời chảy ngược từng đoạn: Agent → FastAPI → Streamlit → màn hình
        (cuối cùng là danh sách nguồn trích dẫn, ghi cả Khoản)
```

Câu chào hỏi hoặc câu ngoài lề không qua bước 2 và 3: LLM trả lời ngay trong một lần gọi. Câu hỏi tiếp nối ("Thế còn xe đạp điện thì sao?") thường cũng không cần tra lại, vì LLM đã có kết quả tra cứu của lượt trước trong memory.

### Ví dụ thật: log của một câu hỏi pháp luật

Câu hỏi: *"Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu?"* (embedding chạy trên GPU)

```
10:59:07  Calling LLM (Step 1)...                                      ← bước 1: hỏi LLM
10:59:09  LLM step 1: no text, total 2.1s, 1 tool call(s)              ← LLM không viết gì, chỉ yêu cầu gọi tool
10:59:09  Executing tool: search_legal_docs with query: 'mức phạt đi xe máy không đội mũ bảo hiểm'
10:59:09  Hybrid search: embed 204ms, qdrant 4ms, bm25 1ms             ← bước 2: tra luật, rất nhanh
10:59:09  Tool search_legal_docs: 6 result(s) in 0.21s
10:59:09  Calling LLM (Step 2)...                                      ← bước 3: đưa 6 đoạn luật cho LLM
10:59:14  Chat stream: first token after 7.0s                          ← chữ đầu tiên xuất hiện trên màn hình
10:59:24  Chat timing: first token 7.0s, LLM 16.2s, tools 0.2s, total 16.4s
```

Thời gian nằm gần như hoàn toàn ở hai lần gọi LLM, còn tra luật chỉ mất khoảng 0,2 giây.

### API

| Endpoint | Dùng để |
|---|---|
| `POST /chat/stream` | Chat nhiều lượt, trả lời chảy ra từng đoạn (UI dùng endpoint này) |
| `POST /chat` | Như trên nhưng trả một lần khi đã xong |
| `DELETE /chat/{session_id}` | Xoá bộ nhớ của một cuộc trò chuyện |
| `POST /ask` | Hỏi đáp một lượt, không có memory (dùng cho eval) |
| `GET /health` | Kiểm tra API đang chạy |

Khi API đang chạy, mở [http://localhost:8000/docs](http://localhost:8000/docs) để gọi thử các endpoint này ngay trên trình duyệt.

### Chạy bằng tay (không dùng Docker)

Cần 2 terminal, Qdrant phải đang chạy (`docker compose up -d qdrant`):

```bash
uvicorn src.api.main:app --port 8000     # terminal 1: API
streamlit run src/ui/app.py              # terminal 2: giao diện, mở http://localhost:8501
```

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
  - `scripts/profile_chat.py`: đo độ trễ chatbot qua API (`/chat` và `/chat/stream`).

- **Thư mục `src/agent/` (Mới):** Cấu trúc lõi của AI Agent bao gồm:
  - `agent.py`: Vòng lặp suy luận (Agent Loop), gọi LLM và kích hoạt tool.
  - `memory.py`: Quản lý bộ nhớ hội thoại.
  - `tools.py`: Công cụ kết nối Hybrid Search cho Agent.

## Hướng phát triển (Tương lai)
- Thử embedding model mạnh hơn (`BAAI/bge-m3`, `intfloat/multilingual-e5-base`): model hiện tại chỉ nhận 128 token nên chunk dài có thể bị cắt.
- Thêm reranker vào tool tra cứu của Agent (hiện tool chỉ dùng Hybrid Search).
- Mở rộng bộ eval generation (`eval/generation_questions.json`) ra ngoài lĩnh vực giao thông.
