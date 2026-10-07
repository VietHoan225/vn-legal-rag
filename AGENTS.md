# AGENTS.md: quy tắc cho agent làm việc trong vn-legal-rag

## Dự án
Hệ thống RAG hỏi đáp văn bản pháp luật tiếng Việt (dữ liệu Zalo Legal).
Mục tiêu: học tập và portfolio. Ưu tiên code đơn giản, đo được, giải thích được.

## Cấu trúc
- src/ingest: làm sạch, chunking
- src/retrieval: embedder, vector store, (sau này) hybrid, rerank
- src/generation: prompt, gọi LLM
- src/api: FastAPI (giai đoạn sau)
- eval/: bộ câu hỏi, script đo, results.md
- scripts/: prepare_data.py, ingest.py
- tests/: pytest, fixtures nhỏ trong tests/fixtures/

## Quy tắc code
- Python 3.11, type hints, docstring ngắn.
- Cấu hình đọc từ .env qua pydantic-settings (src/config.py).
- Không hard-code secret hay đường dẫn tuyệt đối.
- Tự chọn device: "cuda" nếu có, không thì "cpu".
- Vector store bọc sau interface VectorStore, triển khai bằng Qdrant.

## Quy tắc an toàn
- KHÔNG đọc, in, sửa hoặc commit file .env.
- KHÔNG cài lại hoặc nâng cấp torch; torch được cài thủ công theo máy
  và không nằm trong requirements.txt.
- KHÔNG commit data/, qdrant_storage/, notebooks/legacy/.
- notebooks/legacy/ chỉ để tham khảo logic, không sao chép nội dung đề bài.

## Test và đo lường
- Mỗi hàm logic chính có pytest. Test không gọi Gemini hay tải model thật.
- Mọi kết quả thí nghiệm ghi vào eval/results.md kèm ngày và cấu hình.
- Chạy pytest sau mỗi thay đổi lớn và báo kết quả.

## Cách làm việc
- Lập kế hoạch trước, liệt kê file sẽ tạo/sửa, chờ duyệt rồi mới làm.
- Làm từng bước nhỏ, không làm vượt phạm vi được giao.
- Chỗ nào chưa rõ về dữ liệu thì hỏi, đừng đoán.
