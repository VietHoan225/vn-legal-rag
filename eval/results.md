# Nhật ký đánh giá & Thử nghiệm (Evaluation Results)

## Kết quả đánh giá Retrieval

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (prefix: `<tên văn bản> - Điều N`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 71.67% | 36.67% |
| Hit@4 | 78.33% | 48.33% |
| Hit@6 | 85.00% | 50.00% |
| **MRR** | **0.6919** | **0.3825** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 71.7%, Hit@4 = 78.3%, Hit@6 = 85.0%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.3825).

---

## Kết quả đánh giá Retrieval

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (prefix: `<tên văn bản> - Điều N`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 81.67% | 56.67% |
| Hit@4 | 85.00% | 61.67% |
| Hit@6 | 85.00% | 61.67% |
| **MRR** | **0.8097** | **0.5708** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 81.7%, Hit@4 = 85.0%, Hit@6 = 85.0%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.5708).

---

## Kết quả đánh giá Retrieval

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Hybrid Search (BM25):** `Enabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (prefix: `<tên văn bản> - Điều N`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 95.00% | 83.33% |
| Hit@4 | 96.67% | 90.00% |
| Hit@6 | 96.67% | 91.67% |
| **MRR** | **0.9042** | **0.7978** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 95.0%, Hit@4 = 96.7%, Hit@6 = 96.7%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.7978).

## III. Đánh giá AI Agent & Vấn đề Chunking (Case Study)

### 1. Sự cố "Ảo giác" do Chunking (Hallucination)
- **Tình huống:** Khi hỏi Agent về mức phạt không đội mũ bảo hiểm cho xe máy và xe đạp máy.
- **Lỗi quan sát được:** Agent đưa ra mức phạt sai lệch (ghép mức phạt 100-200k của Khoản 1 vào lỗi không đội mũ bảo hiểm của Khoản 2 Điều 6 Nghị định 100/2019/NĐ-CP). Hệ quả là Agent kết luận sai logic (đi xe máy phạt nhẹ hơn xe đạp máy).
- **Nguyên nhân cốt lõi (Root Cause):** 
  - Mô hình lấy kết quả từ Vector DB nhưng nội dung bị "cắt ngang" do chiến lược Chunking hiện tại (`chunk_size=600`, overlap cố định). Cấu trúc của một "Điều, Khoản" luật bị phá vỡ, làm LLM đọc và suy diễn sai lệch ("râu ông nọ cắm cằm bà kia").
  - Mô hình LLM dễ bị ảnh hưởng khi bối cảnh (context) bị cắt ghép không nguyên vẹn.

### 2. Kế hoạch tối ưu (Next Steps)
- **Cải tiến Chunking:** Thay vì cắt mù quáng theo token/ký tự (`chunk_size=600`), cần nâng cấp luồng làm sạch ở `src/ingest/` sang **Semantic Chunking theo cấu trúc Luật** (cắt dựa trên Regex các từ khóa `Điều X`, `Khoản Y`, `Điểm Z`). Điều này đảm bảo mỗi chunk chứa trọn vẹn ngữ cảnh của một vi phạm và mức phạt đi kèm, triệt tiêu lỗi ảo giác do lắp ghép.
- **Bổ sung Eval:** Thêm bộ câu hỏi test riêng cho Agent để đo lường độ chính xác của câu trả lời cuối (Generation accuracy) thay vì chỉ đo độ chính xác của tìm kiếm (Retrieval accuracy).