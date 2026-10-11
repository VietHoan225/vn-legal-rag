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

---

## Kết quả đánh giá Retrieval

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `None`
- **Hybrid Search (BM25):** `Disabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 73.33% | 40.00% |
| Hit@4 | 78.33% | 45.00% |
| Hit@6 | 83.33% | 51.67% |
| **MRR** | **0.7325** | **0.4039** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 73.3%, Hit@4 = 78.3%, Hit@6 = 83.3%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.4039).

---

## Kết quả đánh giá Retrieval 

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Hybrid Search (BM25):** `Disabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 81.67% | 60.00% |
| Hit@4 | 85.00% | 63.33% |
| Hit@6 | 86.67% | 63.33% |
| **MRR** | **0.8117** | **0.6083** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 81.7%, Hit@4 = 85.0%, Hit@6 = 86.7%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.6083).

---

## Kết quả đánh giá Retrieval

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Hybrid Search (BM25):** `Enabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 95.00% | 81.67% |
| Hit@4 | 95.00% | 85.00% |
| Hit@6 | 96.67% | 88.33% |
| **MRR** | **0.9033** | **0.7811** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 95.0%, Hit@4 = 95.0%, Hit@6 = 96.7%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.7811).
---

## So sánh chunking cũ và chunking theo cấu trúc Điều → Khoản 

Cấu hình chung: MiniLM-L12-v2, `chunk_size=600`, `chunk_overlap=80`, device `cuda`, 60 câu Zalo Legal test.
Collection được tạo lại (34.211 → 52.263 chunk), BM25 index build lại cùng bộ chunk.

| Cấu hình | Doc Hit@4 (cũ → mới) | Doc MRR (cũ → mới) | Article Hit@4 (cũ → mới) | Article MRR (cũ → mới) |
|---|---|---|---|---|
| Dense | 78.33% → 78.33% | 0.6919 → 0.7325 | 48.33% → 45.00% | 0.3825 → 0.4039 |
| Dense + Rerank | 85.00% → 85.00% | 0.8097 → 0.8117 | 61.67% → 63.33% | 0.5708 → 0.6083 |
| Hybrid + Rerank | 96.67% → 95.00% | 0.9042 → 0.9033 | 90.00% → 85.00% | 0.7978 → 0.7811 |

Nhận xét:
- Dense và Dense + Rerank: MRR tăng nhẹ ở cả hai mức, chunk theo Khoản giúp xếp hạng đúng tài liệu lên cao hơn.
- Hybrid + Rerank: giảm nhẹ ở mức Điều (Hit@4 −5 điểm, tương đương 3/60 câu). Với 60 câu, chênh lệch này nằm trong mức nhiễu, chưa đủ để kết luận.
- Chunking mới nhằm chủ yếu sửa lỗi ghép sai Khoản khi sinh câu trả lời (mục III), nên cần thêm eval cho generation để thấy hết tác dụng.

---

## Kết quả đánh giá Generation - 2026-10-11 23:05

### 1. Cấu hình thử nghiệm
- **Generator:** `GroqGenerator` (`qwen/qwen3.8-27b`), chưa giới hạn max_tokens
- **Retrieval:** Hybrid (Dense + BM25) + Rerank (`BAAI/bge-reranker-v2-m3`), top_k=5 đoạn làm context
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, device `cuda`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (Điều → Khoản → nhóm Điểm)
- **Bộ câu hỏi:** `eval/generation_questions.json`, 10 câu mức phạt giao thông (Nghị định 100/2019/NĐ-CP), đáp án soạn từ `van-ban-phap-luat.json`

### 2. Bảng chỉ số đo lường

| Chỉ số | Gold (mức phạt) (10/10 câu) |
|---|---|
| Trích đúng văn bản | 100.0% |
| Trích đúng Điều | 90.0% |
| Trích đúng Khoản | 90.0% |
| Đúng mức phạt / dữ kiện | 100.0% |
| Tỉ lệ từ chối | 0.0% |

### 3. Nhận xét
- Không còn lỗi ghép sai Khoản như case study mũ bảo hiểm (mục III): cả xe máy (Điều 6 Khoản 2) và xe đạp điện (Điều 8 Khoản 3) đều trả lời đúng 200.000 – 300.000 đồng.
- Câu sai duy nhất `gen_03` ("Xe máy vượt đèn đỏ..."): retrieval xếp Điều 47 (vượt đèn đỏ ở đường ngang đường sắt) lên đầu thay vì Điều 6 Khoản 4. Mức phạt trùng nhau nên vẫn tính đúng dữ kiện. Đây là lỗi retrieval.
- Bộ Zalo test (60 câu) chưa có kết quả hợp lệ: Groq giới hạn 1.000 output token/phút cho model này, 42/60 câu bị lỗi 429. Script hiện mặc định chỉ chạy 10 câu mỗi bộ và tự retry khi gặp 429.
