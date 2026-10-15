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

## Kết quả đánh giá Generation

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


---

## Đo độ trễ chatbot: streaming - 2026-10-15

### 1. Cấu hình
- **Agent:** `LegalAgent` (giữ nguyên vòng lặp tool), LLM `nvidia/nemotron-3-ultra-550b-a55b:free` qua OpenRouter
- **Retrieval:** Hybrid (Dense + BM25), embedding `paraphrase-multilingual-MiniLM-L12-v2` trên `cuda` (RTX 2060), Qdrant chạy Docker
- **Cách đo:** `scripts/profile_chat.py`, 4 câu trong cùng một session, **mỗi chế độ chạy 1 lần**
  - `/chat`: trả về khi đã có đủ câu trả lời, nên người dùng chờ hết thời gian đó
  - `/chat/stream`: trả lời từng đoạn, "token đầu tiên" là lúc người dùng bắt đầu đọc được

### 2. Kết quả (giây)

| Loại câu | Câu hỏi | `/chat` (chờ hết) | `/chat/stream`: token đầu tiên | `/chat/stream`: tổng |
|---|---|---|---|---|
| chào hỏi | xin chào | 4.7 | 6.8 | 7.1 |
| ngoài lề | 1 + 1 = ? | 2.8 | 2.9 | 3.8 |
| pháp luật | Đi xe máy không đội mũ bảo hiểm bị phạt bao nhiêu? | 26.0 | **7.3** | 18.0 |
| tiếp nối | Thế còn xe đạp điện thì sao? | 25.3 | **5.3** | 18.9 |

### 3. Thời gian nằm ở đâu (log phía server, câu pháp luật)
- Bước LLM quyết định gọi tool: 4.1 - 10.7s
- **Tra luật (hybrid search): 0.03 - 0.23s** (embed 28 - 204ms, Qdrant 4 - 24ms, BM25 1 - 2ms), gần như không đáng kể
- Bước LLM sinh câu trả lời: 11 - 15s, trong đó token đầu tiên mất khoảng 3s, phần còn lại là sinh dần ra

### 4. Nhận xét
- Với câu pháp luật, người dùng bắt đầu đọc được sau khoảng 5 - 7s thay vì chờ 25 - 26s.
- Streaming không làm tổng thời gian ngắn đi. Chênh lệch tổng giữa hai lần chạy (26s so với 18s) chủ yếu do model free dao động từ lần gọi này sang lần gọi khác, vì mỗi chế độ chỉ chạy 1 lần.
- Nút thắt nằm ở LLM (hai lần gọi, mỗi lần chậm), không phải retrieval. Muốn giảm tổng thời gian thật sự thì phải bớt lần gọi LLM hoặc đổi model.


---

## Kết quả đánh giá Retrieval - 2026-10-15 13:07:46

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `BAAI/bge-m3`
- **Reranker Model:** `None`
- **Hybrid Search (BM25):** `Disabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks_bgem3`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 90.00% | 71.67% |
| Hit@4 | 95.00% | 83.33% |
| Hit@6 | 98.33% | 88.33% |
| **MRR** | **0.8797** | **0.7275** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 90.0%, Hit@4 = 95.0%, Hit@6 = 98.3%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.7275).

---

## Kết quả đánh giá Retrieval - 2026-10-15 13:08:40

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `BAAI/bge-m3`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Hybrid Search (BM25):** `Disabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks_bgem3`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 93.33% | 85.00% |
| Hit@4 | 96.67% | 90.00% |
| Hit@6 | 98.33% | 91.67% |
| **MRR** | **0.8964** | **0.8172** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 93.3%, Hit@4 = 96.7%, Hit@6 = 98.3%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.8172).

---

## Kết quả đánh giá Retrieval - 2026-10-15 13:10:10

### 1. Cấu hình thử nghiệm
- **Embedding Model:** `BAAI/bge-m3`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **Hybrid Search (BM25):** `Enabled`
- **Device:** `cuda`
- **Vector Store:** Qdrant (`http://localhost:6333`), Collection: `legal_chunks_bgem3`
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (chia theo Điều → Khoản → nhóm Điểm, prefix: `<tên văn bản> - Điều N[ - Khoản K]`)
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set)

### 2. Bảng chỉ số đo lường

| Chỉ số | Mức Văn bản (Document) | Mức Điều (Article) | Baseline cũ (Prototype) |
|---|---|---|---|
| Hit@2 | 95.00% | 85.00% |
| Hit@4 | 98.33% | 90.00% |
| Hit@6 | 100.00% | 91.67% |
| **MRR** | **0.9117** | **0.8158** | N/A |

### 3. Nhận xét và so sánh
- **So với baseline cũ:**
  - Ở mức Văn bản: Hit@2 = 95.0%, Hit@4 = 98.3%, Hit@6 = 100.0%.
  - Baseline cũ trong prototype notebook đạt Hit@4 ≈ 80.0% mức văn bản.
- **Mức Điều (Article-level):** Đã bổ sung đo lường chính xác tới từng Điều cụ thể (MRR = 0.8158).

---

## So sánh embedding model: MiniLM và bge-m3 - 2026-10-15

### 1. Cấu hình
- **Cũ:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều, đọc tối đa 128 token), collection `legal_chunks`. Số liệu lấy từ lần đo 2026-10-11.
- **Mới:** `BAAI/bge-m3` (1024 chiều, đọc tối đa 8192 token), collection `legal_chunks_bgem3`.
- Giống nhau: cùng 52.263 chunk (Điều → Khoản → nhóm Điểm), cùng BM25 index, cùng reranker `BAAI/bge-reranker-v2-m3`, cùng 60 câu Zalo Legal test.
- Trước khi đổi: 33,4% chunk vượt 128 token của model cũ, tổng cộng 5,3% số token bị cắt khi embed.

### 2. Chất lượng (Hit@4 và MRR)

| Cấu hình | Văn bản Hit@4 (cũ → mới) | Điều Hit@4 (cũ → mới) | Điều MRR (cũ → mới) |
|---|---|---|---|
| Dense | 78.33% → **95.00%** | 45.00% → **83.33%** | 0.4039 → **0.7275** |
| Dense + Rerank | 85.00% → **96.67%** | 63.33% → **90.00%** | 0.6083 → **0.8172** |
| Hybrid + Rerank | 95.00% → 98.33% | 85.00% → 90.00% | 0.7811 → 0.8158 |

### 3. Chi phí

| | MiniLM | bge-m3 |
|---|---|---|
| Embed toàn bộ 52.263 chunk (GPU RTX 2060) | khoảng 1 phút | 38 phút (batch 32) |
| Embed 1 câu hỏi trên CPU (trung vị, 8 luồng) | 21 ms | 163 ms |
| RAM model chiếm khi chạy CPU | ~0,7 GB | ~1,6 GB |
| File cache embedding | 80 MB | 214 MB |

### 4. Nhận xét
- Dense thuần tăng rất mạnh (mức Điều Hit@4 +38 điểm), vượt xa mức nhiễu của bộ 60 câu. Riêng Dense của bge-m3 (83%) đã gần bằng cả pipeline cũ Hybrid + Rerank (85%).
- Cấu hình đang chạy thật (Hybrid + Rerank) tăng ít hơn: mức Điều Hit@4 +5 điểm (3/60 câu), MRR +0,035. Mức này nằm sát ngưỡng nhiễu.
- Với bge-m3, Dense + Rerank bằng Hybrid + Rerank ở mức Điều: BM25 gần như không còn đóng góp thêm.
- Chi phí khi chạy (thêm khoảng 140 ms mỗi câu hỏi trên CPU) rất nhỏ so với 15 - 30 giây của LLM. Chi phí lớn là ingest (38 phút, làm một lần) và RAM.
- Ghi chú: một lần chạy Hybrid + Rerank bị dừng không có lỗi và không ghi kết quả; chạy lại thì bình thường. Chưa rõ nguyên nhân (log kernel không có dấu hiệu hết RAM). Hiện tượng giống lần chạy ngày 2026-10-11.

---

## Cố định số ứng viên đưa vào rerank - 2026-10-15

### 1. Vấn đề và cách sửa
- Trước đây `HybridRetriever.search(limit)` trả về cả danh sách gộp RRF (từ `limit` đến `2 × limit` đoạn), và mỗi nơi tự tính số ứng viên bằng `top_k × 3`. Vì vậy reranker chấm số ứng viên khác nhau giữa các cấu hình: Dense + Rerank chấm 18, Hybrid + Rerank chấm 18 - 36 (trong eval). Hai cấu hình không so được với nhau.
- Hiện tại: `search()` trả về đúng `limit` đoạn, và mọi nơi có rerank dùng chung `RERANK_CANDIDATES` (mặc định 30).
- Thêm chỉ số **Recall@N** (đo trước rerank): đáp án có nằm trong N kết quả đầu của retriever không.

### 2. Cấu hình
- **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, collection `legal_chunks`
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`, device `cuda` (RTX 2060)
- **Chunking:** `chunk_size=600`, `chunk_overlap=80` (Điều → Khoản → nhóm Điểm), cùng BM25 index
- **Số câu hỏi đánh giá:** 60 câu (Zalo Legal test set). Mỗi cấu hình chạy 1 lần.

### 3. Recall trước rerank (mức Điều)

| Retriever | Recall@10 | Recall@20 | Recall@30 | Recall@50 |
|---|---|---|---|---|
| Dense | 56.67% | 63.33% | 66.67% | 73.33% |
| Hybrid | 83.33% | 88.33% | 88.33% | 93.33% |

Mức Văn bản, Hybrid: 96.67% / 98.33% / 98.33% / 100.00%.

### 4. Sau rerank (mức Điều)

| Cấu hình | Ứng viên | Hit@4 | Hit@6 | MRR | Rerank trung vị |
|---|---|---|---|---|---|
| Dense + Rerank (cũ, 2026-10-11) | 18 | 63.33% | 63.33% | 0.6083 | - |
| Dense + Rerank | 30 | 65.00% | 66.67% | 0.6242 | 857 ms |
| Dense + Rerank | 50 | 71.67% | 73.33% | 0.6825 | 1317 ms |
| Hybrid + Rerank (cũ, 2026-10-11) | 18 - 36 | 85.00% | 88.33% | 0.7811 | - |
| Hybrid + Rerank | 30 | 85.00% | 88.33% | 0.7811 | 868 ms |
| Hybrid + Rerank | 50 | 88.33% | 91.67% | 0.8150 | 1379 ms |

Mức Văn bản Hit@4: Hybrid + Rerank 95.00% (30 ứng viên), 96.67% (50 ứng viên).

### 5. Nhận xét
- **BM25 thật sự có đóng góp.** Khi cùng 30 ứng viên, Hybrid + Rerank hơn Dense + Rerank 20 điểm Hit@4 ở mức Điều (12/60 câu), lớn hơn nhiều so với mức nhiễu. Nghi ngờ trước đây (mức tăng có thể chỉ do Hybrid được chấm nhiều ứng viên hơn) không đúng.
- **Hybrid + Rerank với 30 ứng viên cho kết quả giống hệt bản cũ**, nên phần sửa không làm giảm chất lượng của cấu hình đang chạy.
- **Ở 30 ứng viên, reranker đã lấy được gần hết những gì retriever tìm ra:** Recall@30 = 88.33% bằng Hit@6 = 88.33%. Phần chưa tìm được nằm ngoài 30 ứng viên đầu.
- **Lên 50 ứng viên:** Recall tăng 5 điểm (3/60 câu), Hit@4 sau rerank tăng 3.3 điểm (2/60 câu), nằm ở mức nhiễu của bộ 60 câu. Đổi lại thời gian rerank tăng thêm khoảng 0.5 giây mỗi câu (+59%).
- **Quyết định: giữ 30.** Có thể thử lại 50 khi có bộ eval lớn hơn.
- Rerank trên GPU mất khoảng 0.86 giây mỗi câu, chậm hơn nhiều so với hybrid search (0.03 - 0.23 giây, mục đo độ trễ ở trên). Trên CPU sẽ còn chậm hơn.
