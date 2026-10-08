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