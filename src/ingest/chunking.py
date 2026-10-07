"""Legal text chunking module preserving article boundaries and metadata."""

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass
class LegalChunk:
    """Represents a chunk extracted from a legal document."""

    chunk_id: str
    doc_id: int
    article_id: str
    article_number: int
    title: str
    category: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return asdict(self)


def normalize_title_to_code(title: str) -> str:
    """Extract and normalize law code from title (e.g. '41/2019/qh14')."""
    match = re.search(r"Văn bản số\s+(.+)", title, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return title.strip().lower()


def split_legal_sentences(text: str) -> list[str]:
    """Split text into sentences/clauses while avoiding breaking abbreviations and numbers."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    segments = []
    for p in paragraphs:
        # Split on sentence terminals (; ! ?) or terminal dot followed by uppercase/digit.
        # Avoid splitting thousand separators (e.g. 200.000) or decimal points.
        sub = re.split(r"(?<=[;!?])\s+|(?<=[^\d\s]\.)\s+(?=[A-ZÀ-Ỹ0-9])", p)
        for s in sub:
            s_clean = s.strip()
            if s_clean:
                segments.append(s_clean)
    return segments


def split_long_segment(segment: str, max_len: int) -> list[str]:
    """Split an oversized segment at clause/word boundaries without cutting words."""
    if len(segment) <= max_len:
        return [segment]

    parts: list[str] = []
    comma_parts = re.split(r"(?<=,)\s+", segment)
    current = ""

    for p in comma_parts:
        if len(p) > max_len:
            words = p.split()
            w_curr = ""
            for w in words:
                if len(w_curr) + len(w) + 1 <= max_len:
                    w_curr = f"{w_curr} {w}".strip()
                else:
                    if w_curr:
                        parts.append(w_curr)
                    w_curr = w
            if w_curr:
                parts.append(w_curr)
        elif len(current) + len(p) + 1 <= max_len:
            current = f"{current} {p}".strip()
        else:
            if current:
                parts.append(current)
            current = p

    if current:
        parts.append(current)
    return parts


def chunk_document(
    doc: dict[str, Any],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[LegalChunk]:
    """Chunk a legal document by Article, with sliding window for long articles.

    Args:
        doc: Document dictionary containing doc_id, title, category, content.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Target character overlap between consecutive sub-chunks.

    Returns:
        List of LegalChunk objects.
    """
    content = doc.get("content", "")
    doc_id = doc.get("doc_id", 0)
    title = doc.get("title", "")
    category = doc.get("category", "")
    law_code = normalize_title_to_code(title)

    # Match 'Điều N.' or 'Điều N:' only at line/paragraph start
    pattern = re.compile(r"(?m)^\s*(Điều\s+(\d+)[\.:])")
    matches = list(pattern.finditer(content))
    if not matches:
        return []

    chunks: list[LegalChunk] = []
    chunk_idx = 0

    for i, m in enumerate(matches):
        art_num = int(m.group(2))
        art_id = f"{law_code}#{art_num}"
        start_pos = m.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        art_text = content[start_pos:end_pos].strip()

        prefix = f"{title} - Điều {art_num}"
        full_text = f"{prefix}\n{art_text}"

        # If the entire article fits within chunk_size, create 1 chunk
        if len(full_text) <= chunk_size:
            chunks.append(
                LegalChunk(
                    chunk_id=f"{doc_id}_{chunk_idx}",
                    doc_id=doc_id,
                    article_id=art_id,
                    article_number=art_num,
                    title=title,
                    category=category,
                    text=full_text,
                )
            )
            chunk_idx += 1
        else:
            avail_size = max(50, chunk_size - len(prefix) - 1)
            raw_sents = split_legal_sentences(art_text)
            norm_sents: list[str] = []
            for s in raw_sents:
                norm_sents.extend(split_long_segment(s, avail_size))

            curr_sents: list[str] = []
            curr_len = 0

            for s in norm_sents:
                add_len = len(s) + (1 if curr_sents else 0)
                if curr_len + add_len <= avail_size:
                    curr_sents.append(s)
                    curr_len += add_len
                else:
                    if curr_sents:
                        chunk_body = " ".join(curr_sents)
                        chunks.append(
                            LegalChunk(
                                chunk_id=f"{doc_id}_{chunk_idx}",
                                doc_id=doc_id,
                                article_id=art_id,
                                article_number=art_num,
                                title=title,
                                category=category,
                                text=f"{prefix}\n{chunk_body}",
                            )
                        )
                        chunk_idx += 1

                        # Compute overlap from trailing sentences
                        overlap_sents: list[str] = []
                        o_len = 0
                        for old_s in reversed(curr_sents):
                            o_add = len(old_s) + (1 if overlap_sents else 0)
                            if o_len + o_add <= chunk_overlap:
                                overlap_sents.insert(0, old_s)
                                o_len += o_add
                            else:
                                break

                        # Guarantee progress: do not repeat all sentences
                        if len(overlap_sents) == len(curr_sents) and len(curr_sents) > 1:
                            overlap_sents = overlap_sents[1:]
                            o_len = sum(len(x) for x in overlap_sents) + max(0, len(overlap_sents) - 1)

                        # Prune overlap sentences if combined with s exceeds avail_size
                        while overlap_sents and (o_len + len(s) + 1 > avail_size):
                            overlap_sents.pop(0)
                            o_len = sum(len(x) for x in overlap_sents) + max(0, len(overlap_sents) - 1)

                        curr_sents = overlap_sents.copy()
                        curr_len = o_len

                    curr_sents.append(s)
                    curr_len += len(s) + (1 if len(curr_sents) > 1 else 0)

            if curr_sents:
                chunk_body = " ".join(curr_sents)
                chunks.append(
                    LegalChunk(
                        chunk_id=f"{doc_id}_{chunk_idx}",
                        doc_id=doc_id,
                        article_id=art_id,
                        article_number=art_num,
                        title=title,
                        category=category,
                        text=f"{prefix}\n{chunk_body}",
                    )
                )
                chunk_idx += 1

    return chunks


def chunk_documents(
    docs: list[dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[LegalChunk]:
    """Chunk multiple legal documents."""
    all_chunks: list[LegalChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
