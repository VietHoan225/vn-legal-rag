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
    clause_number: int | None = None

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


CLAUSE_PATTERN = re.compile(r"(?m)^[ \t]*(\d+)\.\s")
POINT_PATTERN = re.compile(r"(?m)^[ \t]*[a-zđ]\)\s")


def truncate_at_word(text: str, max_len: int) -> str:
    """Truncate text to at most max_len chars at a word boundary, appending '…'."""
    if len(text) <= max_len:
        return text
    cut = text[: max(1, max_len - 1)].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}…"


def split_clauses(art_text: str) -> tuple[str, list[tuple[int, str]]]:
    """Split article text into (preamble, [(clause_number, clause_text), ...]).

    A clause (Khoản) is a line starting with 'N. '. The preamble holds the
    article heading and any text before clause 1.
    """
    matches = list(CLAUSE_PATTERN.finditer(art_text))
    if not matches:
        return art_text.strip(), []
    preamble = art_text[: matches[0].start()].strip()
    clauses = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(art_text)
        clauses.append((int(m.group(1)), art_text[m.start() : end].strip()))
    return preamble, clauses


def split_points(clause_text: str) -> tuple[str, list[str]]:
    """Split clause text into (lead_in, [point_text, ...]).

    A point (Điểm) is a line starting with 'a) ', 'đ) ', ... The lead-in is
    the text before the first point, usually carrying the penalty range.
    """
    matches = list(POINT_PATTERN.finditer(clause_text))
    if not matches:
        return clause_text.strip(), []
    lead_in = clause_text[: matches[0].start()].strip()
    points = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clause_text)
        points.append(clause_text[m.start() : end].strip())
    return lead_in, points


def window_split(text: str, max_len: int, overlap: int) -> list[str]:
    """Split text into sentence-aligned windows of at most max_len chars with overlap."""
    norm_sents: list[str] = []
    for s in split_legal_sentences(text):
        norm_sents.extend(split_long_segment(s, max_len))

    bodies: list[str] = []
    curr_sents: list[str] = []
    curr_len = 0

    for s in norm_sents:
        add_len = len(s) + (1 if curr_sents else 0)
        if curr_len + add_len <= max_len:
            curr_sents.append(s)
            curr_len += add_len
            continue

        if curr_sents:
            bodies.append(" ".join(curr_sents))

            # Compute overlap from trailing sentences
            overlap_sents: list[str] = []
            o_len = 0
            for old_s in reversed(curr_sents):
                o_add = len(old_s) + (1 if overlap_sents else 0)
                if o_len + o_add <= overlap:
                    overlap_sents.insert(0, old_s)
                    o_len += o_add
                else:
                    break

            # Guarantee progress: do not repeat all sentences
            if len(overlap_sents) == len(curr_sents) and len(curr_sents) > 1:
                overlap_sents = overlap_sents[1:]
                o_len = sum(len(x) for x in overlap_sents) + max(0, len(overlap_sents) - 1)

            # Prune overlap sentences if combined with s exceeds max_len
            while overlap_sents and (o_len + len(s) + 1 > max_len):
                overlap_sents.pop(0)
                o_len = sum(len(x) for x in overlap_sents) + max(0, len(overlap_sents) - 1)

            curr_sents = overlap_sents.copy()
            curr_len = o_len

        curr_sents.append(s)
        curr_len += len(s) + (1 if len(curr_sents) > 1 else 0)

    if curr_sents:
        bodies.append(" ".join(curr_sents))
    return bodies


def chunk_clause(clause_text: str, heading: str, max_len: int, overlap: int) -> list[str]:
    """Chunk one clause so every chunk keeps the article heading and clause lead-in.

    Points are grouped without being cut; each group repeats the (possibly
    truncated) lead-in so a violation always stays next to its penalty.
    """
    whole = f"{heading}\n{clause_text}"
    if len(whole) <= max_len:
        return [whole]

    lead_in, points = split_points(clause_text)
    if not points:
        return [f"{heading}\n{b}" for b in window_split(clause_text, max_len - len(heading) - 1, overlap)]

    bodies: list[str] = []
    lead_ctx = truncate_at_word(lead_in, max_len // 3)
    if lead_ctx != lead_in:
        # Keep the full lead-in searchable on its own.
        bodies.extend(f"{heading}\n{b}" for b in window_split(lead_in, max_len - len(heading) - 1, overlap))

    ctx = f"{heading}\n{lead_ctx}"
    room = max_len - len(ctx) - 1
    pieces: list[str] = []
    for pt in points:
        pieces.extend(split_long_segment(pt, room))

    group: list[str] = []
    group_len = 0
    for piece in pieces:
        add_len = len(piece) + (1 if group else 0)
        if group and group_len + add_len > room:
            bodies.append(ctx + "\n" + "\n".join(group))
            group, group_len = [], 0
            add_len = len(piece)
        group.append(piece)
        group_len += add_len
    if group:
        bodies.append(ctx + "\n" + "\n".join(group))
    return bodies


def chunk_document(
    doc: dict[str, Any],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[LegalChunk]:
    """Chunk a legal document by Article, then Clause (Khoản), then Point (Điểm).

    Short articles become one chunk. Long articles are split per clause; long
    clauses are split into groups of points, each repeating the article heading
    and clause lead-in. Articles without clauses fall back to sentence windows.

    Args:
        doc: Document dictionary containing doc_id, title, category, content.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Overlap between sentence windows (fallback path only).

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

    def add(art_num: int, clause_num: int | None, prefix: str, body: str) -> None:
        chunks.append(
            LegalChunk(
                chunk_id=f"{doc_id}_{len(chunks)}",
                doc_id=doc_id,
                article_id=f"{law_code}#{art_num}",
                article_number=art_num,
                title=title,
                category=category,
                text=f"{prefix}\n{body}",
                clause_number=clause_num,
            )
        )

    for i, m in enumerate(matches):
        art_num = int(m.group(2))
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        art_text = content[m.start() : end_pos].strip()
        art_prefix = f"{title} - Điều {art_num}"

        # If the entire article fits within chunk_size, create 1 chunk
        if len(art_prefix) + 1 + len(art_text) <= chunk_size:
            add(art_num, None, art_prefix, art_text)
            continue

        preamble, clauses = split_clauses(art_text)
        if not clauses:
            avail = max(50, chunk_size - len(art_prefix) - 1)
            for body in window_split(art_text, avail, chunk_overlap):
                add(art_num, None, art_prefix, body)
            continue

        heading_line = preamble.split("\n", 1)[0]
        extra_preamble = preamble[len(heading_line) :].strip()
        if extra_preamble:
            avail = max(50, chunk_size - len(art_prefix) - 1)
            for body in window_split(preamble, avail, chunk_overlap):
                add(art_num, None, art_prefix, body)

        for clause_num, clause_text in clauses:
            prefix = f"{art_prefix} - Khoản {clause_num}"
            avail = max(50, chunk_size - len(prefix) - 1)
            heading = truncate_at_word(heading_line, avail // 4)
            for body in chunk_clause(clause_text, heading, avail, chunk_overlap):
                add(art_num, clause_num, prefix, body)

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
