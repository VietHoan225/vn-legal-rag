"""Catalog of the raw legal documents: list laws and return the full text of one article."""

from dataclasses import dataclass
import difflib
import json
from pathlib import Path
import unicodedata
from typing import Any

from src.ingest.chunking import ARTICLE_PATTERN

TITLE_PREFIX = "Văn bản số "
MAX_SUGGESTIONS = 5


class LawNotFoundError(LookupError):
    """No document matches the requested law number."""


class ArticleNotFoundError(LookupError):
    """The law exists but has no such article."""


def fold(text: str) -> str:
    """Lowercase and strip Vietnamese diacritics so 'NĐ-CP' matches 'nd-cp'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").casefold().strip()


@dataclass
class Law:
    """One document: its number (e.g. '100/2019/NĐ-CP'), category and article spans in `content`."""

    code: str
    category: str
    content: str
    articles: dict[int, tuple[int, int]]


class LawCatalog:
    """In-memory index over the raw documents, keyed by law number."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._laws: dict[str, Law] = {}
        for doc in docs:
            code = str(doc.get("title", "")).removeprefix(TITLE_PREFIX).strip()
            key = fold(code)
            if not code or key in self._laws:
                continue
            content = str(doc.get("content", ""))
            self._laws[key] = Law(code, str(doc.get("category", "")), content, self._article_spans(content))

    @classmethod
    def from_file(cls, path: Path) -> "LawCatalog":
        """Load the raw JSON file once."""
        if not path.exists():
            raise FileNotFoundError(f"Raw legal documents not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    def _article_spans(content: str) -> dict[int, tuple[int, int]]:
        """Map article number -> (start, end) offsets; the first occurrence of a number wins."""
        matches = list(ARTICLE_PATTERN.finditer(content))
        spans: dict[int, tuple[int, int]] = {}
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            spans.setdefault(int(m.group(2)), (m.start(), end))
        return spans

    def __len__(self) -> int:
        return len(self._laws)

    def list_laws(self) -> dict[str, list[str]]:
        """Law numbers grouped by category, biggest categories first."""
        groups: dict[str, list[str]] = {}
        for law in self._laws.values():
            groups.setdefault(law.category, []).append(law.code)
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return {category: sorted(codes) for category, codes in ordered}

    def find(self, law_name: str) -> Law:
        """Find a law by number, ignoring case and diacritics ('100/2019/nd-cp' works).

        Also accepts a longer phrase containing the number ('Nghị định 100/2019/NĐ-CP').
        Raises LawNotFoundError with close suggestions otherwise.
        """
        query = fold(law_name).removeprefix(fold(TITLE_PREFIX)).strip()
        if query in self._laws:
            return self._laws[query]
        contained = [k for k in self._laws if k in query]
        if contained:
            return self._laws[max(contained, key=len)]

        suggestions = [k for k in self._laws if query and query in k]
        if not suggestions:
            suggestions = difflib.get_close_matches(query, list(self._laws), n=MAX_SUGGESTIONS, cutoff=0.6)
        hint = ", ".join(self._laws[k].code for k in suggestions[:MAX_SUGGESTIONS])
        raise LawNotFoundError(
            f"Không tìm thấy văn bản '{law_name}'. Dùng số hiệu như '100/2019/NĐ-CP' (xem list_laws)."
            + (f" Có thể bạn muốn: {hint}." if hint else "")
        )

    def get_article(self, law_name: str, article_number: int) -> tuple[Law, str]:
        """Return (law, full article text). Raises LawNotFoundError or ArticleNotFoundError."""
        law = self.find(law_name)
        span = law.articles.get(article_number)
        if span is None:
            numbers = sorted(law.articles)
            raise ArticleNotFoundError(
                f"Văn bản {law.code} không có Điều {article_number} "
                f"(có {len(numbers)} Điều, từ Điều {numbers[0]} đến Điều {numbers[-1]})."
                if numbers
                else f"Văn bản {law.code} không có Điều nào được nhận diện."
            )
        return law, law.content[span[0] : span[1]].strip()
