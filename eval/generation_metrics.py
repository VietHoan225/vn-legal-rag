"""Rule-based metrics for generated answers (no LLM calls)."""

import re
import unicodedata
from typing import Any

REFUSAL_MARKER = "chưa tìm thấy quy định cụ thể"


def normalize(text: str) -> str:
    """NFC-normalize and lowercase text so Vietnamese diacritics compare reliably."""
    return unicodedata.normalize("NFC", text).lower()


def strip_thinking(answer: str) -> str:
    """Remove <think>...</think> blocks that some reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.S).strip()


def split_article_id(article_id: str) -> tuple[str, str]:
    """Split '100/2019/nđ-cp#6' into ('100/2019/nđ-cp', '6')."""
    doc_code, _, article_number = article_id.partition("#")
    return normalize(doc_code), article_number


def cites_document(answer: str, article_id: str) -> bool:
    """True if the answer mentions the document code, e.g. '100/2019/NĐ-CP'."""
    doc_code, _ = split_article_id(article_id)
    return doc_code in normalize(answer)


def cites_article(answer: str, article_id: str) -> bool:
    """True if the answer mentions both the document code and 'Điều N'."""
    _, article_number = split_article_id(article_id)
    pattern = rf"điều\s+{re.escape(article_number)}(?!\d)"
    return cites_document(answer, article_id) and re.search(pattern, normalize(answer)) is not None


def cites_clause(answer: str, clause_number: int) -> bool:
    """True if the answer mentions 'Khoản K'."""
    return re.search(rf"khoản\s+{clause_number}(?!\d)", normalize(answer)) is not None


def is_refusal(answer: str) -> bool:
    """True if the answer uses the refusal sentence from the RAG prompt."""
    return REFUSAL_MARKER in normalize(answer)


def _digits_only_numbers(text: str) -> str:
    """Drop thousand separators so '200.000' and '200,000' both become '200000'."""
    return re.sub(r"(?<=\d)[.,](?=\d{3}(?!\d))", "", text)


def matches_facts(answer: str, facts: list[str]) -> bool:
    """True if every expected fact (e.g. a fine amount) appears in the answer."""
    haystack = _digits_only_numbers(normalize(answer))
    for fact in facts:
        needle = re.escape(_digits_only_numbers(normalize(fact)))
        # Digit boundaries so '100000' does not match inside '1000000'
        if re.search(rf"(?<!\d){needle}(?!\d)", haystack) is None:
            return False
    return True


def score_answer(answer: str, item: dict[str, Any]) -> dict[str, Any]:
    """Compute per-question metrics. Optional keys: expected_clause, expected_facts.

    Citation metrics only count answers that are not refusals: a refusal that
    mentions the right article is still not a correct answer.
    """
    answer = strip_thinking(answer)
    expected_articles = item.get("expected_articles", [])
    refusal = is_refusal(answer)
    scores: dict[str, Any] = {
        "refusal": refusal,
        "doc_cited": not refusal and any(cites_document(answer, a) for a in expected_articles),
        "article_cited": not refusal and any(cites_article(answer, a) for a in expected_articles),
    }
    if item.get("expected_clause") is not None:
        scores["clause_cited"] = scores["article_cited"] and cites_clause(answer, item["expected_clause"])
    if item.get("expected_facts"):
        scores["facts_correct"] = matches_facts(answer, item["expected_facts"])
    return scores


def aggregate(per_item: list[dict[str, Any]]) -> dict[str, float]:
    """Average boolean metrics over the items that have them."""
    keys = sorted({k for s in per_item for k in s})
    result = {}
    for k in keys:
        values = [s[k] for s in per_item if k in s]
        result[k] = sum(values) / len(values) if values else 0.0
    return result
