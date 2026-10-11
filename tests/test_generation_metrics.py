"""Unit tests for rule-based generation metrics (no LLM calls)."""

from eval.generation_metrics import (
    aggregate,
    cites_article,
    cites_clause,
    cites_document,
    is_refusal,
    matches_facts,
    score_answer,
    strip_thinking,
)

ARTICLE_ID = "100/2019/nđ-cp#6"
GOOD_ANSWER = (
    "**Kết luận trực tiếp**: Phạt tiền từ 200.000 đồng đến 300.000 đồng.\n"
    "**Cơ sở pháp lý**: Điểm i Khoản 2 Điều 6 Nghị định 100/2019/NĐ-CP."
)


def test_cites_document_is_case_insensitive():
    assert cites_document("Theo Nghị định 100/2019/NĐ-CP", ARTICLE_ID)
    assert not cites_document("Theo Nghị định 46/2016/NĐ-CP", ARTICLE_ID)


def test_cites_article_requires_exact_number():
    assert cites_article(GOOD_ANSWER, ARTICLE_ID)
    assert not cites_article("Điều 60 Nghị định 100/2019/NĐ-CP", ARTICLE_ID)
    assert not cites_article("Điều 6 Nghị định 46/2016/NĐ-CP", ARTICLE_ID)


def test_cites_clause():
    assert cites_clause(GOOD_ANSWER, 2)
    assert not cites_clause(GOOD_ANSWER, 1)
    assert not cites_clause("Khoản 12 Điều 6", 1)


def test_is_refusal():
    refusal = "Xin lỗi, dựa trên dữ liệu hiện tại, tôi chưa tìm thấy quy định cụ thể để trả lời câu hỏi của bạn."
    assert is_refusal(refusal)
    assert not is_refusal(GOOD_ANSWER)


def test_matches_facts_ignores_thousand_separators():
    assert matches_facts("phạt 200,000 đến 300000 đồng", ["200.000", "300.000"])
    assert not matches_facts("phạt 100.000 đến 200.000 đồng", ["200.000", "300.000"])
    # 1.000.000 must not be satisfied by 100.000
    assert not matches_facts("phạt 100.000 đồng", ["1.000.000"])
    assert not matches_facts("phạt 1.000.000 đồng", ["100.000"])


def test_strip_thinking():
    assert strip_thinking("<think>suy nghĩ</think>\nTrả lời") == "Trả lời"


def test_score_answer_full_item():
    item = {"expected_articles": [ARTICLE_ID], "expected_clause": 2, "expected_facts": ["200.000", "300.000"]}
    assert score_answer(GOOD_ANSWER, item) == {
        "refusal": False,
        "doc_cited": True,
        "article_cited": True,
        "clause_cited": True,
        "facts_correct": True,
    }


def test_score_answer_wrong_clause_and_fine():
    """The Khoản 1 / Khoản 2 mix-up from the helmet case study."""
    item = {"expected_articles": [ARTICLE_ID], "expected_clause": 2, "expected_facts": ["200.000", "300.000"]}
    answer = "Phạt từ 100.000 đồng đến 200.000 đồng theo Khoản 1 Điều 6 Nghị định 100/2019/NĐ-CP."
    scores = score_answer(answer, item)
    assert scores["article_cited"] is True
    assert scores["clause_cited"] is False
    assert scores["facts_correct"] is False


def test_score_answer_refusal_gets_no_citation_credit():
    answer = (
        "Xin lỗi, dựa trên dữ liệu hiện tại, tôi chưa tìm thấy quy định cụ thể để trả lời câu hỏi của bạn. "
        "Điều 6 Nghị định 100/2019/NĐ-CP không đề cập."
    )
    scores = score_answer(answer, {"expected_articles": [ARTICLE_ID]})
    assert scores == {"refusal": True, "doc_cited": False, "article_cited": False}


def test_score_answer_without_optional_fields():
    scores = score_answer("Không liên quan", {"expected_articles": [ARTICLE_ID]})
    assert set(scores) == {"refusal", "doc_cited", "article_cited"}


def test_aggregate_only_counts_items_with_metric():
    per_item = [
        {"article_cited": True, "facts_correct": True},
        {"article_cited": False, "facts_correct": False},
        {"article_cited": True},
    ]
    result = aggregate(per_item)
    assert result["article_cited"] == 2 / 3
    assert result["facts_correct"] == 0.5
