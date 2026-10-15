"""Unit tests for LawCatalog (small in-memory documents, no data files)."""

import json

import pytest

from src.mcp_server.law_catalog import ArticleNotFoundError, LawCatalog, LawNotFoundError, fold
from tests.fixtures.sample_docs import SAMPLE_LEGAL_DOC_1

TT_DOC = {
    "doc_id": 2,
    "title": "Văn bản số 01/2020/TT-BTC",
    "category": "tt-btc",
    "content": "THÔNG TƯ\n\nĐiều 1. Phạm vi\nNội dung một.\n\nĐiều 2. Đối tượng\nNội dung hai.",
}
AMENDING_DOC = {
    "doc_id": 3,
    "title": "Văn bản số 12/2017/QH14",
    "category": "qh14",
    "content": 'Điều 1. Sửa đổi một số điều\n1. Sửa đổi Điều 3 như sau:\n“Điều 3. Nguyên tắc\nNội dung được trích.”\nĐiều 2. Hiệu lực\nCó hiệu lực từ 2018.',
}


@pytest.fixture
def catalog() -> LawCatalog:
    return LawCatalog([SAMPLE_LEGAL_DOC_1, TT_DOC, AMENDING_DOC])


def test_fold_ignores_case_and_diacritics():
    assert fold("100/2019/NĐ-CP") == fold("100/2019/nd-cp") == "100/2019/nd-cp"


def test_get_article_returns_exact_text(catalog):
    law, text = catalog.get_article("99/2021/NĐ-CP", 2)
    assert law.code == "99/2021/NĐ-CP" and law.category == "nđ-cp"
    assert text.startswith("Điều 2. Đối tượng áp dụng")
    assert text.rstrip().endswith("Luật Giao thông đường bộ.")
    assert "Điều 3. Quy định xử phạt" not in text  # stops at the next article


def test_get_article_last_article_runs_to_end(catalog):
    _, text = catalog.get_article("01/2020/TT-BTC", 2)
    assert text == "Điều 2. Đối tượng\nNội dung hai."


@pytest.mark.parametrize("name", ["99/2021/nd-cp", "99/2021/ND-CP", "  99/2021/NĐ-CP ", "Văn bản số 99/2021/NĐ-CP", "Nghị định 99/2021/NĐ-CP"])
def test_find_is_forgiving_about_case_diacritics_and_wording(catalog, name):
    assert catalog.find(name).code == "99/2021/NĐ-CP"


def test_unknown_law_suggests_close_numbers(catalog):
    with pytest.raises(LawNotFoundError, match="99/2021/NĐ-CP"):
        catalog.find("99/2021/NĐ")  # partial number
    with pytest.raises(LawNotFoundError, match="Không tìm thấy"):
        catalog.find("không có luật này")


def test_unknown_article_reports_available_range(catalog):
    with pytest.raises(ArticleNotFoundError, match=r"không có Điều 99 .*từ Điều 1 đến Điều 3"):
        catalog.get_article("99/2021/NĐ-CP", 99)


def test_quoted_articles_inside_amending_law_are_not_top_level(catalog):
    """A quoted 'Điều 3' inside an amending law is not that law's own Điều 3."""
    _, text = catalog.get_article("12/2017/QH14", 1)
    assert "“Điều 3. Nguyên tắc" in text
    with pytest.raises(ArticleNotFoundError):
        catalog.get_article("12/2017/QH14", 3)


def test_list_laws_groups_by_category_biggest_first():
    docs = [SAMPLE_LEGAL_DOC_1, TT_DOC, AMENDING_DOC, {**SAMPLE_LEGAL_DOC_1, "title": "Văn bản số 05/2021/NĐ-CP"}]
    groups = LawCatalog(docs).list_laws()
    assert list(groups) == ["nđ-cp", "qh14", "tt-btc"]
    assert groups["nđ-cp"] == ["05/2021/NĐ-CP", "99/2021/NĐ-CP"]


def test_duplicate_numbers_keep_first_document():
    other = {**TT_DOC, "doc_id": 9, "content": "Điều 1. Khác\nBản trùng."}
    catalog = LawCatalog([TT_DOC, other])
    assert len(catalog) == 1
    assert "Nội dung một" in catalog.get_article("01/2020/TT-BTC", 1)[1]


def test_from_file_loads_json_and_reports_missing_file(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([TT_DOC], ensure_ascii=False), encoding="utf-8")
    assert len(LawCatalog.from_file(path)) == 1
    with pytest.raises(FileNotFoundError, match="not found"):
        LawCatalog.from_file(tmp_path / "missing.json")
