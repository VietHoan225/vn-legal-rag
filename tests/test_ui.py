"""UI tests with Streamlit's AppTest and a fake streaming API (no network)."""

import json
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py")

SOURCE = {
    "id": "39_1",
    "content": "Phạt tiền từ 200.000 đồng",
    "score": 0.03,
    "metadata": {"title": "Văn bản số 100/2019/NĐ-CP", "article_number": 6, "location": "Điều 6, Khoản 2"},
}


def ndjson(*events: dict) -> list[str]:
    return [json.dumps(e, ensure_ascii=False) for e in events]


class FakeStreamResponse:
    """Minimal stand-in for requests.Response(stream=True)."""

    def __init__(self, lines: list[str], status_code: int = 200, detail: str = ""):
        self.lines = lines
        self.status_code = status_code
        self.detail = detail
        self.encoding = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return {"detail": self.detail}

    def iter_lines(self, decode_unicode=False):
        yield from self.lines


def ask(response: FakeStreamResponse | Exception) -> AppTest:
    """Run the app, send one message, and return the AppTest."""
    with patch("requests.post") as post:
        if isinstance(response, Exception):
            post.side_effect = response
        else:
            post.return_value = response
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("Không đội mũ bảo hiểm phạt bao nhiêu?").run()
    assert not at.exception
    return at


def test_streamed_answer_and_sources_are_stored():
    lines = ndjson(
        {"type": "token", "text": "Phạt "},
        {"type": "token", "text": "200.000 đồng"},
        {"type": "sources", "sources": [SOURCE]},
    )
    at = ask(FakeStreamResponse(lines))

    reply = at.session_state.messages[-1]
    assert reply["content"] == "Phạt 200.000 đồng"
    assert reply["sources"] == [SOURCE]
    assert any("Điều 6, Khoản 2" in m.value for m in at.markdown)


def test_http_error_shows_detail_from_api():
    at = ask(FakeStreamResponse([], status_code=429, detail="Hệ thống đang quá tải"))
    assert "Hệ thống đang quá tải" in at.session_state.messages[-1]["content"]


def test_error_event_after_partial_answer_keeps_the_text():
    lines = ndjson({"type": "token", "text": "Phạt 200"}, {"type": "error", "detail": "Hệ thống đang quá tải"})
    content = ask(FakeStreamResponse(lines)).session_state.messages[-1]["content"]
    assert content.startswith("Phạt 200")
    assert "quá tải" in content


def test_connection_error_message():
    at = ask(requests.exceptions.ConnectionError("refused"))
    assert "Không kết nối được tới API" in at.session_state.messages[-1]["content"]
