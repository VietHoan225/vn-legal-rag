import html
import os
import uuid

import requests
import streamlit as st

# App settings
st.set_page_config(
    page_title="VN Legal RAG",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    .title-text {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        text-align: center;
        opacity: 0.8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .source-box {
        border-left: 4px solid #3b82f6;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.9em;
        background-color: rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title-text">⚖️ Trợ Lý Pháp Luật Việt Nam</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Giải đáp thắc mắc dựa trên cơ sở dữ liệu Zalo Legal</p>', unsafe_allow_html=True)

def source_location(meta: dict) -> str:
    """Citation like 'Điều 6, Khoản 2'; falls back to article number for older API responses."""
    if meta.get("location"):
        return meta["location"]
    return f"Điều {meta.get('article_number', '?')}"


def render_sources(sources: list[dict]) -> None:
    """Show retrieved sources in a collapsible block."""
    if not sources:
        return
    with st.expander("📚 Xem nguồn trích dẫn"):
        for src in sources:
            meta = src.get("metadata", {})
            title = html.escape(str(meta.get("title", "Không rõ")))
            content = html.escape(str(src.get("content", ""))).replace("\n", "<br>")
            st.markdown(
                f"""<div class="source-box">
                <strong>{title}</strong> · {html.escape(source_location(meta))}<br>
                {content}<br>
                <em>Điểm truy hồi: {src.get("score", 0.0):.3f}</em>
                </div>""",
                unsafe_allow_html=True,
            )

API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Agent may call the LLM up to 3 times per message, so allow more than a single call
REQUEST_TIMEOUT = 120
WELCOME = {
    "role": "assistant",
    "content": "Xin chào! Mình là trợ lý pháp luật. Bạn có thắc mắc gì về quy định, nghị định hay luật pháp Việt Nam không?",
    "sources": [],
}


def new_conversation() -> None:
    """Start a fresh chat: new session id (server-side memory) and cleared history."""
    old_id = st.session_state.get("session_id")
    if old_id:
        try:
            requests.delete(f"{API_URL}/chat/{old_id}", timeout=5)
        except requests.exceptions.RequestException:
            pass  # server drops old sessions on its own anyway
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = [dict(WELCOME)]


def error_text(exc: requests.exceptions.RequestException) -> str:
    """Turn a request failure into a short message for the user."""
    if isinstance(exc, requests.exceptions.Timeout):
        return "⏱️ Hệ thống phản hồi quá lâu. Vui lòng thử lại."
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"❌ Không kết nối được tới API ({API_URL}). API đã được bật chưa?"
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail", "")
        except ValueError:
            detail = exc.response.text
        icon = "⏳" if exc.response.status_code in (429, 504) else "❌"
        return f"{icon} {detail or exc}"
    return f"❌ Lỗi: {exc}"


if "session_id" not in st.session_state:
    new_conversation()

with st.sidebar:
    st.button("🗑️ Cuộc trò chuyện mới", on_click=new_conversation, use_container_width=True)
    st.caption("Trợ lý nhớ ngữ cảnh trong cùng một cuộc trò chuyện.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

# React to user input
if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn vào đây..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Đang suy nghĩ...*")

        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"session_id": st.session_state.session_id, "message": prompt},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer") or "Xin lỗi, đã có lỗi xảy ra khi tạo câu trả lời."
            sources = data.get("sources", [])
        except requests.exceptions.RequestException as e:
            answer, sources = error_text(e), []

        message_placeholder.markdown(answer)
        render_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
