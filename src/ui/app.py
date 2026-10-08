import streamlit as st
import requests
import json

# App settings
st.set_page_config(
    page_title="VN Legal RAG",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Mình là trợ lý pháp luật. Bạn có thắc mắc gì về quy định, nghị định hay luật pháp Việt Nam không?",
        "sources": []
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Xem nguồn trích dẫn"):
                for src in message["sources"]:
                    st.markdown(
                        f"""<div class="source-box">
                        <strong>Văn bản:</strong> {src.get("title")} (Điều {src.get("article_number")})<br>
                        <strong>Nội dung:</strong> {src.get("content")}<br>
                        <em>Độ tin cậy (Score): {src.get("score"):.2f}</em>
                        </div>""", 
                        unsafe_allow_html=True
                    )

# React to user input
if prompt := st.chat_input("Nhập câu hỏi pháp lý của bạn vào đây..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Đang tra cứu tài liệu và suy nghĩ...*")
        
        try:
            # Call API
            response = requests.post(
                "http://localhost:8000/ask",
                json={"question": prompt, "limit": 4},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("answer", "Xin lỗi, đã có lỗi xảy ra khi tạo câu trả lời.")
            sources = data.get("sources", [])
            
            # Show final answer
            message_placeholder.markdown(answer)
            
            # Show sources
            if sources:
                with st.expander("📚 Xem nguồn trích dẫn"):
                    for src in sources:
                        st.markdown(
                            f"""<div class="source-box">
                            <strong>Văn bản:</strong> {src.get("title")} (Điều {src.get("article_number")})<br>
                            <strong>Nội dung:</strong> {src.get("content")}<br>
                            <em>Độ tin cậy (Score): {src.get("score"):.2f}</em>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Lỗi kết nối tới API: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg,
                "sources": []
            })
