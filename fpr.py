import streamlit as st
from pathlib import Path
from cv_bot import graph, extract_pdf_text
from langchain_core.messages import AIMessage

st.set_page_config(page_title="Le Phan Anh", layout="wide")
st.title("Le Phan Anh feedback CV")

# ✅ Khởi tạo session_state
if "cv_state" not in st.session_state:
    st.session_state.cv_state = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("📄 Tải lên CV (.pdf)", type=["pdf"])

# ✅ Giai đoạn upload và đánh giá CV
if uploaded_file and st.button("🚀 Đánh giá CV"):
    with st.spinner("Đang xử lý..."):
        # Lưu tạm file PDF
        temp_path = Path("temp_cv.pdf")
        temp_path.write_bytes(uploaded_file.read())

        # Đọc văn bản
        cv_text = extract_pdf_text(str(temp_path))

        # Tạo state ban đầu
        init_state = {
            "file_path": str(temp_path),
            "cv_text": cv_text,
            "user_query": "",
            "already_ask": False,
            "ask_more": False
        }

        # Chạy pipeline
        result = graph.invoke(init_state)
        st.session_state.cv_state = result  # lưu lại để chat sau

        # Hiện kết quả
        st.subheader("🧠 Đánh giá:")
        st.markdown(result.get("evaluation", "Không có đánh giá."))

        st.subheader("📌 Kết luận:")
        st.success(result.get("decision", "Không có kết luận."))

        st.session_state.chat_history = []  # reset lịch sử chat

# ✅ Giai đoạn hỏi thêm sau khi đánh giá
if st.session_state.cv_state:
    st.markdown("---")
    st.subheader("💬 Hỏi thêm về CV")

    user_question = st.chat_input("Bạn muốn hỏi gì tiếp?")
    if user_question:
        state = st.session_state.cv_state.copy()
        state["user_query"] = user_question
        state["ask_more"] = False
        state["already_ask"] = True

        result = graph.invoke(state)
        answer = result.get("mes", "Không có câu trả lời")
        if hasattr(answer, "content"):
            answer = answer.content

        # Lưu lịch sử
        st.session_state.chat_history.append((user_question, answer))

    # Hiện lại toàn bộ lịch sử chat
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)