import os
import tempfile
import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG-Based Document Chatbot")
st.caption("Upload PDF documents and ask questions based on their contents.")

@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🔄 Process Documents", use_container_width=True):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Reading, chunking and indexing documents..."):
                total = 0
                for pdf in uploaded:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf.getvalue())
                        path = tmp.name
                    try:
                        total += engine.add_pdf(path, pdf.name)
                    finally:
                        os.unlink(path)
                st.success(f"Indexed {total} text chunks.")

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        engine.clear()
        st.success("Knowledge base cleared.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your uploaded documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = engine.ask(question)

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(f"- **{source['file']}**, page {source['page']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
