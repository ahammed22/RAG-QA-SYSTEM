import streamlit as st
import tempfile
import os
from rag_engine import load_and_index_pdf, get_answer

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A System")
st.markdown("Upload a PDF and ask any question about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading and indexing your PDF..."):
        vectorstore = load_and_index_pdf(tmp_path)

    st.success("PDF indexed! Ask your question below.")

    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Finding answer..."):
            answer = get_answer(vectorstore, question)
        st.markdown("### 💡 Answer:")
        st.write(answer)

    os.unlink(tmp_path)