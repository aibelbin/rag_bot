"""
app.py — Streamlit RAG Study Assistant

Upload a PDF or TXT, ask questions, and get answers strictly
grounded in the document content.  Uses local embeddings (sentence-
transformers), a local FAISS vector store, and Groq for the LLM.

Run with:
    streamlit run app.py
"""

import streamlit as st

from utils import extract_text, file_hash
from vector_store import index_exists, build_and_save_index, load_index
from retriever import retrieve
from llm import ask_llm

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(page_title="RAG Study Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Study Assistant")
st.caption("Upload a document and ask questions — answers come only from your document.")

# ── Sidebar: API key + file upload ──────────────────────────────────
with st.sidebar:
    st.header("Settings")
    groq_key = st.text_input("Groq API Key", type="password", help="Required to query the LLM.")
    if groq_key:
        import os
        os.environ["GROQ_API_KEY"] = groq_key

    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Max ~200 pages recommended.",
    )

# ── Process uploaded file ───────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    doc_hash = file_hash(file_bytes)

    # Store the current document hash in session state
    if "doc_hash" not in st.session_state or st.session_state.doc_hash != doc_hash:
        st.session_state.doc_hash = doc_hash
        st.session_state.messages = []  # reset chat on new document

    # Build or load the FAISS index
    if index_exists(doc_hash):
        with st.sidebar:
            st.success("✅ Index loaded from cache.")
        index, chunks = load_index(doc_hash)
    else:
        with st.sidebar, st.spinner("Processing document — extracting, chunking, embedding…"):
            text = extract_text(uploaded_file.name, file_bytes)
            if not text.strip():
                st.error("Could not extract any text from this file.")
                st.stop()
            index, chunks = build_and_save_index(text, doc_hash)
            st.success(f"✅ Indexed {len(chunks)} chunks.")

    # Store index & chunks in session state for the chat loop
    st.session_state.index = index
    st.session_state.chunks = chunks

# ── Chat interface ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks_used" in msg:
            with st.expander("📄 Retrieved chunks"):
                for i, c in enumerate(msg["chunks_used"], 1):
                    st.markdown(f"**Chunk {c['chunk_index']}** (score: {c['score']:.3f})")
                    st.text(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
                    st.divider()

# Chat input
question = st.chat_input("Ask a question about your document…")

if question:
    # Guard: need a document loaded
    if "index" not in st.session_state:
        st.warning("Please upload a document first.")
        st.stop()

    if not groq_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks
    results = retrieve(
        query=question,
        index=st.session_state.index,
        chunks=st.session_state.chunks,
        top_k=5,
    )

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Chunk {r['chunk_index']}]: {r['text']}" for r in results
    )

    # Query LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = ask_llm(context=context, question=question)
        st.markdown(answer)

        # Show retrieved chunks for transparency
        with st.expander("📄 Retrieved chunks"):
            for i, c in enumerate(results, 1):
                st.markdown(f"**Chunk {c['chunk_index']}** (score: {c['score']:.3f})")
                st.text(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
                st.divider()

    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks_used": results,
    })
