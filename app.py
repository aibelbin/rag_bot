"""
app.py — Streamlit RAG Study Assistant

Upload notes (PDF/TXT) and an optional syllabus. Ask questions and
get answers strictly grounded in the notes, filtered to only cover
topics within the syllabus.

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
st.caption("Upload your notes and syllabus — answers come only from your notes, scoped to your syllabus.")

# ── Sidebar: upload Notes + Syllabus ────────────────────────────────
with st.sidebar:
    st.header("📝 Notes")
    uploaded_notes = st.file_uploader(
        "Upload study notes",
        type=["pdf", "txt"],
        key="notes_uploader",
        help="Your study material (PDF or TXT).",
    )

    st.divider()

    st.header("📋 Syllabus")
    uploaded_syllabus = st.file_uploader(
        "Upload syllabus",
        type=["pdf", "txt"],
        key="syllabus_uploader",
        help="Optional — limits answers to syllabus topics only.",
    )

# ── Process notes ───────────────────────────────────────────────────
if uploaded_notes is not None:
    notes_bytes = uploaded_notes.read()
    notes_hash = file_hash(notes_bytes)

    if "notes_hash" not in st.session_state or st.session_state.notes_hash != notes_hash:
        st.session_state.notes_hash = notes_hash
        st.session_state.messages = []

    if index_exists(notes_hash):
        with st.sidebar:
            st.success("✅ Notes index loaded from cache.")
        notes_index, notes_chunks = load_index(notes_hash)
    else:
        with st.sidebar, st.spinner("Processing notes — extracting, chunking, embedding…"):
            text = extract_text(uploaded_notes.name, notes_bytes)
            if not text.strip():
                st.error("Could not extract any text from the notes file.")
                st.stop()
            notes_index, notes_chunks = build_and_save_index(text, notes_hash)
            st.success(f"✅ Notes indexed ({len(notes_chunks)} chunks).")

    st.session_state.notes_index = notes_index
    st.session_state.notes_chunks = notes_chunks

# ── Process syllabus ────────────────────────────────────────────────
if uploaded_syllabus is not None:
    syl_bytes = uploaded_syllabus.read()
    syl_hash = file_hash(syl_bytes)

    if "syl_hash" not in st.session_state or st.session_state.syl_hash != syl_hash:
        st.session_state.syl_hash = syl_hash

    if index_exists(syl_hash):
        with st.sidebar:
            st.success("✅ Syllabus index loaded from cache.")
        syl_index, syl_chunks = load_index(syl_hash)
    else:
        with st.sidebar, st.spinner("Processing syllabus…"):
            syl_text = extract_text(uploaded_syllabus.name, syl_bytes)
            if not syl_text.strip():
                st.error("Could not extract any text from the syllabus file.")
                st.stop()
            # Smaller chunks for syllabus — topics are usually short
            syl_index, syl_chunks = build_and_save_index(
                syl_text, syl_hash, chunk_size=200, overlap=40
            )
            st.success(f"✅ Syllabus indexed ({len(syl_chunks)} chunks).")

    st.session_state.syl_index = syl_index
    st.session_state.syl_chunks = syl_chunks
else:
    # Clear syllabus from session if removed
    st.session_state.pop("syl_index", None)
    st.session_state.pop("syl_chunks", None)

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
                    label = f"**Chunk {c['chunk_index']}** (score: {c['score']:.3f})"
                    if "syllabus_topic" in c:
                        label += f"  ·  📋 _{c['syllabus_topic']}…_"
                    st.markdown(label)
                    st.text(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
                    st.divider()

# Chat input
question = st.chat_input("Ask a question about your document…")

if question:
    if "notes_index" not in st.session_state:
        st.warning("Please upload your study notes first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Determine if syllabus filtering is active
    syl_index = st.session_state.get("syl_index")
    syl_chunks = st.session_state.get("syl_chunks")

    # Retrieve relevant chunks (with optional syllabus filtering)
    results = retrieve(
        query=question,
        index=st.session_state.notes_index,
        chunks=st.session_state.notes_chunks,
        top_k=5,
        syllabus_index=syl_index,
        syllabus_chunks=syl_chunks,
    )

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join(
        f"[Chunk {r['chunk_index']}]: {r['text']}" for r in results
    )

    # Build syllabus topics string for the LLM prompt
    syllabus_topics = ""
    if syl_chunks:
        syllabus_topics = "\n".join(
            f"- {c['text'][:200]}" for c in syl_chunks
        )

    # Query LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = ask_llm(
                context=context,
                question=question,
                syllabus_topics=syllabus_topics,
            )
        st.markdown(answer)

        # Show retrieved chunks for transparency
        with st.expander("📄 Retrieved chunks"):
            for i, c in enumerate(results, 1):
                label = f"**Chunk {c['chunk_index']}** (score: {c['score']:.3f})"
                if "syllabus_topic" in c:
                    label += f"  ·  📋 _{c['syllabus_topic']}…_"
                st.markdown(label)
                st.text(c["text"][:500] + ("…" if len(c["text"]) > 500 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks_used": results,
    })
