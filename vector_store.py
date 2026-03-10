"""
vector_store.py — FAISS index management with disk persistence.

The index and its associated chunk metadata are saved under a
`faiss_store/` directory so they survive between Streamlit runs.
Each uploaded file is identified by its SHA-256 hash; if the same
file is uploaded again the existing index is reused.
"""

import json
import os
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from embedder import embed_texts
from utils import chunk_text, file_hash

# Directory where FAISS indices are persisted.
STORE_DIR = Path("faiss_store")


def _doc_dir(doc_hash: str) -> Path:
    """Return the per-document storage directory."""
    return STORE_DIR / doc_hash


def index_exists(doc_hash: str) -> bool:
    """Check whether a persisted index already exists for this document."""
    d = _doc_dir(doc_hash)
    return (d / "index.faiss").exists() and (d / "chunks.json").exists()


def build_and_save_index(
    text: str,
    doc_hash: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> tuple[faiss.Index, List[Dict]]:
    """
    Chunk the document text, embed the chunks, build a FAISS index,
    and persist everything to disk.

    Returns the FAISS index and the list of chunk dicts.
    """
    # 1. Chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # 2. Embed
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # 3. Build FAISS index (Inner Product after L2-normalisation = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 4. Persist
    d = _doc_dir(doc_hash)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(d / "index.faiss"))
    with open(d / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    return index, chunks


def load_index(doc_hash: str) -> tuple[faiss.Index, List[Dict]]:
    """Load a previously persisted FAISS index and its chunk metadata."""
    d = _doc_dir(doc_hash)
    index = faiss.read_index(str(d / "index.faiss"))
    with open(d / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
