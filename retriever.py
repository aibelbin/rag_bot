"""
retriever.py — Retrieve the most relevant chunks for a user query.

Embeds the query with the same local model, searches the FAISS index,
and returns the top-k chunks with their similarity scores.
"""

from typing import List, Dict

import faiss
import numpy as np

from embedder import embed_query


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for the given query.

    Returns a list of dicts, each containing:
      - "text": chunk text
      - "chunk_index": original position in the document
      - "score": cosine similarity score
    """
    # Embed and normalise the query vector
    q_vec = embed_query(query)
    faiss.normalize_L2(q_vec)

    # Search
    scores, indices = index.search(q_vec, min(top_k, index.ntotal))

    results: List[Dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": float(score),
        })
    return results
