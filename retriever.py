"""
retriever.py — Retrieve the most relevant chunks for a user query.

Embeds the query with the same local model, searches the FAISS index,
and returns the top-k chunks with their similarity scores.
Optionally filters results against a syllabus index so only
syllabus-relevant content is returned.
"""

from typing import List, Dict, Optional

import faiss
import numpy as np

from embedder import embed_query, embed_texts


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 5,
    syllabus_index: Optional[faiss.Index] = None,
    syllabus_chunks: Optional[List[Dict]] = None,
    syllabus_threshold: float = 0.3,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for the given query.

    If a syllabus_index is provided, each candidate chunk is checked
    against the syllabus — only chunks whose content scores above
    `syllabus_threshold` against at least one syllabus chunk are kept.

    Returns a list of dicts with keys: text, chunk_index, score,
    and optionally syllabus_topic.
    """
    # Embed and normalise the query vector
    q_vec = embed_query(query)
    faiss.normalize_L2(q_vec)

    # Retrieve more candidates when syllabus filtering is active
    fetch_k = top_k * 3 if syllabus_index is not None else top_k
    scores, indices = index.search(q_vec, min(fetch_k, index.ntotal))

    candidates: List[Dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({
            "text": chunks[idx]["text"],
            "chunk_index": chunks[idx]["chunk_index"],
            "score": float(score),
        })

    # ── Syllabus relevance filter ───────────────────────────────────
    if syllabus_index is not None and syllabus_chunks:
        filtered: List[Dict] = []
        for c in candidates:
            # Embed the chunk text and compare against syllabus
            c_vec = embed_texts([c["text"]])
            faiss.normalize_L2(c_vec)
            s_scores, s_ids = syllabus_index.search(c_vec, 1)
            best_score = float(s_scores[0][0])
            best_id = int(s_ids[0][0])
            if best_score >= syllabus_threshold:
                c["syllabus_topic"] = syllabus_chunks[best_id]["text"][:120]
                c["syllabus_score"] = best_score
                filtered.append(c)
        candidates = filtered

    return candidates[:top_k]
