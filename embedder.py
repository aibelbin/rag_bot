"""
embedder.py — Local embedding model using sentence-transformers.

Loads `all-MiniLM-L6-v2` once and exposes a function to embed
a list of strings into numpy arrays.  No external API calls.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Model is loaded once per process and reused.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into a 2-D float32 numpy array
    of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, embedding_dim)."""
    return embed_texts([query])
