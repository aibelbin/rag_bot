"""
utils.py — Text extraction from PDFs/TXT and chunking into sized segments.
"""

import hashlib
from pathlib import Path
from typing import List, Dict

import PyPDF2
import io


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file's bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode raw bytes of a .txt file to string."""
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(file_name: str, file_bytes: bytes) -> str:
    """Route to the correct extractor based on file extension."""
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 hex digest for the file contents (used for caching)."""
    return hashlib.sha256(file_bytes).hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[Dict]:
    """
    Split text into overlapping chunks by approximate word count.

    Each chunk targets ~chunk_size words with `overlap` words shared between
    consecutive chunks.  Returns a list of dicts with keys:
      - "text": the chunk string
      - "chunk_index": integer position
    """
    words = text.split()
    chunks: List[Dict] = []
    start = 0
    idx = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append({
            "text": " ".join(chunk_words),
            "chunk_index": idx,
        })
        idx += 1
        start += chunk_size - overlap  # slide forward with overlap
    return chunks
