"""Split a paper's text into overlapping chunks for embedding.

A simple, dependency-free recursive splitter on paragraph / sentence / word
boundaries. Good enough for retrieval over a small corpus; swap in
langchain_text_splitters later if you want section-aware chunking.
"""

from __future__ import annotations

from ..config import get_settings
from ..models import Chunk


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Try to break on the nicest separator within the window.
            window = text[start:end]
            for sep in separators:
                idx = window.rfind(sep)
                if idx > size * 0.5:  # only break if reasonably far in
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_paper(paper_id: str, paper_title: str, text: str) -> list[Chunk]:
    settings = get_settings()
    pieces = _split_text(text, settings.chunk_size, settings.chunk_overlap)
    return [
        Chunk(
            chunk_id=f"{paper_id}::{i}",
            paper_id=paper_id,
            paper_title=paper_title,
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]
