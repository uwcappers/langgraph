"""Extract text and a best-guess title from a PDF.

We deliberately keep title extraction simple: the title is only used as a query
to Semantic Scholar's title-match endpoint, which is tolerant of noise. Once
matched, all authoritative metadata comes from the API, not the PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


@dataclass
class ParsedPdf:
    path: Path
    title_guess: str
    full_text: str
    num_pages: int


def _looks_like_title(text: str) -> bool:
    words = text.split()
    return 3 <= len(words) <= 30 and not text.lower().startswith(("abstract", "http"))


def _guess_title(doc: "fitz.Document") -> str:
    # 1) Trust embedded metadata if it looks like a real title.
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if meta_title and _looks_like_title(meta_title):
        return meta_title

    # 2) Otherwise, take the largest-font line block on page 1 (usually the title).
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    candidates: list[tuple[float, str]] = []
    for block in blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            size = max(s["size"] for s in spans)
            line_text = " ".join(s["text"] for s in spans).strip()
            if line_text and _looks_like_title(line_text):
                candidates.append((size, line_text))
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # 3) Fall back to the filename.
    return Path(doc.name).stem.replace("_", " ").replace("-", " ")


def parse_pdf(path: str | Path) -> ParsedPdf:
    path = Path(path)
    with fitz.open(path) as doc:
        title = _guess_title(doc)
        text_parts = [doc[i].get_text("text") for i in range(doc.page_count)]
        num_pages = doc.page_count
    full_text = _clean(" \n".join(text_parts))
    return ParsedPdf(path=path, title_guess=title, full_text=full_text, num_pages=num_pages)


def _clean(text: str) -> str:
    # Join hyphenated line breaks, collapse whitespace.
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
