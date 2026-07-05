"""Split a reference text (textbook) into tightly-scoped sections via its TOC.

Scoping a Foundation node to a section (not a whole book) keeps the `explains` edge
precise. We use the PDF's table of contents (`get_toc`) — a cheap, reliable signal —
and fall back to the whole document when there is no TOC. No heuristic heading
detection (that is a deferred item in docs/sensor-knowledge-layer.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf

from .pdf_parser import _clean


@dataclass
class Section:
    title: str
    text: str
    number: str | None  # e.g. "8.2" parsed from the heading, if present
    level: int          # TOC depth (0 = whole-doc fallback)


def _section_number(title: str) -> str | None:
    m = re.match(r"\s*(\d+(?:\.\d+)+)", title)  # require at least one dot: "8.2"
    return m.group(1) if m else None


def split_pdf_sections(
    path: str | Path, max_level: int = 2, min_chars: int = 200
) -> list[Section]:
    """Return the sections of a reference PDF.

    `max_level` caps TOC depth (chapters+sections, not every deep subsection).
    Sections shorter than `min_chars` (front-matter, TOC pages) are dropped.
    """
    path = Path(path)
    with fitz.open(path) as doc:
        toc = doc.get_toc()  # [[level, title, page(1-based)], ...]
        n = doc.page_count
        page_text = [doc[i].get_text("text") for i in range(n)]

    if not toc:
        text = _clean("\n".join(page_text))
        return [Section(title=path.stem, text=text, number=None, level=0)]

    entries = [e for e in toc if e[0] <= max_level] or toc
    sections: list[Section] = []
    for idx, (level, title, page) in enumerate(entries):
        start = max(page - 1, 0)
        end = entries[idx + 1][2] - 1 if idx + 1 < len(entries) else n
        end = max(end, start + 1)
        text = _clean("\n".join(page_text[start:end]))
        if len(text) < min_chars:
            continue
        sections.append(
            Section(title=title.strip(), text=text, number=_section_number(title), level=level)
        )
    # If everything got filtered (e.g. a scanned book), fall back to the whole doc.
    if not sections:
        return [Section(title=path.stem, text=_clean("\n".join(page_text)), number=None, level=0)]
    return sections
