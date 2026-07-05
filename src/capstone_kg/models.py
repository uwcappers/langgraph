"""Pydantic data models shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Author(BaseModel):
    name: str
    author_id: str | None = None  # Semantic Scholar author id


class PaperRef(BaseModel):
    """A lightweight reference to another paper (a citation edge target)."""

    paper_id: str | None = None  # Semantic Scholar id, if resolved
    title: str
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class Paper(BaseModel):
    """A paper in the corpus. `paper_id` is the Semantic Scholar id when resolved,
    otherwise a stable slug derived from the filename."""

    paper_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[Author] = Field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None

    # Provenance
    source_pdf: str | None = None  # filename in papers/, if ingested locally
    is_seed: bool = True  # True = a paper we ingested; False = a frontier candidate

    # Citation edges (Semantic Scholar ids where possible)
    references: list[PaperRef] = Field(default_factory=list)  # papers this one cites
    citation_count: int = 0
    reference_count: int = 0

    # LLM-extracted concepts (populated by the concept-extraction step)
    concepts: list[str] = Field(default_factory=list)

    def short(self) -> str:
        who = self.authors[0].name.split()[-1] if self.authors else "?"
        return f"{who} et al. ({self.year or 'n.d.'}) — {self.title}"


class Chunk(BaseModel):
    """A chunk of a paper's text, ready for embedding."""

    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    section: str | None = None
