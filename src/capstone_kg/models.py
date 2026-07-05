"""Pydantic data models shared across the pipeline.

The corpus is heterogeneous: every ingested thing is a `Source` with a `kind`
discriminator. Three concrete kinds exist today:

    paper       — an academic paper (Semantic Scholar linked)
    foundation  — a section of a reference text / textbook (e.g. Fraden §8.2)
    component   — a real part + its datasheet specs (e.g. a Hall-effect IC)

`Concept` is not a Source — it is derived (not ingested from a file) and forms the
bridge layer that connects the three kinds (see docs/sensor-knowledge-layer.md).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Source hierarchy
# ---------------------------------------------------------------------------
class Source(BaseModel):
    """Base for anything ingested into the corpus and made retrievable.

    `source_id` is the unique node id used across the vector store and graph:
      - papers      -> Semantic Scholar id, or "local:<slug>" when unmatched
      - foundations -> "foundation:<slug>[:<section>]"
      - components  -> "component:<slug>"
    """

    source_id: str
    kind: Literal["paper", "foundation", "component"]
    title: str
    source_file: str | None = None  # filename under papers/, if ingested from disk
    concepts: list[str] = Field(default_factory=list)  # canonical concept slugs


class Paper(Source):
    kind: Literal["paper"] = "paper"

    abstract: str | None = None
    year: int | None = None
    authors: list[Author] = Field(default_factory=list)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None

    is_seed: bool = True  # True = a paper we ingested; False = a frontier candidate
    references: list[PaperRef] = Field(default_factory=list)  # papers this one cites
    citation_count: int = 0
    reference_count: int = 0

    def short(self) -> str:
        who = self.authors[0].name.split()[-1] if self.authors else "?"
        return f"{who} et al. ({self.year or 'n.d.'}) — {self.title}"


class Foundation(Source):
    """A tightly-scoped section of a reference text (not a whole book), so the
    `explains` edge points at the right few pages."""

    kind: Literal["foundation"] = "foundation"

    edition: str | None = None
    section_path: str | None = None  # e.g. "Fraden §8.2 Linear Capacitive Sensors"


class Spec(BaseModel):
    """A single datasheet quantity: parsed value+unit for comparison, plus the
    original string so nothing is lost to imperfect parsing."""

    raw: str
    value: float | None = None
    unit: str | None = None


class ComponentSpecs(BaseModel):
    """Structured datasheet specs — the 'real noise floor, not abstracted high
    resolution' numbers. All optional; extraction fills what it finds."""

    measurement_range: Spec | None = None
    sensitivity: Spec | None = None
    noise_floor: Spec | None = None
    bandwidth: Spec | None = None
    temperature_coefficient: Spec | None = None
    supply_voltage: Spec | None = None
    output_type: str | None = None  # e.g. "analog", "I2C", "SPI"


class Component(Source):
    """A real part and its datasheet."""

    kind: Literal["component"] = "component"

    part_number: str | None = None
    manufacturer: str | None = None
    transduction: str | None = None  # canonical transduction concept slug
    specs: ComponentSpecs = Field(default_factory=ComponentSpecs)


# Discriminated union for (de)serialization of a mixed corpus.
AnySource = Annotated[Union[Paper, Foundation, Component], Field(discriminator="kind")]
SourceListAdapter: TypeAdapter[list[AnySource]] = TypeAdapter(list[AnySource])


# ---------------------------------------------------------------------------
# Concept — the bridge layer (derived, not a Source)
# ---------------------------------------------------------------------------
class Concept(BaseModel):
    """A transduction principle / technical concept. Hierarchy-capable but grown,
    not designed: most concepts start with parent_id=None and are split into
    children only when the corpus forces the distinction."""

    concept_id: str  # hierarchical slug, e.g. "transduction:optical" or "...:fbg"
    label: str
    category: Literal["transduction", "mechanism", "application"]
    parent_id: str | None = None  # null by design for most nodes
    aliases: list[str] = Field(default_factory=list)  # raw phrases mapped here


# ---------------------------------------------------------------------------
# Retrieval chunk
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """A chunk of a source's text, ready for embedding."""

    chunk_id: str
    source_id: str
    source_title: str
    source_kind: str = "paper"  # paper | foundation | component
    text: str
    section: str | None = None
