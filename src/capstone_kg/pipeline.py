"""End-to-end ingestion orchestration.

Three source kinds are routed by folder (see papers/README.md):

    papers/               paper       -> Semantic Scholar match (metadata + citations)
    papers/foundations/   foundation  -> split into TOC sections
    papers/datasheets/    component   -> structured datasheet specs

Every source is then: concept-extracted (vocab-aware) -> chunked + embedded ->
written to the corpus. The citation graph is (for now) still built from papers only;
foundation/component/concept nodes + edges arrive in increment 5.

Everything is idempotent and cached, so re-running after adding a source is cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from .config import PAPERS_DIR, ensure_dirs
from .corpus import load_corpus, save_corpus, upsert
from .enrich.semantic_scholar import SemanticScholarClient, raw_to_paper
from .graph.citation_graph import build_graph, graph_summary, save_graph
from .graph.concepts import extract_concepts
from .ingest.chunker import chunk_source
from .ingest.datasheet import extract_component
from .ingest.foundation import split_pdf_sections
from .ingest.pdf_parser import parse_pdf
from .models import Foundation, Paper, Source
from .store.vectorstore import VectorStore

console = Console()

FOUNDATIONS_SUBDIR = "foundations"
DATASHEETS_SUBDIR = "datasheets"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


@dataclass
class _Unit:
    """One thing to embed: a source, the text to chunk, and the text to mine for
    concepts (which differ for papers — chunk full text, mine the abstract)."""

    source: Source
    chunk_text: str
    concept_text: str | None  # None => use the source's own text (abstract for papers)


@dataclass
class IngestResult:
    ingested: list[Source] = field(default_factory=list)
    matched: int = 0
    unmatched: int = 0
    foundations: int = 0
    components: int = 0


def scan_sources(papers_dir: Path = PAPERS_DIR) -> list[tuple[Path, str]]:
    """Discover PDFs and tag each with its kind, based on the folder."""

    def pdfs(d: Path) -> list[Path]:
        return sorted(d.glob("*.pdf")) + sorted(d.glob("*.PDF")) if d.exists() else []

    jobs = [(p, "paper") for p in pdfs(papers_dir)]
    jobs += [(p, "foundation") for p in pdfs(papers_dir / FOUNDATIONS_SUBDIR)]
    jobs += [(p, "component") for p in pdfs(papers_dir / DATASHEETS_SUBDIR)]
    return jobs


def ingest_papers(
    papers_dir: Path = PAPERS_DIR,
    extract_concepts_flag: bool = True,
    rebuild: bool = False,
) -> IngestResult:
    """Ingest every PDF under `papers_dir` and (re)build all artifacts."""
    ensure_dirs()
    jobs = scan_sources(papers_dir)
    if not jobs:
        console.print(
            f"[yellow]No PDFs found under {papers_dir}.[/] "
            "Drop papers in papers/, textbooks in papers/foundations/, "
            "datasheets in papers/datasheets/."
        )
        return IngestResult()

    corpus: list[Source] = [] if rebuild else list(load_corpus())
    store = VectorStore() if rebuild else VectorStore.load()
    already = store.source_ids()

    result = IngestResult()
    s2 = SemanticScholarClient()
    try:
        for path, kind in jobs:
            console.print(f"[bold]{kind.capitalize()}[/] {path.name}")
            if kind == "paper":
                units = _units_for_paper(s2, path, result)
            elif kind == "foundation":
                units = _units_for_foundation(path)
                result.foundations += len(units)
            else:
                units = _units_for_component(path)
                result.components += len(units)

            for unit in units:
                src = unit.source
                if extract_concepts_flag:
                    found = extract_concepts(src, text=unit.concept_text)
                    src.concepts = _dedupe(src.concepts + found)
                    if src.concepts:
                        console.print(f"  concepts: {', '.join(src.concepts)}")
                if src.source_id not in already:
                    chunks = chunk_source(src.source_id, src.title, unit.chunk_text, src.kind)
                    store.add_chunks(chunks)
                    already.add(src.source_id)
                    console.print(f"  embedded {len(chunks)} chunks")
                corpus = upsert(corpus, src)
                result.ingested.append(src)
    finally:
        s2.close()

    # The heterogeneous knowledge graph: papers + foundations + components + concepts.
    g = build_graph(corpus)
    save_graph(g)
    store.save()
    save_corpus(corpus)

    summary = graph_summary(g)
    nodes, edges = summary["nodes"], summary["edges"]
    console.print(
        f"\n[bold green]Done.[/] nodes: {_fmt(nodes)} · edges: {_fmt(edges)}"
    )
    return result


# ---- per-kind ingestion -----------------------------------------------------
def _units_for_paper(s2: SemanticScholarClient, path: Path, result: IngestResult) -> list[_Unit]:
    parsed = parse_pdf(path)
    paper = _resolve_paper(s2, parsed.title_guess, path.name)
    if paper.source_id.startswith("local:"):
        result.unmatched += 1
        console.print("  [yellow]No Semantic Scholar match[/] — indexed text only.")
    else:
        result.matched += 1
        console.print(
            f"  [green]Matched[/] {paper.short()}  "
            f"({paper.reference_count} refs, {paper.citation_count} citations)"
        )
    # Chunk the full text; mine concepts from the abstract (concept_text=None).
    return [_Unit(source=paper, chunk_text=parsed.full_text, concept_text=None)]


def _units_for_foundation(path: Path) -> list[_Unit]:
    stem = _slug(path.stem)
    sections = split_pdf_sections(path)
    console.print(f"  {len(sections)} section(s) from TOC")
    units: list[_Unit] = []
    for idx, sec in enumerate(sections):
        section_path = f"{path.stem} — {_section_label(sec.number, sec.title)}"
        fnd = Foundation(
            source_id=f"foundation:{stem}:{idx:03d}",
            title=sec.title,
            source_file=path.name,
            section_path=section_path,
        )
        units.append(_Unit(source=fnd, chunk_text=sec.text, concept_text=sec.text))
    return units


def _units_for_component(path: Path) -> list[_Unit]:
    parsed = parse_pdf(path)
    comp = extract_component(
        parsed.full_text,
        source_id=f"component:{_slug(path.stem)}",
        title=parsed.title_guess,
        source_file=path.name,
    )
    if comp.part_number:
        console.print(f"  [green]{comp.part_number}[/] ({comp.manufacturer or 'unknown mfr'})")
    return [_Unit(source=comp, chunk_text=parsed.full_text, concept_text=parsed.full_text)]


def _resolve_paper(s2: SemanticScholarClient, title: str, filename: str) -> Paper:
    """Match a PDF to Semantic Scholar; fall back to a local-only paper."""
    match = s2.match_paper(title)
    if match and match.get("paperId"):
        full = s2.get_paper(match["paperId"]) or match
        return raw_to_paper(full, is_seed=True, source_file=filename)
    return Paper(
        source_id=f"local:{_slug(Path(filename).stem)}",
        title=title,
        source_file=filename,
        is_seed=True,
    )


def _section_label(number: str | None, title: str) -> str:
    """Human label for a foundation section, avoiding a duplicated number when the
    TOC title already leads with it (e.g. title '8.2 Linear Capacitive Sensors')."""
    if number and title.startswith(number):
        return f"§{title}"
    if number:
        return f"§{number} {title}"
    return title


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _fmt(counts: dict) -> str:
    return ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "0"
