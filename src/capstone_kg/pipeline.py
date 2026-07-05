"""End-to-end ingestion orchestration.

    PDF -> text + title
        -> Semantic Scholar match (canonical metadata + references)
        -> concept extraction (optional, needs Claude)
        -> chunk + embed  (vector store)
        -> corpus + citation graph

Everything is idempotent and cached, so re-running after adding a paper is cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .config import PAPERS_DIR, ensure_dirs
from .corpus import load_corpus, save_corpus, upsert
from .enrich.semantic_scholar import SemanticScholarClient, raw_to_paper
from .graph.citation_graph import build_graph, save_graph
from .graph.concepts import extract_concepts
from .ingest.chunker import chunk_source
from .ingest.pdf_parser import parse_pdf
from .models import Paper
from .store.vectorstore import VectorStore

console = Console()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


@dataclass
class IngestResult:
    ingested: list[Paper]
    matched: int
    unmatched: int


def ingest_papers(
    papers_dir: Path = PAPERS_DIR,
    extract_concepts_flag: bool = True,
    rebuild: bool = False,
) -> IngestResult:
    """Ingest every PDF in `papers_dir` and (re)build all artifacts."""
    ensure_dirs()
    pdfs = sorted(papers_dir.glob("*.pdf")) + sorted(papers_dir.glob("*.PDF"))
    if not pdfs:
        console.print(f"[yellow]No PDFs found in {papers_dir}.[/] Drop your papers there.")
        return IngestResult([], 0, 0)

    corpus = [] if rebuild else load_corpus()
    store = VectorStore() if rebuild else VectorStore.load()
    already = store.source_ids()

    s2 = SemanticScholarClient()
    matched = unmatched = 0
    ingested: list[Paper] = []

    try:
        for pdf in pdfs:
            console.print(f"[bold]Parsing[/] {pdf.name}")
            parsed = parse_pdf(pdf)

            paper = _resolve_paper(s2, parsed.title_guess, pdf.name)
            if paper.source_id.startswith("local:"):
                unmatched += 1
                console.print(f"  [yellow]No Semantic Scholar match[/] — indexed text only.")
            else:
                matched += 1
                console.print(
                    f"  [green]Matched[/] {paper.short()}  "
                    f"({paper.reference_count} refs, {paper.citation_count} citations)"
                )

            if extract_concepts_flag:
                paper.concepts = extract_concepts(paper)
                if paper.concepts:
                    console.print(f"  concepts: {', '.join(paper.concepts)}")

            # Embed chunks (skip if this paper is already in the store).
            if paper.source_id not in already:
                chunks = chunk_source(paper.source_id, paper.title, parsed.full_text)
                store.add_chunks(chunks)
                console.print(f"  embedded {len(chunks)} chunks")

            corpus = upsert(corpus, paper)
            ingested.append(paper)
    finally:
        s2.close()

    # (Re)build graph from the full corpus and persist everything.
    g = build_graph(corpus)
    save_graph(g)
    store.save()
    save_corpus(corpus)

    console.print(
        f"\n[bold green]Done.[/] {len(corpus)} papers in corpus, "
        f"{g.number_of_nodes()} graph nodes, {g.number_of_edges()} citation edges."
    )
    return IngestResult(ingested, matched, unmatched)


def _resolve_paper(s2: SemanticScholarClient, title: str, filename: str) -> Paper:
    """Match a PDF to Semantic Scholar; fall back to a local-only paper."""
    match = s2.match_paper(title)
    if match and match.get("paperId"):
        full = s2.get_paper(match["paperId"]) or match
        return raw_to_paper(full, is_seed=True, source_file=filename)
    # Fallback: no external metadata, but still searchable by content.
    return Paper(
        source_id=f"local:{_slug(Path(filename).stem)}",
        title=title,
        source_file=filename,
        is_seed=True,
    )
