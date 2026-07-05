"""Command-line interface.

    capstone ingest              # parse papers/, link citations, build the graph
    capstone ask "question"      # query the corpus (GraphRAG over the concept bridge)
    capstone frontier            # papers your seeds cite most — what to add next
    capstone concepts            # concept coverage: what's grounded, what isn't
    capstone gaps                # concepts your papers use but nothing grounds
    capstone specs               # component datasheet specs side by side
    capstone status              # what's in the corpus / graph, by kind
    capstone viz                 # export an interactive graph (needs [viz] extra)
"""

from __future__ import annotations

from collections import Counter

import typer
from rich.console import Console
from rich.table import Table

from .corpus import load_corpus
from .graph.citation_graph import (
    concept_coverage,
    frontier_candidates,
    graph_summary,
    grounding_gaps,
    load_graph,
)
from .models import Component
from .pipeline import ingest_papers

app = typer.Typer(add_completion=False, help="Capstone knowledge graph over research papers.")
console = Console()


@app.command()
def ingest(
    no_concepts: bool = typer.Option(False, help="skip LLM concept extraction."),
    rebuild: bool = typer.Option(False, help="Rebuild from scratch, ignoring cached artifacts."),
):
    """Ingest every PDF under papers/ and (re)build the graph + vector store."""
    ingest_papers(extract_concepts_flag=not no_concepts, rebuild=rebuild)


@app.command()
def ask(
    question: str = typer.Argument(..., help="your question."),
    k: int = typer.Option(6, help="Number of chunks to retrieve."),
):
    """Ask a question; get a source-grounded answer via the LangGraph pipeline."""
    from .agent.query_graph import answer_question

    result = answer_question(question, top_k=k)
    console.print()
    console.print(result.get("answer", "(no answer)"))


@app.command()
def frontier(top: int = typer.Option(20, help="how many candidates to show.")):
    """Rank not-yet-ingested papers by how central they are to your seeds."""
    cands = frontier_candidates(load_graph(), top=top)
    if not cands:
        console.print("[yellow]No frontier candidates yet.[/] Ingest papers with matches first.")
        return
    table = Table(title="Frontier candidates (add these next)")
    table.add_column("#", justify="right")
    table.add_column("Cited by seeds", justify="right")
    table.add_column("Global cites", justify="right")
    table.add_column("Year", justify="right")
    table.add_column("Title")
    for i, c in enumerate(cands, 1):
        table.add_row(str(i), str(c.cited_by_seeds), str(c.citation_count),
                      str(c.year or "?"), c.title)
    console.print(table)


@app.command()
def concepts(
    category: str = typer.Option("", help="filter by category: transduction|mechanism|application.")
):
    """Concept coverage — who uses / explains / implements each concept."""
    cov = concept_coverage(load_graph())
    if category:
        cov = [c for c in cov if c.category == category]
    if not cov:
        console.print("[yellow]No concepts yet.[/] Ingest sources with concept extraction on.")
        return
    table = Table(title="Concept coverage")
    table.add_column("Category")
    table.add_column("Concept")
    table.add_column("Used", justify="right")       # papers
    table.add_column("Explained", justify="right")  # foundations
    table.add_column("Implemented", justify="right")  # components
    table.add_column("Grounded?")
    for c in cov:
        state = "[red]GAP[/]" if c.is_gap else ("[green]✓[/]" if c.is_grounded else "—")
        table.add_row(c.category, c.label, str(c.used_by), str(c.explained_by),
                      str(c.implemented_by), state)
    console.print(table)
    n_gaps = sum(1 for c in cov if c.is_gap)
    if n_gaps:
        console.print(f"[red]{n_gaps} grounding gap(s)[/] — see `capstone gaps`.")


@app.command()
def gaps():
    """Concepts your papers rely on but no foundation/datasheet grounds (black boxes)."""
    gg = grounding_gaps(load_graph())
    if not gg:
        console.print("[green]No grounding gaps[/] — every concept your papers use is "
                      "backed by a foundation or component.")
        return
    table = Table(title="Grounding gaps — add a textbook section or datasheet for these")
    table.add_column("Concept")
    table.add_column("Category")
    table.add_column("Used by (papers)", justify="right")
    for c in gg:
        table.add_row(c.label, c.category, str(c.used_by))
    console.print(table)


@app.command()
def specs():
    """Component datasheet specs, side by side — real noise floors and bandwidths."""
    comps = [s for s in load_corpus() if isinstance(s, Component)]
    if not comps:
        console.print("[yellow]No components yet.[/] Drop datasheets in papers/datasheets/.")
        return

    def raw(spec) -> str:
        return spec.raw if spec else "—"

    table = Table(title="Component specs")
    for col in ("Part", "Transduction", "Sensitivity", "Noise floor",
                "Bandwidth", "Tempco", "Supply", "Output"):
        table.add_column(col)
    for c in comps:
        s = c.specs
        table.add_row(
            c.part_number or c.title, (c.transduction or "—").replace("transduction:", ""),
            raw(s.sensitivity), raw(s.noise_floor), raw(s.bandwidth),
            raw(s.temperature_coefficient), raw(s.supply_voltage), s.output_type or "—",
        )
    console.print(table)


@app.command()
def status():
    """Show what's in the corpus and graph, by kind."""
    corpus = load_corpus()
    g = load_graph()
    seeds = [p for p in corpus if p.kind == "paper" and getattr(p, "is_seed", False)]
    if seeds:
        table = Table(title="Seed papers")
        table.add_column("Paper")
        table.add_column("Year", justify="right")
        table.add_column("Refs", justify="right")
        table.add_column("Concepts")
        for p in seeds:
            table.add_row(p.title[:70], str(p.year or "?"),
                          str(getattr(p, "reference_count", 0)), ", ".join(p.concepts[:4]))
        console.print(table)

    kc = Counter(s.kind for s in corpus)
    summary = graph_summary(g)
    console.print(
        f"\ncorpus: {kc['paper']} papers, {kc['foundation']} foundation sections, "
        f"{kc['component']} components"
    )
    console.print(f"graph nodes: {summary['nodes']}")
    console.print(f"graph edges: {summary['edges']}")


@app.command()
def viz(output: str = typer.Option("data/graph.html", help="Output HTML path.")):
    """Export an interactive HTML visualization of the knowledge graph."""
    from .graph.visualize import export_html

    console.print(f"[green]Wrote[/] {export_html(output)}")


if __name__ == "__main__":
    app()
