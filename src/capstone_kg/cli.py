"""Command-line interface.

    capstone ingest              # parse papers/, link citations, build the graph
    capstone ask "question"      # query the corpus (GraphRAG)
    capstone frontier            # papers your seeds cite most — what to add next
    capstone status              # what's in the corpus / graph
    capstone viz                 # export an interactive graph (needs [viz] extra)
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from .corpus import load_corpus
from .graph.citation_graph import frontier_candidates, load_graph
from .pipeline import ingest_papers

app = typer.Typer(add_completion=False, help="Capstone knowledge graph over research papers.")
console = Console()


@app.command()
def ingest(
    no_concepts: bool = typer.Option(False, help="skip LLM concept extraction."),
    rebuild: bool = typer.Option(False, help="Rebuild from scratch, ignoring cached artifacts."),
):
    """Ingest every PDF in papers/ and (re)build the graph + vector store."""
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
    g = load_graph()
    cands = frontier_candidates(g, top=top)
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
def status():
    """Show what's currently in the corpus and graph."""
    corpus = load_corpus()
    g = load_graph()
    seeds = [p for p in corpus if p.is_seed]
    table = Table(title="Corpus")
    table.add_column("Paper")
    table.add_column("Year", justify="right")
    table.add_column("Refs", justify="right")
    table.add_column("Concepts")
    for p in seeds:
        table.add_row(p.title[:70], str(p.year or "?"), str(p.reference_count),
                      ", ".join(p.concepts[:4]))
    console.print(table)
    console.print(
        f"\n{len(seeds)} seed papers · {g.number_of_nodes()} graph nodes · "
        f"{g.number_of_edges()} citation edges"
    )


@app.command()
def viz(output: str = typer.Option("data/graph.html", help="Output HTML path.")):
    """Export an interactive HTML visualization of the citation graph."""
    from .graph.visualize import export_html

    path = export_html(output)
    console.print(f"[green]Wrote[/] {path}")


if __name__ == "__main__":
    app()
