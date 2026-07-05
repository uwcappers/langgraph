"""The citation graph: a directed NetworkX graph of papers.

Nodes  = papers (seed papers we ingested + frontier candidates they cite).
Edges  = "A cites B" (directed from citing paper to referenced paper).

This graph answers "how do these papers connect?" and drives frontier
expansion: papers cited by *several* of your seed papers are, by definition,
the shared foundation of your topic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..config import GRAPH_PATH
from ..models import Paper


def build_graph(papers: list[Paper]) -> nx.DiGraph:
    """Construct a citation DiGraph from a list of papers.

    Seed papers become fully-attributed nodes. Their references become nodes too
    (marked is_seed=False) so shared citations are visible even before we ingest
    the cited papers themselves.
    """
    g = nx.DiGraph()
    seed_ids = {p.source_id for p in papers if p.is_seed}

    for p in papers:
        _add_paper_node(g, p)
        for ref in p.references:
            if not ref.paper_id:
                continue  # need a stable id to form a reliable edge
            if ref.paper_id not in g:
                g.add_node(
                    ref.paper_id,
                    title=ref.title,
                    year=ref.year or 0,
                    is_seed=False,
                    citation_count=0,
                    doi=ref.doi or "",
                    url="",
                )
            g.add_edge(p.source_id, ref.paper_id)

    # Mark which reference nodes are actually seeds (in case order matters).
    for sid in seed_ids:
        if sid in g:
            g.nodes[sid]["is_seed"] = True
    return g


def _add_paper_node(g: nx.DiGraph, p: Paper) -> None:
    g.add_node(
        p.source_id,
        title=p.title,
        year=p.year or 0,
        is_seed=p.is_seed,
        citation_count=p.citation_count,
        doi=p.doi or "",
        url=p.url or "",
        concepts=";".join(p.concepts),
    )


@dataclass
class FrontierCandidate:
    paper_id: str
    title: str
    year: int
    cited_by_seeds: int  # how many of our seed papers cite it
    citation_count: int  # global citations (impact)

    @property
    def score(self) -> float:
        # Shared foundation first, break ties by global impact.
        return self.cited_by_seeds * 1000 + self.citation_count


def frontier_candidates(g: nx.DiGraph, top: int = 20) -> list[FrontierCandidate]:
    """Rank not-yet-ingested papers by how central they are to your seeds.

    A paper cited by 3 of your 5 seeds is a stronger foundation than one cited
    by a single seed, regardless of global fame.
    """
    seed_ids = {n for n, d in g.nodes(data=True) if d.get("is_seed")}
    out: list[FrontierCandidate] = []
    for node, data in g.nodes(data=True):
        if data.get("is_seed"):
            continue
        citing_seeds = sum(1 for pred in g.predecessors(node) if pred in seed_ids)
        if citing_seeds == 0:
            continue
        out.append(
            FrontierCandidate(
                paper_id=node,
                title=data.get("title", "?"),
                year=int(data.get("year", 0)),
                cited_by_seeds=citing_seeds,
                citation_count=int(data.get("citation_count", 0)),
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top]


def save_graph(g: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(g, path)


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    if not path.exists():
        return nx.DiGraph()
    return nx.read_gexf(path)
