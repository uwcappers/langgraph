"""Offline tests for citation-graph construction and frontier ranking."""

from capstone_kg.graph.citation_graph import build_graph, frontier_candidates
from capstone_kg.models import Paper, PaperRef


def _seed(pid: str, refs: list[str]) -> Paper:
    return Paper(
        source_id=pid,
        title=f"Seed {pid}",
        is_seed=True,
        references=[PaperRef(paper_id=r, title=f"Ref {r}") for r in refs],
    )


def test_edges_and_nodes():
    papers = [_seed("A", ["X", "Y"]), _seed("B", ["Y", "Z"])]
    g = build_graph(papers)
    # 2 seeds + 3 distinct references
    assert g.number_of_nodes() == 5
    assert g.has_edge("A", "X")
    assert g.has_edge("B", "Y")
    assert g.nodes["A"]["is_seed"] is True
    assert g.nodes["X"]["is_seed"] is False


def test_frontier_ranks_shared_citations_first():
    # Y is cited by both seeds; X and Z by one each -> Y ranks first.
    papers = [_seed("A", ["X", "Y"]), _seed("B", ["Y", "Z"])]
    g = build_graph(papers)
    cands = frontier_candidates(g, top=10)
    ids = [c.paper_id for c in cands]
    assert ids[0] == "Y"
    assert cands[0].cited_by_seeds == 2
    assert set(ids) == {"X", "Y", "Z"}  # seeds themselves are excluded


def test_seed_not_listed_as_frontier():
    # If a seed is also cited by another seed, it must not appear as a candidate.
    papers = [_seed("A", ["B"]), _seed("B", [])]
    g = build_graph(papers)
    cands = frontier_candidates(g)
    assert all(c.paper_id != "B" for c in cands)
