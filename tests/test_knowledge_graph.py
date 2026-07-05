"""Tests for the heterogeneous knowledge graph: typed edges, grounding gaps,
hierarchy, frontier exclusion, and persistence."""

from capstone_kg.graph.citation_graph import (
    build_graph,
    concept_coverage,
    frontier_candidates,
    graph_summary,
    grounding_gaps,
    load_graph,
    save_graph,
)
from capstone_kg.models import Component, Foundation, Paper, PaperRef


def _corpus():
    return [
        Paper(source_id="P1", title="Capacitive glove", is_seed=True,
              concepts=["transduction:capacitive", "application:force-sensing"],
              references=[PaperRef(paper_id="R1", title="Shared ref")]),
        Paper(source_id="P2", title="Hall glove", is_seed=True,
              concepts=["transduction:magnetic"],
              references=[PaperRef(paper_id="R1", title="Shared ref")]),
        Paper(source_id="P3", title="FBG glove", is_seed=True,
              concepts=["transduction:optical:fbg"]),
        Foundation(source_id="F1", title="Capacitance", concepts=["transduction:capacitive"]),
        Foundation(source_id="F2", title="Optical sensing", concepts=["transduction:optical"]),
        Component(source_id="C1", title="A1324", transduction="transduction:magnetic",
                  concepts=["transduction:magnetic"]),
    ]


def test_typed_edges():
    g = build_graph(_corpus())
    assert g.edges["P1", "transduction:capacitive"]["etype"] == "uses"
    assert g.edges["F1", "transduction:capacitive"]["etype"] == "explains"
    assert g.edges["C1", "transduction:magnetic"]["etype"] == "implements"
    assert g.edges["P1", "R1"]["etype"] == "cites"


def test_node_kinds():
    g = build_graph(_corpus())
    assert g.nodes["P1"]["kind"] == "paper"
    assert g.nodes["F1"]["kind"] == "foundation"
    assert g.nodes["C1"]["kind"] == "component"
    assert g.nodes["transduction:capacitive"]["kind"] == "concept"


def test_grounding_gaps():
    g = build_graph(_corpus())
    gap_ids = {c.concept_id for c in grounding_gaps(g)}
    # force-sensing is used by P1 but no foundation/component grounds it -> gap.
    assert "application:force-sensing" in gap_ids
    # capacitive is used (P1) AND explained (F1) -> grounded, not a gap.
    assert "transduction:capacitive" not in gap_ids
    # magnetic is used (P2) AND implemented (C1) -> grounded, not a gap.
    assert "transduction:magnetic" not in gap_ids


def test_concept_coverage_counts():
    cov = {c.concept_id: c for c in concept_coverage(build_graph(_corpus()))}
    cap = cov["transduction:capacitive"]
    assert (cap.used_by, cap.explained_by, cap.implemented_by) == (1, 1, 0)
    assert cap.is_grounded and not cap.is_gap


def test_hierarchy_edge():
    g = build_graph(_corpus())
    # fbg is used and optical is explained -> both present -> subconcept_of edge exists.
    assert g.edges["transduction:optical:fbg", "transduction:optical"]["etype"] == "subconcept_of"


def test_frontier_is_papers_only():
    g = build_graph(_corpus())
    cands = frontier_candidates(g)
    ids = {c.paper_id for c in cands}
    assert ids == {"R1"}  # the shared reference; no concept/foundation/component nodes
    assert next(c for c in cands if c.paper_id == "R1").cited_by_seeds == 2


def test_graph_summary():
    s = graph_summary(build_graph(_corpus()))
    assert s["nodes"]["paper"] == 4  # P1,P2,P3 + reference R1
    assert s["nodes"]["foundation"] == 2
    assert s["nodes"]["component"] == 1
    assert s["edges"]["uses"] >= 3 and s["edges"]["cites"] == 2


def test_gexf_roundtrip_preserves_types(tmp_path):
    g = build_graph(_corpus())
    p = tmp_path / "g.gexf"
    save_graph(g, p)
    g2 = load_graph(p)
    assert g2.nodes["transduction:capacitive"]["kind"] == "concept"
    assert g2.edges["F1", "transduction:capacitive"]["etype"] == "explains"
    assert g2.number_of_nodes() == g.number_of_nodes()
