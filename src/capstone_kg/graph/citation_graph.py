"""The knowledge graph: a heterogeneous, typed NetworkX DiGraph.

Node kinds:  paper | foundation | component | concept
Edge types:
    cites          paper       -> paper        (who cites whom)
    uses           paper       -> concept
    explains       foundation  -> concept
    implements     component   -> concept
    subconcept_of  concept     -> concept       (hierarchy, e.g. optical:fbg -> optical)

The source->concept edge type is determined purely by the SOURCE's kind (see
docs/sensor-knowledge-layer.md, decision B), so a component may `implements` both a
transduction concept and an application concept — the concept node's `category`
carries that distinction, no extra edge types needed.

This is where the black box opens: a `concept` used by papers but neither `explains`ed
by a foundation nor `implements`ed by a component is a *grounding gap* — physics your
work assumes but hasn't yet backed. See `grounding_gaps`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from .. import vocab
from ..config import GRAPH_PATH
from ..models import Component, Foundation, Paper, Source

# source kind -> the edge type it emits toward a concept
_EDGE_BY_KIND = {"paper": "uses", "foundation": "explains", "component": "implements"}


def build_graph(sources: list[Source]) -> nx.DiGraph:
    """Construct the typed knowledge graph from a heterogeneous corpus."""
    g = nx.DiGraph()

    # 1) Source nodes (+ citation edges + reference paper nodes).
    for s in sources:
        if isinstance(s, Paper):
            _add_paper_node(g, s)
            for ref in s.references:
                if not ref.paper_id:
                    continue
                if ref.paper_id not in g:
                    g.add_node(
                        ref.paper_id, kind="paper", title=ref.title, year=ref.year or 0,
                        is_seed=False, citation_count=0, doi=ref.doi or "", url="",
                    )
                g.add_edge(s.source_id, ref.paper_id, etype="cites")
        elif isinstance(s, Foundation):
            _add_foundation_node(g, s)
        elif isinstance(s, Component):
            _add_component_node(g, s)

    # Reference nodes that turned out to be seeds keep their seed attributes.
    for s in sources:
        if isinstance(s, Paper) and s.is_seed and s.source_id in g:
            g.nodes[s.source_id]["is_seed"] = True

    # 2) Concept nodes + typed source->concept edges.
    for s in sources:
        etype = _EDGE_BY_KIND.get(s.kind, "uses")
        for slug in s.concepts:
            _ensure_concept(g, slug)
            g.add_edge(s.source_id, slug, etype=etype)

    # 3) Concept hierarchy edges (only between concepts both present in the graph).
    for node, data in list(g.nodes(data=True)):
        if data.get("kind") != "concept":
            continue
        parent = data.get("parent_id")
        if parent and parent in g:
            g.add_edge(node, parent, etype="subconcept_of")

    return g


# ---- node builders ----------------------------------------------------------
def _add_paper_node(g: nx.DiGraph, p: Paper) -> None:
    g.add_node(
        p.source_id, kind="paper", title=p.title, year=p.year or 0, is_seed=p.is_seed,
        citation_count=p.citation_count, doi=p.doi or "", url=p.url or "",
        concepts=";".join(p.concepts),
    )


def _add_foundation_node(g: nx.DiGraph, f: Foundation) -> None:
    g.add_node(
        f.source_id, kind="foundation", title=f.title, is_seed=False,
        section_path=f.section_path or "", concepts=";".join(f.concepts),
    )


def _add_component_node(g: nx.DiGraph, c: Component) -> None:
    g.add_node(
        c.source_id, kind="component", title=c.title, is_seed=False,
        part_number=c.part_number or "", manufacturer=c.manufacturer or "",
        transduction=c.transduction or "", concepts=";".join(c.concepts),
    )


def _ensure_concept(g: nx.DiGraph, slug: str) -> None:
    if slug in g:
        return
    c = vocab.get(slug)
    g.add_node(
        slug, kind="concept", title=(c.label if c else slug),
        category=(c.category if c else "unknown"),
        parent_id=(c.parent_id if c and c.parent_id else ""),
    )


# ---- frontier (papers only) -------------------------------------------------
@dataclass
class FrontierCandidate:
    paper_id: str
    title: str
    year: int
    cited_by_seeds: int
    citation_count: int

    @property
    def score(self) -> float:
        return self.cited_by_seeds * 1000 + self.citation_count


def frontier_candidates(g: nx.DiGraph, top: int = 20) -> list[FrontierCandidate]:
    """Rank not-yet-ingested *papers* by how central they are to your seeds."""
    seed_ids = {n for n, d in g.nodes(data=True) if d.get("is_seed")}
    out: list[FrontierCandidate] = []
    for node, data in g.nodes(data=True):
        if data.get("kind") != "paper" or data.get("is_seed"):
            continue
        citing_seeds = sum(
            1 for pred in g.predecessors(node)
            if pred in seed_ids and g.edges[pred, node].get("etype") == "cites"
        )
        if citing_seeds == 0:
            continue
        out.append(FrontierCandidate(
            paper_id=node, title=data.get("title", "?"), year=int(data.get("year", 0)),
            cited_by_seeds=citing_seeds, citation_count=int(data.get("citation_count", 0)),
        ))
    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top]


# ---- concept coverage & grounding gaps --------------------------------------
@dataclass
class ConceptCoverage:
    concept_id: str
    label: str
    category: str
    used_by: int          # papers that use this concept
    explained_by: int     # foundations that explain it
    implemented_by: int   # components that implement it

    @property
    def is_grounded(self) -> bool:
        return (self.explained_by + self.implemented_by) > 0

    @property
    def is_gap(self) -> bool:
        """Used by ≥1 paper but grounded by no foundation or component."""
        return self.used_by > 0 and not self.is_grounded


def concept_coverage(g: nx.DiGraph) -> list[ConceptCoverage]:
    """Per-concept counts of uses / explains / implements edges."""
    out: list[ConceptCoverage] = []
    for node, data in g.nodes(data=True):
        if data.get("kind") != "concept":
            continue
        counts = Counter(
            g.edges[pred, node].get("etype") for pred in g.predecessors(node)
        )
        out.append(ConceptCoverage(
            concept_id=node, label=data.get("title", node),
            category=data.get("category", "unknown"),
            used_by=counts.get("uses", 0),
            explained_by=counts.get("explains", 0),
            implemented_by=counts.get("implements", 0),
        ))
    out.sort(key=lambda c: (c.category, -c.used_by, c.concept_id))
    return out


def grounding_gaps(g: nx.DiGraph) -> list[ConceptCoverage]:
    """Concepts your papers rely on but nothing grounds — the black boxes."""
    gaps = [c for c in concept_coverage(g) if c.is_gap]
    gaps.sort(key=lambda c: c.used_by, reverse=True)
    return gaps


def graph_summary(g: nx.DiGraph) -> dict:
    node_kinds = Counter(d.get("kind", "?") for _, d in g.nodes(data=True))
    edge_types = Counter(d.get("etype", "?") for *_, d in g.edges(data=True))
    return {"nodes": dict(node_kinds), "edges": dict(edge_types)}


# ---- persistence ------------------------------------------------------------
def save_graph(g: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(g, path)


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    if not path.exists():
        return nx.DiGraph()
    return nx.read_gexf(path)
