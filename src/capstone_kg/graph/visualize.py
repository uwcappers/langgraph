"""Interactive citation-graph export (optional, needs the [viz] extra)."""

from __future__ import annotations

from pathlib import Path

from .citation_graph import load_graph


def export_html(output: str = "data/graph.html") -> Path:
    try:
        from pyvis.network import Network
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install viz extras first:  pip install -e '.[viz]'") from e

    g = load_graph()
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # Colour + size by node kind; seed papers stand out from cited-only papers.
    color = {"paper": "#4c78a8", "foundation": "#54a24b",
             "component": "#e0843d", "concept": "#b279a2"}
    size = {"paper": 16, "foundation": 20, "component": 20, "concept": 12}
    for node, data in g.nodes(data=True):
        kind = data.get("kind", "paper")
        is_seed = data.get("is_seed") in (True, "true", "True")
        net.add_node(
            node,
            label=(data.get("title", node) or node)[:40],
            title=f"[{kind}] {data.get('title', node)}",
            color="#e0533d" if (kind == "paper" and is_seed) else color.get(kind, "#4c78a8"),
            size=26 if (kind == "paper" and is_seed) else size.get(kind, 12),
        )
    for src, dst, edata in g.edges(data=True):
        net.add_edge(src, dst, title=edata.get("etype", ""))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out), notebook=False)
    return out
