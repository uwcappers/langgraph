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

    for node, data in g.nodes(data=True):
        is_seed = data.get("is_seed") in (True, "true", "True")
        net.add_node(
            node,
            label=(data.get("title", node) or node)[:40],
            title=data.get("title", node),
            color="#e0533d" if is_seed else "#4c78a8",
            size=25 if is_seed else 12,
        )
    for src, dst in g.edges():
        net.add_edge(src, dst)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out), notebook=False)
    return out
