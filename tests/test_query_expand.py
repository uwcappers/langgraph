"""Tests for the query pipeline's concept-bridge expansion (no LLM; embeddings only).

These exercise retrieve -> expand, stopping before the answer node so no API key is
needed. The embedding model is downloaded once and cached.
"""

from capstone_kg.agent.query_graph import _expand_node, _retrieve_node
from capstone_kg.graph.citation_graph import build_graph
from capstone_kg.models import Chunk, Foundation, Paper
from capstone_kg.store.vectorstore import VectorStore


def _store():
    s = VectorStore()
    s.add_chunks([
        Chunk(chunk_id="P1::0", source_id="P1", source_title="Capacitive glove",
              source_kind="paper",
              text="We use a capacitive sensor to measure grip force at the fingertips."),
        Chunk(chunk_id="F1::0", source_id="F1", source_title="Capacitance",
              source_kind="foundation",
              text="The capacitance between parallel plates depends on area, gap, "
                   "and the dielectric permittivity of the material between them."),
    ])
    return s


def _graph():
    return build_graph([
        Paper(source_id="P1", title="Capacitive glove", is_seed=True,
              concepts=["transduction:capacitive"]),
        Foundation(source_id="F1", title="Capacitance", concepts=["transduction:capacitive"]),
    ])


def test_bridge_pulls_grounding_foundation():
    store, g = _store(), _graph()
    # top_k=1 -> only the paper is retrieved; the foundation must arrive via the bridge.
    state = {"question": "how is fingertip contact force measured?", "top_k": 1}
    state.update(_retrieve_node(store)(state))
    assert [h.chunk.source_id for h in state["hits"]] == ["P1"]

    out = _expand_node(store, g)(state)
    grounding_ids = {h.chunk.source_id for h in out["grounding"]}
    assert "F1" in grounding_ids  # foundation that explains 'capacitive' was pulled in
    assert any("physics" in n.lower() for n in out["graph_notes"])


def test_where_source_filters_search():
    store = _store()
    hits = store.search("dielectric permittivity", k=5, where_source={"F1"})
    assert hits and all(h.chunk.source_id == "F1" for h in hits)
