"""The LangGraph query pipeline (GraphRAG over the knowledge graph).

    retrieve  ->  expand (citations + concept bridge)  ->  answer

1. retrieve  — semantic search over all source chunks (papers, foundations, components).
2. expand    — for each retrieved paper, walk the graph:
                 * citation neighbors (what it cites / is cited by, within the corpus)
                 * the CONCEPT BRIDGE — for each concept the source touches, find the
                   foundations that `explains` it and components that `implements` it,
                   and pull their most relevant chunk in as extra grounding sources.
               This lets an answer drop from a paper's claim to the underlying physics
               (a textbook section) and real numbers (a datasheet), even when those
               weren't in the top-k retrieval.
3. answer    — Claude synthesizes a source-grounded answer with [n] citations.
"""

from __future__ import annotations

from typing import TypedDict

import networkx as nx
from langgraph.graph import END, START, StateGraph

from ..graph.citation_graph import load_graph
from ..llm import get_llm
from ..store.vectorstore import SearchHit, VectorStore


class QueryState(TypedDict, total=False):
    question: str
    hits: list[SearchHit]
    grounding: list[SearchHit]
    graph_notes: list[str]
    answer: str
    top_k: int


def _etype(g: nx.DiGraph, u: str, v: str) -> str:
    return g.edges[u, v].get("etype", "")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _retrieve_node(store: VectorStore):
    def node(state: QueryState) -> QueryState:
        return {"hits": store.search(state["question"], k=state.get("top_k", 6))}

    return node


def _expand_node(store: VectorStore, g: nx.DiGraph):
    def node(state: QueryState) -> QueryState:
        hits = state.get("hits", [])
        retrieved = {h.chunk.source_id for h in hits}
        notes: list[str] = []
        grounding_ids: set[str] = set()
        seen_src: set[str] = set()
        seen_concept: set[str] = set()

        for hit in hits:
            sid = hit.chunk.source_id
            if sid not in g or sid in seen_src:
                continue
            seen_src.add(sid)
            data = g.nodes[sid]
            title = data.get("title", hit.chunk.source_title)

            if data.get("kind") == "paper":
                cites = [g.nodes[t].get("title", t) for t in g.successors(sid)
                         if _etype(g, sid, t) == "cites"][:4]
                cited_by = [g.nodes[s].get("title", s) for s in g.predecessors(sid)
                            if _etype(g, s, sid) == "cites"][:4]
                line = f'"{title}"'
                if cites:
                    line += f" — cites: {', '.join(cites)}"
                if cited_by:
                    line += f" — cited by (in corpus): {', '.join(cited_by)}"
                if cites or cited_by:
                    notes.append(line)

            # Concept bridge.
            for c in g.successors(sid):
                if g.nodes[c].get("kind") != "concept" or c in seen_concept:
                    continue
                seen_concept.add(c)
                label = g.nodes[c].get("title", c)
                explains = [p for p in g.predecessors(c) if _etype(g, p, c) == "explains"]
                implements = [p for p in g.predecessors(c) if _etype(g, p, c) == "implements"]
                grounding_ids.update(x for x in explains + implements if x not in retrieved)
                parts = []
                if explains:
                    parts.append("physics: " + ", ".join(
                        g.nodes[e].get("title", e) for e in explains[:2]))
                if implements:
                    parts.append("real parts: " + ", ".join(
                        g.nodes[i].get("title", i) for i in implements[:2]))
                if parts:
                    notes.append(f'concept "{label}" — ' + "; ".join(parts))

        grounding: list[SearchHit] = (
            store.search(state["question"], k=3, where_source=grounding_ids)
            if grounding_ids else []
        )
        return {"graph_notes": _dedupe(notes), "grounding": grounding}

    return node


_ANSWER_PROMPT = """You are a research assistant for an engineering capstone: a \
dexterous robotic glove that captures force data without impeding natural hand motion.

Answer the question using ONLY the numbered sources below. Cite sources inline as [1], \
[2], etc. Sources tagged (foundation) are textbook physics and (component) are real \
datasheets — prefer them when the question needs first-principles grounding or real \
numbers (noise floor, bandwidth, sensitivity). If the sources are insufficient, say so.

Question: {question}

Sources:
{sources}

Graph notes (how the sources relate; citations and the physics/parts behind each concept):
{graph_notes}

Answer (inline [n] citations, then a "Sources" list):"""


def _answer_node():
    def node(state: QueryState) -> QueryState:
        hits = state.get("hits", [])
        if not hits:
            return {"answer": "No indexed sources matched this question. "
                              "Ingest sources first with `capstone ingest`."}
        all_hits = hits + state.get("grounding", [])
        sources = "\n\n".join(
            f"[{i}] ({h.chunk.source_kind}: {h.chunk.source_title})\n{h.chunk.text[:1100]}"
            for i, h in enumerate(all_hits, 1)
        )
        graph_notes = "\n".join(state.get("graph_notes", [])) or "(none)"
        prompt = _ANSWER_PROMPT.format(
            question=state["question"], sources=sources, graph_notes=graph_notes
        )
        resp = get_llm().invoke(prompt)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"answer": text}

    return node


def build_query_graph(store: VectorStore, g: nx.DiGraph):
    """Compile the LangGraph pipeline."""
    builder = StateGraph(QueryState)
    builder.add_node("retrieve", _retrieve_node(store))
    builder.add_node("expand", _expand_node(store, g))
    builder.add_node("answer", _answer_node())
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "expand")
    builder.add_edge("expand", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def answer_question(question: str, top_k: int = 6) -> QueryState:
    """Convenience entry point: load artifacts and run the graph once."""
    store = VectorStore.load()
    g = load_graph()
    app = build_query_graph(store, g)
    return app.invoke({"question": question, "top_k": top_k})
