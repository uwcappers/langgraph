"""The LangGraph query pipeline (GraphRAG).

    retrieve  ->  expand (citation graph)  ->  answer

1. retrieve   — semantic search over paper chunks (the vector store).
2. expand     — for each retrieved paper, pull its graph neighborhood (what it
                cites / what cites it within the corpus) so the answer can reason
                about how the papers connect, not just their content.
3. answer     — Claude synthesizes a source-grounded answer with [n] citations.
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
    graph_notes: list[str]
    answer: str
    top_k: int


def _retrieve_node(store: VectorStore):
    def node(state: QueryState) -> QueryState:
        k = state.get("top_k", 6)
        return {"hits": store.search(state["question"], k=k)}

    return node


def _expand_node(g: nx.DiGraph):
    def node(state: QueryState) -> QueryState:
        notes: list[str] = []
        seen: set[str] = set()
        for hit in state.get("hits", []):
            pid = hit.chunk.paper_id
            if pid in seen or pid not in g:
                continue
            seen.add(pid)
            title = g.nodes[pid].get("title", hit.chunk.paper_title)
            cites = [g.nodes[t].get("title", t) for t in g.successors(pid)][:5]
            cited_by = [g.nodes[s].get("title", s) for s in g.predecessors(pid)][:5]
            line = f'"{title}"'
            if cites:
                line += f" — cites: {', '.join(cites)}"
            if cited_by:
                line += f" — cited by (in corpus): {', '.join(cited_by)}"
            notes.append(line)
        return {"graph_notes": notes}

    return node


_ANSWER_PROMPT = """You are a research assistant for an engineering capstone: a \
dexterous robotic glove that captures force data without impeding natural hand motion.

Answer the question using ONLY the sources below. Cite sources inline as [1], [2], \
etc. matching the numbered excerpts. If the sources are insufficient, say so plainly. \
When relevant, use the citation-graph notes to explain how the papers relate.

Question: {question}

Sources:
{sources}

Citation-graph notes:
{graph_notes}

Answer (with inline [n] citations, then a "Sources" list):"""


def _answer_node():
    def node(state: QueryState) -> QueryState:
        hits = state.get("hits", [])
        if not hits:
            return {"answer": "No indexed papers matched this question. "
                              "Ingest papers first with `capstone ingest`."}
        sources = "\n\n".join(
            f"[{i}] ({h.chunk.paper_title})\n{h.chunk.text[:1200]}"
            for i, h in enumerate(hits, 1)
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
    builder.add_node("expand", _expand_node(g))
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
