"""A tiny persistent vector store: brute-force cosine similarity over numpy.

Perfect for a corpus of tens (even low hundreds) of papers — a few thousand
chunks search in milliseconds and there are zero native/database dependencies.
Persisted as a single JSON file. Swap in Chroma/FAISS if the corpus grows large.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import VECTORSTORE_PATH
from ..models import Chunk
from . import embeddings


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (n, dim), L2-normalized

    # ---- build ---------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = embeddings.embed_documents([c.text for c in chunks])
        vecs = _normalize(vecs)
        self._chunks.extend(chunks)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def source_ids(self) -> set[str]:
        return {c.source_id for c in self._chunks}

    # ---- search --------------------------------------------------------------
    def search(
        self, query: str, k: int = 6, where_source: set[str] | None = None
    ) -> list[SearchHit]:
        """Top-k chunks by cosine similarity. `where_source`, if given, restricts
        results to chunks from those source ids (used for concept-bridge grounding)."""
        if self._matrix is None or not self._chunks:
            return []
        q = _normalize(embeddings.embed_query(query)[None, :])[0]
        scores = self._matrix @ q  # cosine, since everything is normalized
        hits: list[SearchHit] = []
        for i in np.argsort(-scores):
            if where_source is not None and self._chunks[i].source_id not in where_source:
                continue
            hits.append(SearchHit(chunk=self._chunks[i], score=float(scores[i])))
            if len(hits) >= k:
                break
        return hits

    # ---- persistence ---------------------------------------------------------
    def save(self, path: Path = VECTORSTORE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._matrix.tolist() if self._matrix is not None else [],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path = VECTORSTORE_PATH) -> "VectorStore":
        store = cls()
        if not path.exists():
            return store
        payload = json.loads(path.read_text())
        store._chunks = [Chunk(**c) for c in payload["chunks"]]
        vectors = payload.get("vectors") or []
        store._matrix = np.array(vectors, dtype=np.float32) if vectors else None
        return store


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
