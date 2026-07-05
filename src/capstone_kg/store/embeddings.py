"""Local text embeddings via fastembed (ONNX runtime, no torch).

The model is downloaded and cached on first use. `BAAI/bge-small-en-v1.5`
(384-dim) is a strong, small default; change CAPSTONE_EMBED_MODEL to swap it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import get_settings


@lru_cache
def _model():
    from fastembed import TextEmbedding

    settings = get_settings()
    return TextEmbedding(model_name=settings.embed_model)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed passages. Returns an (n, dim) float32 array."""
    vectors = list(_model().embed(texts))
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a query. Returns a (dim,) float32 array."""
    vectors = list(_model().query_embed([text]))
    return np.array(vectors[0], dtype=np.float32)
