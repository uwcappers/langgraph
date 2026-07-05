"""Persistence for the corpus: the metadata of every ingested source.

The corpus is heterogeneous (papers, foundations, components); a discriminated
union on `kind` round-trips each subtype to/from the right model.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import CORPUS_PATH
from .models import AnySource, SourceListAdapter, Source


def save_corpus(sources: list[Source], path: Path = CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [s.model_dump() for s in sources]
    path.write_text(json.dumps(payload, indent=2))


def load_corpus(path: Path = CORPUS_PATH) -> list[AnySource]:
    if not path.exists():
        return []
    return SourceListAdapter.validate_python(json.loads(path.read_text()))


def upsert(sources: list[Source], new: Source) -> list[Source]:
    """Add or replace a source by source_id, preserving order."""
    by_id = {s.source_id: s for s in sources}
    by_id[new.source_id] = new
    return list(by_id.values())
