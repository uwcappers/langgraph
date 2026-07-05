"""Persistence for the corpus: the metadata of every ingested paper."""

from __future__ import annotations

import json
from pathlib import Path

from .config import CORPUS_PATH
from .models import Paper


def save_corpus(papers: list[Paper], path: Path = CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([p.model_dump() for p in papers], indent=2))


def load_corpus(path: Path = CORPUS_PATH) -> list[Paper]:
    if not path.exists():
        return []
    return [Paper(**row) for row in json.loads(path.read_text())]


def upsert(papers: list[Paper], new: Paper) -> list[Paper]:
    """Add or replace a paper by paper_id, preserving order."""
    by_id = {p.paper_id: p for p in papers}
    by_id[new.paper_id] = new
    return list(by_id.values())
