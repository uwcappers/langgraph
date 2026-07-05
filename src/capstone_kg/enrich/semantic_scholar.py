"""Semantic Scholar Graph API client.

Used for two things:
  1. Identifying an ingested PDF (title -> canonical paper id + metadata).
  2. Pulling references / citations so we can build the citation graph and
     surface frontier candidates.

Responses are cached to disk so re-runs are fast and stay within rate limits.
The API works without a key at a lower rate limit; set SEMANTIC_SCHOLAR_API_KEY
to raise it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import CACHE_DIR, get_settings
from ..models import Author, Paper, PaperRef

API_BASE = "https://api.semanticscholar.org/graph/v1"

# Fields we request for a full paper record.
PAPER_FIELDS = (
    "paperId,title,abstract,year,authors,venue,externalIds,"
    "citationCount,referenceCount,url"
)
REFERENCE_FIELDS = "paperId,title,year,externalIds"


class SemanticScholarClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: float = 30.0) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        settings = get_settings()
        headers = {"User-Agent": "capstone-kg/0.1"}
        if settings.semantic_scholar_api_key:
            headers["x-api-key"] = settings.semantic_scholar_api_key
        self._client = httpx.Client(timeout=timeout, headers=headers)

    # ---- low-level cached GET ------------------------------------------------
    def _cache_key(self, path: str, params: dict[str, Any]) -> Path:
        raw = path + "?" + json.dumps(params, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:20]
        return self.cache_dir / f"s2_{digest}.json"

    def _get(self, path: str, params: dict[str, Any]) -> dict | None:
        cache_file = self._cache_key(path, params)
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        url = f"{API_BASE}{path}"
        for attempt in range(5):
            resp = self._client.get(url, params=params)
            if resp.status_code == 429:  # rate limited — back off
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            cache_file.write_text(json.dumps(data))
            # be polite even on success
            time.sleep(1.0)
            return data
        return None

    # ---- high-level operations ----------------------------------------------
    def match_paper(self, title: str) -> dict | None:
        """Find the single best paper matching a title string."""
        data = self._get("/paper/search/match", {"query": title, "fields": PAPER_FIELDS})
        if not data or not data.get("data"):
            return None
        return data["data"][0]

    def get_paper(self, paper_id: str) -> dict | None:
        """Fetch a full paper record (includes references)."""
        return self._get(
            f"/paper/{paper_id}",
            {"fields": f"{PAPER_FIELDS},references.paperId,references.title,"
                       "references.year,references.externalIds"},
        )

    def get_citations(self, paper_id: str, limit: int = 200) -> list[dict]:
        """Papers that cite `paper_id` (i.e., newer work — the frontier)."""
        data = self._get(
            f"/paper/{paper_id}/citations",
            {"fields": REFERENCE_FIELDS, "limit": limit},
        )
        if not data:
            return []
        return [row["citingPaper"] for row in data.get("data", []) if row.get("citingPaper")]

    def close(self) -> None:
        self._client.close()


# ---- conversion helpers -----------------------------------------------------
def _external_ids(raw: dict) -> tuple[str | None, str | None]:
    ext = raw.get("externalIds") or {}
    return ext.get("DOI"), ext.get("ArXiv")


def raw_to_paper(raw: dict, *, is_seed: bool, source_pdf: str | None = None) -> Paper:
    """Convert a Semantic Scholar paper record into our Paper model."""
    doi, arxiv = _external_ids(raw)
    refs: list[PaperRef] = []
    for r in raw.get("references", []) or []:
        if not r or not r.get("title"):
            continue
        r_doi, r_arxiv = _external_ids(r)
        refs.append(
            PaperRef(
                paper_id=r.get("paperId"),
                title=r["title"],
                year=r.get("year"),
                doi=r_doi,
                arxiv_id=r_arxiv,
            )
        )
    return Paper(
        paper_id=raw.get("paperId") or (source_pdf or raw.get("title", "unknown")),
        title=raw.get("title", "Untitled"),
        abstract=raw.get("abstract"),
        year=raw.get("year"),
        authors=[Author(name=a.get("name", "?"), author_id=a.get("authorId"))
                 for a in raw.get("authors", []) or []],
        venue=raw.get("venue"),
        doi=doi,
        arxiv_id=arxiv,
        url=raw.get("url"),
        source_pdf=source_pdf,
        is_seed=is_seed,
        references=refs,
        citation_count=raw.get("citationCount", 0) or 0,
        reference_count=raw.get("referenceCount", 0) or 0,
    )
