"""Central configuration and filesystem layout.

All paths are derived from the repository root so the project is relocatable.
Runtime knobs (model names, API keys) come from the environment / a .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (src/capstone_kg/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the user drops PDFs.
PAPERS_DIR = REPO_ROOT / "papers"

# Generated artifacts (gitignored).
DATA_DIR = REPO_ROOT / "data"
VECTORSTORE_PATH = DATA_DIR / "vectorstore.json"
GRAPH_PATH = DATA_DIR / "citation_graph.gexf"
CORPUS_PATH = DATA_DIR / "corpus.json"  # metadata for every ingested paper
CACHE_DIR = DATA_DIR / "cache"  # cached Semantic Scholar responses
LEARNED_ALIASES_PATH = DATA_DIR / "learned_aliases.json"  # concept aliases learned on ingest
UNMAPPED_PATH = DATA_DIR / "unmapped_concepts.json"  # phrases that hit the "other" route


class Settings(BaseSettings):
    """Runtime settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(default="", alias="ANTHROPIC_BASE_URL")
    llm_model: str = Field(default="claude-opus-4-8", alias="CAPSTONE_LLM_MODEL")

    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="CAPSTONE_EMBED_MODEL")

    semantic_scholar_api_key: str = Field(default="", alias="SEMANTIC_SCHOLAR_API_KEY")

    # Chunking
    chunk_size: int = Field(default=1200, alias="CAPSTONE_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CAPSTONE_CHUNK_OVERLAP")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_dirs() -> None:
    """Create the runtime directories if they do not exist."""
    for d in (
        PAPERS_DIR,
        PAPERS_DIR / "foundations",
        PAPERS_DIR / "datasheets",
        DATA_DIR,
        CACHE_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
