"""Vocab-aware concept extraction.

Given a source's text, an LLM maps it onto canonical concepts from the controlled
vocabulary (capstone_kg.vocab). The model returns `raw phrase | concept_id` pairs;
we then:
  - keep pairs that resolve to a real canonical slug,
  - `learn_alias` any novel phrasing so future runs are more deterministic,
  - route genuinely-unknown principles ("OTHER") to the unmapped log — never
    inventing an orphan concept.

The LLM call sits behind an injectable `llm` seam (a `str -> str` callable) so the
parsing / normalization / learning logic is fully unit-testable without a live key.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .. import vocab
from ..config import LEARNED_ALIASES_PATH, UNMAPPED_PATH, get_settings
from ..llm import get_llm
from ..models import Source

LLMFn = Callable[[str], str]

_PROMPT = """You are indexing sources for a project building a dexterous robotic \
glove that captures force data without impeding natural hand motion.

Below is a controlled vocabulary of concept ids. Read the source text and list every \
concept from the vocabulary that the text genuinely discusses. For each, output ONE \
line:

    <the exact phrase from the text> | <concept_id>

Use a concept_id EXACTLY as written in the vocabulary. If the text relies on a \
transduction principle or concept that is genuinely NOT in the vocabulary, output it \
with the id `OTHER` (do not force a wrong match). Output only these lines, nothing else.

Controlled vocabulary:
{vocab}

Source title: {title}
Source text:
{text}
"""


def _default_llm(prompt: str) -> str:
    resp = get_llm().invoke(prompt)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def parse_lines(text: str) -> list[tuple[str, str]]:
    """Parse the model's output into (raw_phrase, concept_id) pairs."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if "|" not in line:
            continue
        raw, slug = line.rsplit("|", 1)
        raw, slug = raw.strip(), slug.strip().strip("`")
        if raw and slug:
            pairs.append((raw, slug))
    return pairs


def map_pairs(
    pairs: list[tuple[str, str]],
    learned_path: Path = LEARNED_ALIASES_PATH,
    unmapped_path: Path = UNMAPPED_PATH,
) -> list[str]:
    """Resolve pairs to canonical slugs, learning aliases and logging the unmapped."""
    concepts: list[str] = []
    unmapped: list[str] = []
    for raw, slug in pairs:
        if slug.upper() == "OTHER":
            unmapped.append(raw)
            continue
        # Trust the model's slug if valid, else try to rescue via the raw phrase.
        canon = vocab.normalize(slug, learned_path=learned_path) or vocab.normalize(
            raw, learned_path=learned_path
        )
        if canon:
            concepts.append(canon)
            vocab.learn_alias(raw, canon, path=learned_path)  # no-op if already known
        else:
            unmapped.append(raw)
    if unmapped:
        vocab.log_unmapped(unmapped, path=unmapped_path)
    # Dedupe, preserving first-seen order.
    seen: set[str] = set()
    return [c for c in concepts if not (c in seen or seen.add(c))]


def _source_text(source: Source) -> str:
    abstract = getattr(source, "abstract", None)
    return f"{source.title}\n\n{abstract}" if abstract else source.title


def extract_concepts(
    source: Source,
    text: str | None = None,
    llm: LLMFn | None = None,
    learned_path: Path = LEARNED_ALIASES_PATH,
    unmapped_path: Path = UNMAPPED_PATH,
) -> list[str]:
    """Extract canonical concept slugs for a source.

    `text` overrides the default source text (used by ingestion to pass a foundation
    section or datasheet body). Returns [] when no LLM is available.
    """
    if llm is None:
        llm = _default_llm if get_settings().anthropic_api_key else None
    if llm is None:
        return []
    body = text if text is not None else _source_text(source)
    if not body.strip():
        return []
    prompt = _PROMPT.format(
        vocab=vocab.vocab_prompt_block(), title=source.title, text=body[:6000]
    )
    return map_pairs(parse_lines(llm(prompt)), learned_path, unmapped_path)
