"""LLM concept extraction.

Given a paper's title + abstract, extract a small set of normalized engineering
concepts (sensing modality, actuation, materials, application, etc.). These become
node attributes and enable concept-level queries like "which papers use capacitive
force sensing?" — beyond what keyword search alone can do.

Degrades gracefully: if no Anthropic key is configured, returns [].
"""

from __future__ import annotations

from ..config import get_settings
from ..llm import get_llm
from ..models import Paper

_PROMPT = """You are indexing research papers for a project building a dexterous \
robotic glove that captures force data without impeding natural hand motion.

From the paper below, extract 4-10 concise, normalized technical concepts most \
useful for connecting it to related work. Prefer canonical terms (e.g. \
"capacitive force sensing", "soft pneumatic actuator", "exoskeleton glove", \
"tactile array", "hand kinematics"). Return ONLY a comma-separated list, no prose.

Title: {title}
Abstract: {abstract}
"""


def extract_concepts(paper: Paper) -> list[str]:
    settings = get_settings()
    if not settings.anthropic_api_key:
        return []
    if not paper.abstract:
        return []
    llm = get_llm()
    msg = _PROMPT.format(title=paper.title, abstract=paper.abstract[:4000])
    resp = llm.invoke(msg)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    concepts = [c.strip().lower() for c in text.split(",")]
    return [c for c in concepts if c and len(c) < 60][:10]
