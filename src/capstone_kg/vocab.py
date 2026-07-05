"""The controlled concept vocabulary + normalization.

This is the alignment layer that makes the concept bridge work: a paper's
"capacitive tactile sensor" and Fraden's "capacitance" chapter must map to the
*same* `concept_id`. Everything here is pure data + pure functions (no LLM).

Design (see docs/sensor-knowledge-layer.md):
  - Hierarchical slugs, e.g. "transduction:optical" and "transduction:optical:fbg".
  - Flat by default: children exist only where the corpus genuinely needs the split
    (optical is seeded with children because FBG vs. vision-based are radically
    different transduction mechanisms — the motivating example).
  - `normalize()` maps a raw phrase -> canonical slug via exact/alias matching, or
    returns None (the "other" route). Nothing is force-fit into a wrong bucket.
  - Aliases learned on ingest are persisted so future runs are more deterministic.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .config import LEARNED_ALIASES_PATH, UNMAPPED_PATH
from .models import Concept

# ---------------------------------------------------------------------------
# The controlled vocabulary. (concept_id, label, category, parent_id, aliases)
# ---------------------------------------------------------------------------
_V: list[tuple[str, str, str, str | None, list[str]]] = [
    # --- transduction principles (these are what foundations/datasheets ground) ---
    ("transduction:piezoresistive", "Piezoresistive sensing", "transduction", None,
     ["piezoresistive", "resistive", "piezoresistance", "force sensitive resistor",
      "force-sensitive resistor", "fsr", "resistive force sensor"]),
    ("transduction:capacitive", "Capacitive sensing", "transduction", None,
     ["capacitive", "capacitance", "capacitive sensing", "capacitive sensor",
      "parallel plate", "parallel-plate capacitor"]),
    ("transduction:piezoelectric", "Piezoelectric sensing", "transduction", None,
     ["piezoelectric", "piezo", "pvdf", "piezoelectric sensor"]),
    ("transduction:magnetic", "Magnetic / Hall-effect sensing", "transduction", None,
     ["magnetic", "hall effect", "hall-effect", "hall sensor", "hall element",
      "magnetometer", "magnetic sensing", "magnetic field sensor"]),
    ("transduction:optical", "Optical sensing", "transduction", None,
     ["optical", "optical sensing", "light based", "light-based", "photodiode",
      "waveguide", "optoelectronic"]),
    ("transduction:optical:fbg", "Fiber Bragg Grating", "transduction",
     "transduction:optical",
     ["fbg", "fiber bragg grating", "fibre bragg grating", "bragg grating",
      "bragg wavelength shift"]),
    ("transduction:optical:vision-based", "Vision-based tactile", "transduction",
     "transduction:optical",
     ["vision based tactile", "vision-based tactile", "visuotactile", "gelsight",
      "camera based tactile", "camera-based tactile", "digit sensor"]),
    ("transduction:barometric", "Barometric / MEMS pressure", "transduction", None,
     ["barometric", "mems pressure", "mems barometer", "air pressure sensor",
      "barometric pressure sensor"]),
    ("transduction:strain-gauge", "Strain gauge", "transduction", None,
     ["strain gauge", "strain gage", "metal foil gauge", "foil strain gauge"]),
    ("transduction:triboelectric", "Triboelectric sensing", "transduction", None,
     ["triboelectric", "teng", "triboelectric nanogenerator", "self powered sensor"]),
    ("transduction:inductive", "Inductive / eddy-current", "transduction", None,
     ["inductive", "eddy current", "eddy-current", "inductive sensing"]),
    ("transduction:quantum-tunneling", "Quantum tunneling composite", "transduction",
     None, ["quantum tunneling composite", "quantum tunnelling composite", "qtc"]),
    # --- mechanisms ---
    ("mechanism:hand-kinematics", "Hand kinematics", "mechanism", None,
     ["hand kinematics", "finger kinematics", "joint angle", "joint angles",
      "motion capture", "hand motion", "finger motion"]),
    ("mechanism:soft-actuation", "Soft actuation", "mechanism", None,
     ["soft actuator", "soft actuation", "pneumatic actuator", "tendon driven",
      "tendon-driven", "soft robotics"]),
    # --- applications ---
    ("application:force-sensing", "Force / pressure sensing", "application", None,
     ["force sensing", "force measurement", "contact force", "pressure sensing",
      "force data capture", "grip force", "normal force"]),
    ("application:tactile-array", "Tactile array", "application", None,
     ["tactile array", "taxel", "tactile skin", "tactile sensor array",
      "tactile sensing array"]),
    ("application:slip-detection", "Slip detection", "application", None,
     ["slip detection", "incipient slip", "slip sensing"]),
    ("application:exoskeleton-glove", "Exoskeleton glove", "application", None,
     ["exoskeleton glove", "exo-glove", "exo glove", "hand exoskeleton",
      "assistive glove", "robotic glove"]),
    ("application:wearable", "Wearable / e-skin", "application", None,
     ["wearable sensor", "e-skin", "electronic skin", "data glove",
      "instrumented glove", "wearable"]),
]

CANONICAL_CONCEPTS: list[Concept] = [
    Concept(concept_id=cid, label=label, category=cat, parent_id=parent, aliases=aliases)
    for cid, label, cat, parent, aliases in _V
]
BY_ID: dict[str, Concept] = {c.concept_id: c for c in CANONICAL_CONCEPTS}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Lowercase, punctuation -> space, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


@lru_cache
def _seed_index() -> dict[str, str]:
    """Normalized phrase -> concept_id, from labels + aliases + slug leaves."""
    idx: dict[str, str] = {}
    for c in CANONICAL_CONCEPTS:
        keys = {c.label, c.concept_id.split(":")[-1].replace("-", " "), *c.aliases}
        for k in keys:
            nk = _norm(k)
            if nk:
                idx.setdefault(nk, c.concept_id)
    return idx


def _load_learned(path: Path = LEARNED_ALIASES_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def normalize(phrase: str, learned_path: Path = LEARNED_ALIASES_PATH) -> str | None:
    """Map a raw concept phrase to a canonical concept_id, or None if unmapped."""
    if phrase in BY_ID:  # already a canonical slug
        return phrase
    key = _norm(phrase)
    if not key:
        return None
    learned = _load_learned(learned_path)
    if key in learned:
        return learned[key]
    return _seed_index().get(key)


def learn_alias(raw: str, concept_id: str, path: Path = LEARNED_ALIASES_PATH) -> None:
    """Persist a newly-observed phrase -> concept mapping for future determinism."""
    if concept_id not in BY_ID:
        raise ValueError(f"unknown concept_id: {concept_id}")
    key = _norm(raw)
    if not key or key in _seed_index():
        return
    learned = _load_learned(path)
    learned[key] = concept_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(learned, indent=2, sort_keys=True))


def log_unmapped(phrases: list[str], path: Path = UNMAPPED_PATH) -> None:
    """Append phrases that hit the 'other' route for later taxonomy review.

    No orphan concept nodes are created; nothing is silently dropped.
    """
    phrases = [p.strip() for p in phrases if p.strip()]
    if not phrases:
        return
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = sorted(set(existing) | set(phrases))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------
def get(concept_id: str) -> Concept | None:
    return BY_ID.get(concept_id)


def ancestors(concept_id: str) -> list[str]:
    """Parent chain, nearest first (empty for a top-level concept)."""
    out: list[str] = []
    cur = BY_ID.get(concept_id)
    while cur and cur.parent_id:
        out.append(cur.parent_id)
        cur = BY_ID.get(cur.parent_id)
    return out


def children(concept_id: str) -> list[Concept]:
    return [c for c in CANONICAL_CONCEPTS if c.parent_id == concept_id]


def all_concepts() -> list[Concept]:
    return list(CANONICAL_CONCEPTS)


def vocab_prompt_block() -> str:
    """A compact listing of the vocabulary for the extraction prompt (increment 3)."""
    lines = []
    for c in CANONICAL_CONCEPTS:
        alias_hint = f" (e.g. {c.aliases[0]})" if c.aliases else ""
        lines.append(f"- {c.concept_id}: {c.label}{alias_hint}")
    return "\n".join(lines)
