"""Datasheet -> Component with structured specs.

A datasheet is where "beyond surface level" lives: a real noise floor and a real
bandwidth, not an abstract's "high resolution". An LLM pulls the spec table into
`ComponentSpecs`; each field keeps its original string (`raw`) alongside a parsed
`value`/`unit` so numbers are both faithful and comparable across parts.

As with concept extraction, the LLM sits behind an injectable `llm` seam and returns
JSON, so the JSON->Component mapping and the value/unit parsing are unit-testable
without a live key.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from .. import vocab
from ..config import LEARNED_ALIASES_PATH, get_settings
from ..llm import get_llm
from ..models import Component, ComponentSpecs, Spec

LLMFn = Callable[[str], str]

_SPEC_FIELDS = (
    "measurement_range",
    "sensitivity",
    "noise_floor",
    "bandwidth",
    "temperature_coefficient",
    "supply_voltage",
)

_PROMPT = """You are reading a sensor component datasheet. Extract the following as \
strict JSON (no prose, no code fences):

{{
  "part_number": string | null,
  "manufacturer": string | null,
  "transduction": string | null,   // the physical sensing principle, e.g. "hall effect"
  "specs": {{
    "measurement_range": string | null,        // e.g. "±40 mT"
    "sensitivity": string | null,              // e.g. "5 mV/mT"
    "noise_floor": string | null,              // e.g. "30 µV RMS"
    "bandwidth": string | null,                // e.g. "17 kHz"
    "temperature_coefficient": string | null,  // e.g. "0.12 %/°C"
    "supply_voltage": string | null,           // e.g. "3.3 V"
    "output_type": string | null               // "analog" | "I2C" | "SPI" | ...
  }}
}}

Use the datasheet's exact numbers and units. Use null for anything not stated.

Datasheet text:
{text}
"""

# leading sign / ± / ~, then a number, then the rest is the unit
_NUM = re.compile(r"^[~≈]?\s*[-+±]?\s*(\d+(?:\.\d+)?)\s*(.*)$")


def parse_spec(raw: str | None) -> Spec | None:
    """Parse a spec string into value + unit, always retaining the raw text."""
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    m = _NUM.match(raw)
    if not m:
        return Spec(raw=raw)  # non-numeric (e.g. "ratiometric"); keep the text
    unit = m.group(2).strip() or None
    return Spec(raw=raw, value=float(m.group(1)), unit=unit)


def build_component(
    data: dict,
    *,
    source_id: str,
    title: str,
    source_file: str | None = None,
    learned_path: Path = LEARNED_ALIASES_PATH,
) -> Component:
    """Map an extracted datasheet dict to a Component (pure; no LLM)."""
    specs_in = data.get("specs") or {}
    specs = ComponentSpecs(
        **{f: parse_spec(specs_in.get(f)) for f in _SPEC_FIELDS},
        output_type=(specs_in.get("output_type") or None),
    )
    transduction = None
    raw_principle = data.get("transduction")
    if raw_principle:
        transduction = vocab.normalize(raw_principle, learned_path=learned_path)
    return Component(
        source_id=source_id,
        title=data.get("part_number") or title,
        source_file=source_file,
        part_number=data.get("part_number"),
        manufacturer=data.get("manufacturer"),
        transduction=transduction,
        specs=specs,
        concepts=[transduction] if transduction else [],
    )


def _load_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(text[start : end + 1])


def _default_llm(prompt: str) -> str:
    resp = get_llm().invoke(prompt)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def extract_component(
    text: str,
    *,
    source_id: str,
    title: str,
    source_file: str | None = None,
    llm: LLMFn | None = None,
    learned_path: Path = LEARNED_ALIASES_PATH,
) -> Component:
    """Extract a Component from datasheet text. Falls back to a bare Component
    (title only) when no LLM is available."""
    if llm is None:
        llm = _default_llm if get_settings().anthropic_api_key else None
    if llm is None or not text.strip():
        return Component(source_id=source_id, title=title, source_file=source_file)
    data = _load_json(llm(_PROMPT.format(text=text[:8000])))
    return build_component(
        data,
        source_id=source_id,
        title=title,
        source_file=source_file,
        learned_path=learned_path,
    )
