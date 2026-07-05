"""Tests for vocab-aware concept extraction (no live LLM)."""

import json

from capstone_kg.graph.concepts import extract_concepts, map_pairs, parse_lines
from capstone_kg.models import Paper


def test_parse_lines_tolerates_formatting():
    text = (
        "- capacitive tactile sensor | transduction:capacitive\n"
        "here is some prose without a delimiter\n"
        "* Hall effect | `transduction:magnetic`\n"
        "grip force | application:force-sensing\n"
    )
    pairs = parse_lines(text)
    assert pairs == [
        ("capacitive tactile sensor", "transduction:capacitive"),
        ("Hall effect", "transduction:magnetic"),
        ("grip force", "application:force-sensing"),
    ]


def test_map_pairs_resolves_learns_and_dedupes(tmp_path):
    learned, unmapped = tmp_path / "l.json", tmp_path / "u.json"
    pairs = [
        ("capacitive tactile sensor", "transduction:capacitive"),
        ("a capacitance bridge", "transduction:capacitive"),  # duplicate concept
        ("magneto-resistive readout", "transduction:magnetic"),  # novel phrasing
    ]
    out = map_pairs(pairs, learned_path=learned, unmapped_path=unmapped)
    assert out == ["transduction:capacitive", "transduction:magnetic"]  # deduped, ordered
    # The novel phrasing was learned for next time.
    learned_map = json.loads(learned.read_text())
    assert learned_map.get("magneto resistive readout") == "transduction:magnetic"


def test_map_pairs_rescues_via_raw_phrase(tmp_path):
    # Model emitted a bogus slug, but the raw phrase is a known alias -> rescued.
    out = map_pairs(
        [("capacitive", "transduction:bogus")],
        learned_path=tmp_path / "l.json",
        unmapped_path=tmp_path / "u.json",
    )
    assert out == ["transduction:capacitive"]


def test_map_pairs_routes_other_and_unknown_to_unmapped(tmp_path):
    unmapped = tmp_path / "u.json"
    out = map_pairs(
        [("some novel principle", "OTHER"), ("gobbledygook", "also-bogus")],
        learned_path=tmp_path / "l.json",
        unmapped_path=unmapped,
    )
    assert out == []
    assert set(json.loads(unmapped.read_text())) == {"some novel principle", "gobbledygook"}


def test_extract_concepts_with_injected_llm(tmp_path):
    paper = Paper(source_id="p1", title="A capacitive glove", abstract="We measure grip force.")
    fake_llm = lambda _p: (
        "capacitive | transduction:capacitive\n"
        "grip force | application:force-sensing\n"
    )
    out = extract_concepts(
        paper,
        llm=fake_llm,
        learned_path=tmp_path / "l.json",
        unmapped_path=tmp_path / "u.json",
    )
    assert out == ["transduction:capacitive", "application:force-sensing"]


def test_extract_concepts_empty_text_short_circuits():
    # Empty body must return [] WITHOUT invoking the LLM.
    def boom(_p):
        raise AssertionError("LLM should not be called for empty text")

    paper = Paper(source_id="p1", title="   ")
    assert extract_concepts(paper, text="", llm=boom) == []
