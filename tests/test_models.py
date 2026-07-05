"""Tests for the heterogeneous source models and their (de)serialization."""

from capstone_kg.corpus import upsert
from capstone_kg.models import (
    Component,
    ComponentSpecs,
    Concept,
    Foundation,
    Paper,
    SourceListAdapter,
    Spec,
)


def test_kind_defaults():
    assert Paper(source_id="p1", title="P").kind == "paper"
    assert Foundation(source_id="f1", title="F").kind == "foundation"
    assert Component(source_id="c1", title="C").kind == "component"


def test_mixed_corpus_roundtrips_to_correct_subtypes():
    sources = [
        Paper(source_id="p1", title="A glove paper", year=2020),
        Foundation(
            source_id="foundation:fraden:8.2",
            title="Linear Capacitive Sensors",
            section_path="Fraden §8.2",
        ),
        Component(
            source_id="component:a1324",
            title="A1324 Hall sensor",
            part_number="A1324",
            transduction="transduction:magnetic",
            specs=ComponentSpecs(
                sensitivity=Spec(raw="5 mV/mT", value=5.0, unit="mV/mT"),
                bandwidth=Spec(raw="17 kHz", value=17.0, unit="kHz"),
            ),
        ),
    ]
    dumped = [s.model_dump() for s in sources]
    restored = SourceListAdapter.validate_python(dumped)

    assert isinstance(restored[0], Paper)
    assert isinstance(restored[1], Foundation)
    assert isinstance(restored[2], Component)
    # Nested spec survives the round trip.
    assert restored[2].specs.sensitivity.value == 5.0
    assert restored[2].specs.sensitivity.unit == "mV/mT"


def test_upsert_keys_on_source_id_across_kinds():
    a = Paper(source_id="p1", title="v1")
    b = Paper(source_id="p1", title="v2")  # same id, should replace
    c = Foundation(source_id="foundation:x", title="F")
    out = upsert(upsert([a], b), c)
    assert len(out) == 2
    assert {s.source_id for s in out} == {"p1", "foundation:x"}
    assert next(s for s in out if s.source_id == "p1").title == "v2"


def test_concept_is_flat_by_default():
    c = Concept(concept_id="transduction:optical", label="Optical", category="transduction")
    assert c.parent_id is None
    assert c.aliases == []
