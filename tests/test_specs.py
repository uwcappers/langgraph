"""Tests for datasheet spec parsing and Component construction (no live LLM)."""

import pytest

from capstone_kg.ingest.datasheet import build_component, extract_component, parse_spec
from capstone_kg.models import Component


@pytest.mark.parametrize(
    "raw, value, unit",
    [
        ("5 mV/mT", 5.0, "mV/mT"),
        ("±40 mT", 40.0, "mT"),
        ("17 kHz", 17.0, "kHz"),
        ("30 µV RMS", 30.0, "µV RMS"),
        ("3.3 V", 3.3, "V"),
        ("~2.5 mV", 2.5, "mV"),
    ],
)
def test_parse_spec_numeric(raw, value, unit):
    s = parse_spec(raw)
    assert s.value == value and s.unit == unit and s.raw == raw


def test_parse_spec_non_numeric_keeps_raw():
    s = parse_spec("ratiometric analog output")
    assert s.value is None and s.unit is None and s.raw == "ratiometric analog output"


def test_parse_spec_none_and_blank():
    assert parse_spec(None) is None
    assert parse_spec("   ") is None


def test_build_component_from_dict():
    data = {
        "part_number": "A1324",
        "manufacturer": "Allegro",
        "transduction": "hall effect",
        "specs": {
            "sensitivity": "5 mV/mT",
            "bandwidth": "17 kHz",
            "noise_floor": "30 µV RMS",
            "output_type": "analog",
            "measurement_range": None,
        },
    }
    c = build_component(data, source_id="component:a1324", title="datasheet.pdf")
    assert isinstance(c, Component)
    assert c.part_number == "A1324"
    assert c.title == "A1324"  # part number preferred over filename
    assert c.transduction == "transduction:magnetic"  # normalized via vocab
    assert c.concepts == ["transduction:magnetic"]
    assert c.specs.sensitivity.value == 5.0
    assert c.specs.bandwidth.unit == "kHz"
    assert c.specs.measurement_range is None
    assert c.specs.output_type == "analog"


def test_extract_component_parses_json_with_fences():
    payload = (
        "```json\n"
        '{"part_number": "BMP390", "manufacturer": "Bosch", '
        '"transduction": "barometric", "specs": {"supply_voltage": "3.3 V"}}\n'
        "```"
    )
    c = extract_component(
        "some datasheet text", source_id="component:bmp390", title="bmp390.pdf",
        llm=lambda _p: payload,
    )
    assert c.part_number == "BMP390"
    assert c.transduction == "transduction:barometric"
    assert c.specs.supply_voltage.value == 3.3


def test_extract_component_no_llm_returns_bare_component():
    c = extract_component("text", source_id="component:x", title="x.pdf", llm=None)
    # With no key configured, returns a bare component rather than crashing.
    assert isinstance(c, Component) and c.source_id == "component:x"
