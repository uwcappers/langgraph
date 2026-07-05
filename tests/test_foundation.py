"""Tests for TOC-based foundation section splitting."""

import fitz

from capstone_kg.ingest.foundation import _section_number, split_pdf_sections


def _make_pdf_with_toc(path):
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i}. " + "capacitive sensing physics. " * 30)
    doc.set_toc([[1, "1 Introduction", 1], [1, "8.2 Linear Capacitive Sensors", 3]])
    doc.save(str(path))
    doc.close()


def test_splits_on_toc_boundaries(tmp_path):
    p = tmp_path / "fraden.pdf"
    _make_pdf_with_toc(p)
    secs = split_pdf_sections(p, min_chars=10)
    assert len(secs) == 2
    assert secs[0].title == "1 Introduction"
    assert secs[1].title == "8.2 Linear Capacitive Sensors"
    assert secs[1].number == "8.2"
    assert "capacitive" in secs[1].text.lower()


def test_no_toc_falls_back_to_whole_doc(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "some content " * 50)
    p = tmp_path / "flat.pdf"
    doc.save(str(p))
    doc.close()
    secs = split_pdf_sections(p, min_chars=10)
    assert len(secs) == 1
    assert secs[0].level == 0  # whole-doc sentinel


def test_section_number_parsing():
    assert _section_number("8.2 Linear Capacitive Sensors") == "8.2"
    assert _section_number("1.4.3 Deep subsection") == "1.4.3"
    assert _section_number("1 Introduction") is None  # a dotted number is required
    assert _section_number("Chapter Nine") is None
