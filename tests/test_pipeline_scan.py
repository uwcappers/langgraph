"""Tests for source discovery / folder-based kind routing."""

from capstone_kg.pipeline import _section_label, scan_sources


def test_section_label_avoids_duplicated_number():
    # Title already leads with the number -> don't repeat it.
    assert _section_label("8.2", "8.2 Linear Capacitive Sensors") == "§8.2 Linear Capacitive Sensors"
    # Title lacks the number -> prepend it.
    assert _section_label("8.2", "Linear Capacitive Sensors") == "§8.2 Linear Capacitive Sensors"
    # No number -> title as-is.
    assert _section_label(None, "Introduction") == "Introduction"


def test_scan_routes_by_folder(tmp_path):
    (tmp_path / "glove_paper.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "foundations").mkdir()
    (tmp_path / "foundations" / "fraden.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "datasheets").mkdir()
    (tmp_path / "datasheets" / "a1324.pdf").write_bytes(b"%PDF-1.4")

    jobs = scan_sources(tmp_path)
    by_name = {p.name: kind for p, kind in jobs}
    assert by_name == {
        "glove_paper.pdf": "paper",
        "fraden.pdf": "foundation",
        "a1324.pdf": "component",
    }


def test_scan_missing_subfolders_is_ok(tmp_path):
    (tmp_path / "only_a_paper.pdf").write_bytes(b"%PDF-1.4")
    jobs = scan_sources(tmp_path)
    assert [kind for _, kind in jobs] == ["paper"]
