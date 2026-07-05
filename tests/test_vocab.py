"""Tests for the controlled concept vocabulary and normalization."""

import json

import pytest

from capstone_kg import vocab


# ---- normalization ---------------------------------------------------------
@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("Hall effect", "transduction:magnetic"),
        ("hall-effect sensor", None),  # not an exact alias; extractor would map it
        ("capacitive sensing", "transduction:capacitive"),
        ("Capacitance", "transduction:capacitive"),
        ("FBG", "transduction:optical:fbg"),
        ("Fiber Bragg Grating", "transduction:optical:fbg"),
        ("GelSight", "transduction:optical:vision-based"),
        ("force data capture", "application:force-sensing"),
        ("blockchain", None),
    ],
)
def test_normalize(phrase, expected):
    assert vocab.normalize(phrase) == expected


def test_canonical_slug_passes_through():
    assert vocab.normalize("transduction:capacitive") == "transduction:capacitive"


def test_label_and_slug_leaf_match():
    # The human label and the slug leaf both resolve.
    assert vocab.normalize("Piezoresistive sensing") == "transduction:piezoresistive"
    assert vocab.normalize("piezoresistive") == "transduction:piezoresistive"


# ---- hierarchy -------------------------------------------------------------
def test_hierarchy_parent_and_children():
    fbg = vocab.get("transduction:optical:fbg")
    assert fbg.parent_id == "transduction:optical"
    assert vocab.ancestors("transduction:optical:fbg") == ["transduction:optical"]
    child_ids = {c.concept_id for c in vocab.children("transduction:optical")}
    assert child_ids == {"transduction:optical:fbg", "transduction:optical:vision-based"}


def test_most_concepts_are_flat():
    flat = [c for c in vocab.all_concepts() if c.parent_id is None]
    assert len(flat) > len(vocab.all_concepts()) - len(flat)  # majority flat by design


# ---- vocabulary integrity --------------------------------------------------
def test_no_duplicate_ids():
    ids = [c.concept_id for c in vocab.all_concepts()]
    assert len(ids) == len(set(ids))


def test_every_parent_reference_exists():
    for c in vocab.all_concepts():
        if c.parent_id is not None:
            assert c.parent_id in vocab.BY_ID, f"{c.concept_id} -> missing parent"


def test_categories_are_valid():
    valid = {"transduction", "mechanism", "application"}
    assert all(c.category in valid for c in vocab.all_concepts())


# ---- alias learning + unmapped log (isolated to tmp paths) ----------------
def test_learn_alias_roundtrip(tmp_path):
    p = tmp_path / "learned.json"
    assert vocab.normalize("piezo film transducer", learned_path=p) is None
    vocab.learn_alias("piezo film transducer", "transduction:piezoelectric", path=p)
    assert vocab.normalize("piezo film transducer", learned_path=p) == "transduction:piezoelectric"


def test_learn_alias_ignores_existing_seed(tmp_path):
    p = tmp_path / "learned.json"
    vocab.learn_alias("capacitive", "transduction:capacitive", path=p)  # already seeded
    assert not p.exists()  # nothing new written


def test_learn_alias_rejects_unknown_concept(tmp_path):
    with pytest.raises(ValueError):
        vocab.learn_alias("whatever", "transduction:nonexistent", path=tmp_path / "l.json")


def test_log_unmapped_dedupes(tmp_path):
    p = tmp_path / "unmapped.json"
    vocab.log_unmapped(["novel sensor X", "novel sensor X", " weird thing "], path=p)
    vocab.log_unmapped(["novel sensor X", "another"], path=p)
    assert set(json.loads(p.read_text())) == {"novel sensor X", "weird thing", "another"}
