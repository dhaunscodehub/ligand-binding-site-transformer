"""Chemical Component Dictionary lookups and the cache."""

from __future__ import annotations

import json

import pytest

from bindsite.chemcomp import (
    ComponentCache, classify, component_type, parent_residue,
)

# A cache with known entries, so these tests need no network.
FIXTURE = {
    "HEM": {"id": "HEM", "type": "non-polymer", "name": "PROTOPORPHYRIN IX",
            "formula": "C34 H32 Fe N4 O4", "formula_weight": 616.5, "parent": None},
    "YOF": {"id": "YOF", "type": "L-peptide linking", "name": "3-FLUOROTYROSINE",
            "formula": "C9 H10 F N O3", "formula_weight": 199.2, "parent": ["TYR"]},
    "MSE": {"id": "MSE", "type": "L-peptide linking", "name": "SELENOMETHIONINE",
            "formula": "C5 H11 N O2 Se", "formula_weight": 196.1, "parent": "MET"},
    "NAG": {"id": "NAG", "type": "D-saccharide, beta linking", "name": "NAG",
            "formula": "C8 H15 N O6", "formula_weight": 221.2, "parent": None},
    "SO4": {"id": "SO4", "type": "non-polymer", "name": "SULFATE ION",
            "formula": "O4 S", "formula_weight": 96.1, "parent": None},
}


@pytest.fixture
def cache(tmp_path):
    path = tmp_path / "cc.json"
    path.write_text(json.dumps(FIXTURE))
    return ComponentCache(path, offline=True)


def test_cache_loads_from_disk(cache):
    assert len(cache) == len(FIXTURE)


def test_modified_residues_classify_as_polymer(cache):
    assert classify("YOF", cache) == "polymer"
    assert classify("MSE", cache) == "polymer"


def test_genuine_ligands_classify_as_ligand(cache):
    assert classify("HEM", cache) == "ligand"


def test_linking_saccharides_classify_as_saccharide(cache):
    assert classify("NAG", cache) == "saccharide"


def test_additives_are_typed_non_polymer_by_the_ccd(cache):
    """The CCD cannot distinguish a cryoprotectant from a substrate, which is
    why the curated exclusion lists are still needed."""
    assert classify("SO4", cache) == "ligand"


def test_unknown_component_classifies_as_unknown(cache):
    """An unresolvable lookup must not be guessed at."""
    assert classify("ZZZ", cache) == "unknown"
    assert "ZZZ" in cache.misses


def test_component_type_is_returned_verbatim(cache):
    assert component_type("HEM", cache) == "non-polymer"


def test_component_type_is_none_when_unknown(cache):
    assert component_type("ZZZ", cache) is None


def test_parent_residue_handles_a_list(cache):
    """The API returns a list for this field."""
    assert parent_residue("YOF", cache) == "TYR"


def test_parent_residue_handles_a_string(cache):
    assert parent_residue("MSE", cache) == "MET"


def test_parent_residue_is_none_for_a_ligand(cache):
    assert parent_residue("HEM", cache) is None


def test_an_empty_cache_is_not_discarded(tmp_path):
    """ComponentCache defines __len__, so an empty one is falsy. Passing it
    through `cache or ComponentCache()` would silently replace it, refetching
    every component and accumulating nothing."""
    path = tmp_path / "empty.json"
    cache = ComponentCache(path, offline=True)
    assert len(cache) == 0
    classify("ZZZ", cache)
    # The miss was recorded in the caller's cache, not a throwaway one.
    assert cache.misses == {"ZZZ"}


def test_case_is_normalised(cache):
    assert classify("hem", cache) == "ligand"
    assert classify(" HEM ", cache) == "ligand"


def test_offline_cache_records_failures_without_network(tmp_path):
    cache = ComponentCache(tmp_path / "c.json", offline=True)
    assert cache.get("HEM") is None
    report = cache.report()
    assert report["offline"] is True
    assert report["fallback_used"] is True
    assert report["n_lookup_failures"] == 1


def test_cache_round_trips(tmp_path, cache):
    path = cache.save()
    reloaded = ComponentCache(path, offline=True)
    assert len(reloaded) == len(FIXTURE)
    assert classify("YOF", reloaded) == "polymer"


def test_corrupt_cache_file_is_tolerated(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    cache = ComponentCache(path, offline=True)
    assert len(cache) == 0


def test_report_lists_the_cache_path(cache):
    assert str(cache.path) == cache.report()["cache_path"]
