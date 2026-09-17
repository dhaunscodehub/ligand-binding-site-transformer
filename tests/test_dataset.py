"""Dataset assembly: curation attrition and manifest round-tripping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bindsite.data.dataset import Dataset, build_dataset, save_manifest
from bindsite.exceptions import DataError

CACHE = Path(__file__).resolve().parent.parent / "data/pdb"
pytestmark = pytest.mark.skipif(
    not CACHE.is_dir() or len(list(CACHE.glob("*.pdb"))) < 5,
    reason="cached PDB structures not available",
)


@pytest.fixture(scope="module")
def cached_ids():
    return sorted(p.stem for p in CACHE.glob("*.pdb"))[:25]


def test_build_dataset_curates_and_reports_attrition(cached_ids):
    dataset = build_dataset(cached_ids, cache_dir=CACHE)
    statistics = dataset.statistics()
    assert statistics["n_attempted"] == len(cached_ids)
    # Most PDB entries hold only water, ions and additives, so attrition is
    # expected and must be visible rather than silent.
    assert statistics["n_rejected"] + statistics["n_proteins"] == len(cached_ids)


def test_rejections_are_categorised(cached_ids):
    dataset = build_dataset(cached_ids, cache_dir=CACHE)
    reasons = dataset.statistics()["rejection_reasons"]
    assert isinstance(reasons, dict)
    if dataset.rejected:
        assert sum(reasons.values()) == len(dataset.rejected)


def test_every_entry_has_a_ligand_and_binding_residues(cached_ids):
    dataset = build_dataset(cached_ids, cache_dir=CACHE, min_binding_residues=3)
    for entry in dataset.entries.values():
        assert entry.structure.ligands
        assert entry.structure.n_binding >= 3


def test_binding_fraction_stays_below_the_ceiling(cached_ids):
    """A ligand contacting more than half the chain is a polymer, not a pocket."""
    dataset = build_dataset(cached_ids, cache_dir=CACHE)
    for entry in dataset.entries.values():
        assert entry.structure.binding_fraction <= 0.5


def test_length_filters_are_applied(cached_ids):
    dataset = build_dataset(
        cached_ids, cache_dir=CACHE, min_length=100, max_length=250
    )
    for entry in dataset.entries.values():
        assert 100 <= entry.structure.n_residues <= 250


def test_a_larger_cutoff_raises_the_positive_rate(cached_ids):
    low = build_dataset(cached_ids, cache_dir=CACHE, cutoff=4.0)
    high = build_dataset(cached_ids, cache_dir=CACHE, cutoff=6.0)
    if low.entries and high.entries:
        assert high.statistics()["positive_rate"] > low.statistics()["positive_rate"]


def test_sequences_and_labels_align(cached_ids):
    dataset = build_dataset(cached_ids, cache_dir=CACHE)
    sequences, labels = dataset.sequences(), dataset.labels()
    assert set(sequences) == set(labels)
    for name in sequences:
        assert len(sequences[name]) == labels[name].size


def test_statistics_report_the_cutoff_and_curation(cached_ids):
    dataset = build_dataset(cached_ids, cache_dir=CACHE, cutoff=4.5)
    statistics = dataset.statistics()
    assert statistics["contact_cutoff"] == 4.5
    assert "curation" in statistics


def test_ccd_curation_is_recorded_when_used(cached_ids, tmp_path):
    from bindsite.chemcomp import ComponentCache

    cache = ComponentCache(tmp_path / "cc.json", offline=True)
    dataset = build_dataset(cached_ids, cache_dir=CACHE, component_cache=cache)
    assert "chemical component dictionary" in dataset.statistics()["curation"]


def test_statistics_on_an_empty_dataset():
    assert Dataset().statistics() == {"n_proteins": 0}


def test_manifest_round_trips(cached_ids, tmp_path):
    dataset = build_dataset(cached_ids, cache_dir=CACHE)
    path = save_manifest(dataset, tmp_path / "manifest.json")
    manifest = json.loads(path.read_text())
    assert sorted(manifest["proteins"]) == dataset.names
    assert manifest["statistics"]["n_proteins"] == len(dataset)


def test_manifest_records_rejections(cached_ids, tmp_path):
    dataset = build_dataset(cached_ids, cache_dir=CACHE)
    manifest = json.loads(save_manifest(dataset, tmp_path / "m.json").read_text())
    assert set(manifest["rejected"]) == set(dataset.rejected)


def test_max_proteins_caps_the_build(cached_ids):
    dataset = build_dataset(cached_ids, cache_dir=CACHE, max_proteins=2)
    assert len(dataset) <= 2


def test_missing_manifest_is_reported(tmp_path):
    from bindsite.data.dataset import load_from_manifest

    with pytest.raises(DataError, match="not found"):
        load_from_manifest(tmp_path / "absent.json")


def test_manifest_without_proteins_is_reported(tmp_path):
    from bindsite.data.dataset import load_from_manifest

    path = tmp_path / "m.json"
    path.write_text(json.dumps({"proteins": {}}))
    with pytest.raises(DataError, match="lists no proteins"):
        load_from_manifest(path)


def test_the_shipped_manifest_is_consistent():
    """The committed manifest's statistics must match its protein list."""
    path = Path(__file__).resolve().parent.parent / "data/dataset_manifest.json"
    if not path.is_file():
        pytest.skip("no manifest built")
    manifest = json.loads(path.read_text())
    proteins = manifest["proteins"]
    statistics = manifest["statistics"]
    assert statistics["n_proteins"] == len(proteins)
    assert statistics["n_residues"] == sum(p["n_residues"] for p in proteins.values())
    assert statistics["n_binding_residues"] == sum(
        p["n_binding"] for p in proteins.values()
    )
    assert statistics["positive_rate"] == pytest.approx(
        statistics["n_binding_residues"] / statistics["n_residues"]
    )
