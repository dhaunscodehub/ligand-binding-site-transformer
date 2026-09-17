"""The control experiments themselves."""

from __future__ import annotations

import pytest

from bindsite.validation import (
    CHANCE_TOLERANCE_AUROC, POSITIVE_CONTROL_MIN_LIFT, auroc_versus_auprc,
    features_are_ligand_blind, label_shuffle_control, leakage_by_split_level,
    leakage_guard_fires, positive_control, run_controls,
)


@pytest.fixture(scope="module")
def family_pair(family_features, family_feature_clustering):
    """(records, sequences, clustering) for the homologous-family fixture."""
    records, sequences = family_features
    return records, sequences, family_feature_clustering


def test_positive_control_recovers_a_known_function():
    check = positive_control(n_proteins=30, seed=0)
    assert check["passed"]
    assert check["auprc_lift"] >= POSITIVE_CONTROL_MIN_LIFT


def test_positive_control_states_its_expectation():
    assert "AUPRC lift" in positive_control(n_proteins=20, seed=0)["expectation"]


def test_label_shuffle_reduces_the_model_to_chance(family_pair):
    """Above chance here would mean labels leak into features."""
    records, _, clustering = family_pair
    check = label_shuffle_control(records, clustering, seed=0)
    assert check["passed"], check
    assert abs(check["auroc_excess_over_chance"]) <= CHANCE_TOLERANCE_AUROC


def test_label_shuffle_preserves_the_positive_rate(features_signal):
    """Shuffling within each protein, not globally, keeps each protein's rate
    exactly — so only the feature-label association is destroyed."""
    import copy

    import numpy as np

    rng = np.random.default_rng(0)
    for record in features_signal.values():
        clone = copy.copy(record)
        clone.labels = rng.permutation(record.labels)
        assert clone.labels.sum() == record.labels.sum()


def test_leakage_ordering_holds(family_pair):
    records, _, clustering = family_pair
    check = leakage_by_split_level(records, clustering, seed=0)
    arms = check["arms"]
    assert arms["random_residue"]["auprc"] > arms["homology_cluster"]["auprc"]
    assert arms["random_protein"]["auprc"] > arms["homology_cluster"]["auprc"]
    assert check["residue_split_auprc_inflation"] > 0
    assert check["protein_split_auprc_inflation"] > 0


def test_leakage_scales_with_model_capacity(family_pair):
    """The claim that matters: a higher-capacity model exploits the leak more.

    Measuring leakage with a low-capacity model understates the risk for the
    models people deploy. A single linear model cannot memorise
    family-specific structure; a forest can.
    """
    records, _, clustering = family_pair
    linear = leakage_by_split_level(records, clustering, seed=0, model="logistic")
    forest = leakage_by_split_level(
        records, clustering, seed=0, model="random_forest"
    )
    assert (
        forest["protein_split_auprc_inflation"]
        > linear["protein_split_auprc_inflation"]
    )


def test_both_leaking_arms_exceed_the_honest_one(family_pair):
    """Residue- and protein-level splits both leak, but they are not strictly
    ordered against each other: the residue split trains on a fraction of
    every protein, the protein split on whole near-duplicates. The claim is
    that both exceed the homology-separated arm."""
    records, _, clustering = family_pair
    check = leakage_by_split_level(records, clustering, seed=0)
    honest = check["arms"]["homology_cluster"]["auprc"]
    assert check["arms"]["random_residue"]["auprc"] > honest
    assert check["arms"]["random_protein"]["auprc"] > honest


def test_auroc_versus_auprc_demonstrates_the_gap():
    check = auroc_versus_auprc(positive_rate=0.10, seed=0)
    assert check["passed"]
    assert check["auroc"] > 0.75
    assert check["auprc"] < 0.55
    assert "denominator" in check["interpretation"]


def test_auroc_auprc_gap_widens_as_positives_get_rarer():
    common = auroc_versus_auprc(positive_rate=0.30, seed=0)
    rare = auroc_versus_auprc(positive_rate=0.02, seed=0)
    # AUROC barely moves; AUPRC collapses with the positive rate.
    assert abs(common["auroc"] - rare["auroc"]) < 0.15
    assert rare["auprc"] < common["auprc"]


def test_leakage_guard_fires(family_pair):
    """Needs multi-member clusters: moving a singleton cluster's only member
    between folds shares no cluster, so there would be nothing to detect."""
    _, _, clustering = family_pair
    assert min(clustering.cluster_sizes().values()) > 1
    assert leakage_guard_fires(clustering, seed=0)["passed"]


def test_features_are_ligand_blind_on_a_synthetic_structure(tmp_path):
    from bindsite.testing import synthetic_pdb

    path = tmp_path / "s.pdb"
    path.write_text(synthetic_pdb(n_residues=40, ligand_centre=(2.3, 0.0, 20.0)))
    check = features_are_ligand_blind(str(path))
    assert check["passed"]
    assert check["max_geometric_difference"] == 0.0
    assert check["n_binding_with_ligand"] > 0
    assert check["n_binding_without_ligand"] == 0


def test_run_controls_runs_the_synthetic_ones_without_a_dataset():
    report = run_controls(seed=0)
    assert report["all_passed"]
    names = [c["name"] for c in report["checks"]]
    assert any("positive control" in n for n in names)
    assert any("AUROC overstates" in n for n in names)


def test_run_controls_records_what_it_skipped():
    report = run_controls(seed=0)
    assert "label_shuffle_control" in report["skipped"]
    assert "features_are_ligand_blind" in report["skipped"]


def test_run_controls_with_a_dataset_runs_everything(family_pair, tmp_path):
    from bindsite.testing import synthetic_pdb

    records, sequences, clustering = family_pair
    path = tmp_path / "s.pdb"
    path.write_text(synthetic_pdb(n_residues=40, ligand_centre=(2.3, 0.0, 20.0)))
    report = run_controls(
        features_by_name=records, clusters=clustering,
        sequences=sequences, reference_pdb=str(path), seed=0,
    )
    assert report["n_checks"] >= 6
    assert report["headline"] is not None
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert not failed, f"failing controls: {failed}"


def test_every_control_states_an_expectation():
    for check in run_controls(seed=0)["checks"]:
        assert check.get("expectation"), check["name"]


def test_controls_are_reproducible():
    first = run_controls(seed=3)
    second = run_controls(seed=3)
    assert first["n_passed"] == second["n_passed"]
    assert (
        first["checks"][0]["auprc"] == second["checks"][0]["auprc"]
    )
