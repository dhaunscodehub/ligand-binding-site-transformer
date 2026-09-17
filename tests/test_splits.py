"""Homology-separated splits and the leakage guards."""

from __future__ import annotations

import pytest

from bindsite.data.splits import (
    MIN_CLUSTERS, Split, assert_no_homology_leakage, cluster_kfold,
    homology_split, label_balance, random_split, residue_split,
)
from bindsite.exceptions import DataError, LeakageError
from bindsite.homology import Clustering, cluster_connected_components


def _clustering(n_clusters: int = 6, per_cluster: int = 4) -> Clustering:
    labels = {
        f"P{c}_{m}": c for c in range(n_clusters) for m in range(per_cluster)
    }
    return Clustering(
        labels=labels, identity_threshold=0.30, coverage_threshold=0.50,
        backend="test", exact=True,
    )


def test_homology_split_shares_no_cluster():
    split = homology_split(_clustering(), test_fraction=0.25, val_fraction=0.25, seed=0)
    assert not set(split.train_clusters) & set(split.test_clusters)
    assert not set(split.train_clusters) & set(split.val_clusters)
    assert not set(split.val_clusters) & set(split.test_clusters)


def test_homology_split_covers_every_protein_once():
    clustering = _clustering()
    split = homology_split(clustering, test_fraction=0.25, val_fraction=0.25, seed=0)
    allocated = sorted(split.train + split.val + split.test)
    assert allocated == sorted(clustering.labels)


def test_members_follow_their_cluster():
    clustering = _clustering()
    split = homology_split(clustering, test_fraction=0.25, val_fraction=0.0, seed=0)
    for fold in (split.train, split.test):
        clusters = {clustering.labels[p] for p in fold}
        for cluster in clusters:
            members = [p for p, c in clustering.labels.items() if c == cluster]
            assert all(m in fold for m in members)


def test_homology_split_is_deterministic():
    clustering = _clustering()
    first = homology_split(clustering, test_fraction=0.25, seed=7)
    second = homology_split(clustering, test_fraction=0.25, seed=7)
    assert first.test == second.test


def test_homology_split_records_realised_fractions():
    """Clusters move as units, so the realised fractions differ from those
    requested; reporting them prevents a silent mismatch."""
    split = homology_split(_clustering(), test_fraction=0.25, val_fraction=0.0, seed=0)
    assert "realised_test_fraction" in split.extra
    assert "requested_test_fraction" in split.extra


@pytest.mark.parametrize("n_clusters", [1, 2])
def test_too_few_clusters_is_refused(n_clusters):
    with pytest.raises(DataError, match=f"at least {MIN_CLUSTERS} clusters"):
        homology_split(_clustering(n_clusters=n_clusters), test_fraction=0.25)


def test_refusal_explains_what_it_means():
    with pytest.raises(DataError) as info:
        homology_split(_clustering(n_clusters=1), test_fraction=0.25)
    assert "unseen family" in str(info.value)


def test_empty_clustering_is_refused():
    empty = Clustering(labels={}, identity_threshold=0.3, coverage_threshold=0.5,
                       backend="test")
    with pytest.raises(DataError, match="no sequences"):
        homology_split(empty)


def test_fractions_that_leave_no_training_data_are_refused():
    with pytest.raises(DataError, match="leaving nothing to train"):
        homology_split(_clustering(n_clusters=4), test_fraction=0.5, val_fraction=0.5)


def test_leakage_guard_accepts_a_clean_split():
    clustering = _clustering()
    split = homology_split(clustering, test_fraction=0.25, seed=0)
    report = assert_no_homology_leakage(split, clustering)
    assert report["passed"]


def test_leakage_guard_rejects_a_shared_cluster():
    clustering = _clustering()
    split = homology_split(clustering, test_fraction=0.25, val_fraction=0.0, seed=0)
    corrupted = Split(
        name=split.name, train=list(split.train[1:]), val=[],
        test=list(split.test) + [split.train[0]], strategy=split.strategy,
        train_clusters=list(split.train_clusters),
        test_clusters=list(split.test_clusters),
    )
    with pytest.raises(LeakageError, match="appear in both"):
        assert_no_homology_leakage(corrupted, clustering)


def test_leakage_guard_message_names_the_consequence():
    clustering = _clustering()
    split = homology_split(clustering, test_fraction=0.25, val_fraction=0.0, seed=0)
    corrupted = Split(
        name=split.name, train=list(split.train[1:]), val=[],
        test=list(split.test) + [split.train[0]], strategy=split.strategy,
    )
    with pytest.raises(LeakageError) as info:
        assert_no_homology_leakage(corrupted, clustering)
    assert "optimistic" in str(info.value)


def test_leakage_guard_verifies_against_sequences(family_sequences):
    """The check that matters: it does not trust the clustering."""
    clustering = cluster_connected_components(family_sequences, identity=0.30)
    split = homology_split(clustering, test_fraction=0.4, val_fraction=0.0, seed=0)
    report = assert_no_homology_leakage(
        split, clustering, sequences=family_sequences
    )
    check = next(c for c in report["checks"] if c["check"] == "max train-test identity")
    assert check["max_identity"] <= 0.30 + 1e-9


def test_sequence_verification_catches_what_clustering_missed(family_sequences):
    """A hand-built split that puts two family members on opposite sides must
    be rejected even though no cluster id is shared."""
    names = sorted(family_sequences)
    # F0M0 and F0M1 are in the same family, so they are homologous.
    split = Split(
        name="bad", train=["F0M0"], val=[], test=["F0M1"],
        strategy="homology_cluster",
        train_clusters=[0], test_clusters=[1],
    )
    with pytest.raises(LeakageError, match="share"):
        assert_no_homology_leakage(split, None, sequences=family_sequences,
                                   max_identity=0.30)


def test_random_split_records_shared_clusters():
    clustering = _clustering()
    split = random_split(
        sorted(clustering.labels), clustering, test_fraction=0.25, seed=0
    )
    assert split.extra["shared_clusters_train_test"] > 0
    assert split.strategy == "random_protein"


def test_random_split_marks_itself_as_comparison_only():
    clustering = _clustering()
    split = random_split(sorted(clustering.labels), clustering, seed=0)
    assert "comparison only" in split.extra["note"]


def test_residue_split_puts_every_protein_in_every_fold():
    split = residue_split(["A", "B", "C"], seed=0)
    assert split.train == split.val == split.test
    assert split.extra["residue_level"] is True
    assert "most severe leak" in split.extra["note"]


def test_cluster_kfold_holds_out_disjoint_clusters():
    folds = cluster_kfold(_clustering(n_clusters=6), n_folds=3, seed=0)
    assert len(folds) == 3
    held = [set(f.test_clusters) for f in folds]
    for index, clusters in enumerate(held):
        assert clusters
        for other in held[index + 1:]:
            assert not clusters & other


def test_cluster_kfold_covers_every_cluster():
    clustering = _clustering(n_clusters=6)
    folds = cluster_kfold(clustering, n_folds=3, seed=0)
    covered = set()
    for fold in folds:
        covered |= set(fold.test_clusters)
    assert covered == set(clustering.labels.values())


def test_cluster_kfold_rejects_more_folds_than_clusters():
    with pytest.raises(DataError, match="only 3 clusters"):
        cluster_kfold(_clustering(n_clusters=3), n_folds=5)


def test_cluster_kfold_rejects_one_fold():
    with pytest.raises(DataError, match="at least 2"):
        cluster_kfold(_clustering(), n_folds=1)


def test_label_balance_reports_the_positive_rate():
    import numpy as np

    labels = {
        "A": np.array([True, False, False, False]),
        "B": np.array([False, False, False, False]),
    }
    split = Split(name="m", train=["A"], val=[], test=["B"], strategy="homology_cluster")
    balance = label_balance(labels, split)
    assert balance["train_positive_rate"] == pytest.approx(0.25)
    assert balance["test_positive_rate"] == pytest.approx(0.0)


def test_label_balance_handles_an_empty_fold():
    import numpy as np

    labels = {"A": np.array([True, False])}
    split = Split(name="m", train=["A"], val=[], test=[], strategy="homology_cluster")
    balance = label_balance(labels, split)
    assert balance["test_n_residues"] == 0
    assert balance["test_positive_rate"] is None
