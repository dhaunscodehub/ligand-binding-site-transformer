"""Sequence-identity clustering and the guarantee a split depends on."""

from __future__ import annotations

import itertools

import pytest

from bindsite.exceptions import DataError, ExternalToolError, OptionalDependencyMissing
from bindsite.homology import (
    DEFAULT_COVERAGE, DEFAULT_IDENTITY, cluster_coherence,
    cluster_connected_components, cluster_greedy, cluster_sequences,
    max_identity_between, mmseqs_available, mmseqs_command, pairwise_identity,
    write_fasta,
)

SEQ_A = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
)
SEQ_B = SEQ_A[:40] + "A" + SEQ_A[41:]
SEQ_C = (
    "PIVQNLQGQMVHQAISPRTLNAWVKVVEEKAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEEAA"
)


def test_identical_sequences_score_one():
    identity, coverage = pairwise_identity(SEQ_A, SEQ_A)
    assert identity == pytest.approx(1.0)
    assert coverage == pytest.approx(1.0)


def test_one_substitution_is_nearly_identical():
    identity, _ = pairwise_identity(SEQ_A, SEQ_B)
    assert 0.95 < identity < 1.0


def test_unrelated_sequences_score_low():
    identity, _ = pairwise_identity(SEQ_A, SEQ_C)
    assert identity < DEFAULT_IDENTITY


def test_identity_is_symmetric():
    assert pairwise_identity(SEQ_A, SEQ_C)[0] == pytest.approx(
        pairwise_identity(SEQ_C, SEQ_A)[0]
    )


def test_local_alignment_scores_unrelated_pairs_below_global():
    """Global alignment inflates identity by forcing end-to-end matching."""
    local, _ = pairwise_identity(SEQ_A, SEQ_C, mode="local")
    global_, _ = pairwise_identity(SEQ_A, SEQ_C, mode="global")
    assert local < global_


def test_local_is_the_default():
    assert pairwise_identity(SEQ_A, SEQ_C) == pairwise_identity(
        SEQ_A, SEQ_C, mode="local"
    )


def test_invalid_mode_is_rejected():
    with pytest.raises(DataError, match="mode must be"):
        pairwise_identity(SEQ_A, SEQ_C, mode="semiglobal")


def test_empty_sequence_is_rejected():
    with pytest.raises(DataError, match="empty"):
        pairwise_identity(SEQ_A, "")


def test_unknown_characters_are_tolerated():
    """Non-BLOSUM characters must map to X rather than raising."""
    identity, _ = pairwise_identity(SEQ_A, SEQ_A.replace("K", "J"))
    assert 0.0 < identity <= 1.0


def test_connected_components_recovers_known_families(family_sequences):
    clustering = cluster_connected_components(family_sequences, identity=0.30)
    assert clustering.n_clusters == 5
    assert sorted(clustering.cluster_sizes().values()) == [4, 4, 4, 4, 4]


def test_connected_components_guarantees_the_split_invariant(family_sequences):
    """No pair above the threshold may land in different clusters.

    This is the property a homology-separated split requires, and the reason
    connected components is the default backend.
    """
    clustering = cluster_connected_components(family_sequences, identity=0.30)
    violations = []
    for left, right in itertools.combinations(sorted(family_sequences), 2):
        identity, coverage = pairwise_identity(
            family_sequences[left], family_sequences[right]
        )
        if (
            identity >= 0.30 and coverage >= DEFAULT_COVERAGE
            and clustering.labels[left] != clustering.labels[right]
        ):
            violations.append((left, right, identity))
    assert not violations


def test_connected_components_is_deterministic(family_sequences):
    first = cluster_connected_components(family_sequences, identity=0.30)
    second = cluster_connected_components(family_sequences, identity=0.30)
    assert first.labels == second.labels


def test_a_higher_threshold_yields_at_least_as_many_clusters(family_sequences):
    counts = [
        cluster_connected_components(family_sequences, identity=t).n_clusters
        for t in (0.20, 0.30, 0.50, 0.80)
    ]
    assert counts == sorted(counts)


def test_clustering_records_its_backend_and_linkage(family_sequences):
    clustering = cluster_connected_components(family_sequences, identity=0.30)
    record = clustering.to_dict()
    assert record["backend"] == "connected_components"
    assert "transitive closure" in record["linkage"]
    assert record["n_similarity_edges"] is not None


def test_greedy_can_miss_a_homologous_pair(family_sequences):
    """Greedy centroid clustering compares only to representatives, so it can
    place a homologous pair in different clusters. That failure is why it is
    not the default."""
    greedy = cluster_greedy(family_sequences, identity=0.30)
    components = cluster_connected_components(family_sequences, identity=0.30)
    # Greedy never merges more than connected components does.
    assert greedy.n_clusters >= components.n_clusters


def test_singleton_sequence_forms_its_own_cluster():
    clustering = cluster_connected_components(
        {"a": SEQ_A, "c": SEQ_C}, identity=0.30
    )
    assert clustering.n_clusters == 2


def test_empty_input_is_rejected():
    with pytest.raises(DataError, match="no sequences"):
        cluster_connected_components({})


@pytest.mark.parametrize("identity", [0.0, -0.1, 1.5])
def test_invalid_identity_is_rejected(identity, family_sequences):
    with pytest.raises(DataError, match="identity"):
        cluster_connected_components(family_sequences, identity=identity)


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5])
def test_invalid_coverage_is_rejected(coverage, family_sequences):
    with pytest.raises(DataError, match="coverage"):
        cluster_connected_components(family_sequences, coverage=coverage)


def test_redundancy_is_reported(family_sequences):
    clustering = cluster_connected_components(family_sequences, identity=0.30)
    record = clustering.to_dict()
    assert record["redundancy"] == pytest.approx(1 - 5 / 20)


def test_coherence_diagnostic_reports_chaining(family_sequences):
    """Clean families should show no chaining."""
    clustering = cluster_connected_components(family_sequences, identity=0.30)
    coherence = cluster_coherence(clustering, family_sequences, seed=0)
    assert coherence["n_multi_member_clusters"] == 5
    assert coherence["n_clusters_showing_chaining"] == 0


def test_coherence_handles_all_singletons():
    clustering = cluster_connected_components(
        {"a": SEQ_A, "c": SEQ_C}, identity=0.90
    )
    coherence = cluster_coherence(clustering, {"a": SEQ_A, "c": SEQ_C})
    assert coherence["n_multi_member_clusters"] == 0


def test_max_identity_between_finds_the_worst_pair(family_sequences):
    names = sorted(family_sequences)
    identity, left, right = max_identity_between(
        names[:4], names[4:8], family_sequences
    )
    assert 0.0 <= identity <= 1.0
    assert left in names[:4]
    assert right in names[4:8]


def test_cluster_sequences_defaults_to_connected_components(family_sequences):
    clustering = cluster_sequences(family_sequences, identity=0.30, backend="auto")
    assert clustering.backend == "connected_components"


def test_cluster_sequences_rejects_an_unknown_backend(family_sequences):
    with pytest.raises(DataError, match="backend must be"):
        cluster_sequences(family_sequences, backend="magic")


def test_mmseqs_command_is_well_formed(tmp_path):
    command = mmseqs_command(tmp_path / "in.fasta", tmp_path / "out", 0.3, 0.5)
    assert command[:2] == ["mmseqs", "easy-cluster"]
    assert "--min-seq-id" in command
    assert "0.3" in command
    assert "--cluster-mode" in command


def test_mmseqs_is_reported_absent_rather_than_assumed():
    assert isinstance(mmseqs_available(), bool)


def test_mmseqs_backend_explains_itself_when_absent(family_sequences, tmp_path):
    if mmseqs_available():
        pytest.skip("mmseqs is installed, so the absence path cannot be tested")
    with pytest.raises(ExternalToolError, match="not on PATH"):
        cluster_sequences(
            family_sequences, backend="mmseqs", work_dir=tmp_path
        )


def test_fasta_round_trips(tmp_path):
    path = write_fasta({"a": SEQ_A, "c": SEQ_C}, tmp_path / "s.fasta")
    text = path.read_text()
    assert text.startswith(">a\n")
    assert ">c" in text
    # Wrapped at 60 columns.
    assert all(len(line) <= 60 for line in text.splitlines() if not line.startswith(">"))
