"""Per-residue features, and the guarantee that none of them sees the ligand."""

from __future__ import annotations

import numpy as np
import pytest

from bindsite.exceptions import DataError
from bindsite.features import (
    AMINO_ACIDS, KYTE_DOOLITTLE, MAX_ASA, N_GEOMETRIC_FEATURES,
    N_PHYSICOCHEMICAL_FEATURES, N_SEQUENCE_FEATURES, ca_distance_matrix,
    feature_names, featurise, geometric_features, physicochemical_features,
    sequence_one_hot,
)
from bindsite.structure import structure_from_pdb_text
from bindsite.testing import synthetic_pdb


def test_one_hot_is_one_per_residue():
    encoded = sequence_one_hot("ACDEF")
    assert encoded.shape == (5, N_SEQUENCE_FEATURES)
    assert np.array_equal(encoded.sum(axis=1), np.ones(5))


def test_unknown_residue_uses_the_dedicated_column():
    encoded = sequence_one_hot("X")
    assert encoded[0, -1] == 1.0
    assert encoded[0, :-1].sum() == 0.0


def test_every_amino_acid_has_a_distinct_column():
    encoded = sequence_one_hot(AMINO_ACIDS)
    assert np.array_equal(encoded[:, : len(AMINO_ACIDS)], np.eye(len(AMINO_ACIDS)))


def test_physicochemical_scales_cover_every_amino_acid():
    for scale in (KYTE_DOOLITTLE, MAX_ASA):
        assert set(scale) == set(AMINO_ACIDS)


def test_physicochemical_shape_and_finiteness():
    out = physicochemical_features(AMINO_ACIDS)
    assert out.shape == (20, N_PHYSICOCHEMICAL_FEATURES)
    assert np.isfinite(out).all()


def test_physicochemical_values_are_normalised():
    """Fixed published constants, so the same residue gets the same value in
    every split — dataset statistics would make features split-dependent."""
    out = physicochemical_features(AMINO_ACIDS)
    assert np.abs(out).max() <= 1.5


def test_hydrophobic_and_charged_residues_differ():
    out = physicochemical_features("IK")
    assert out[0, 0] > 0  # isoleucine, hydrophobic
    assert out[1, 0] < 0  # lysine, hydrophilic
    assert out[1, 1] > 0  # lysine, positive


def test_glycine_and_proline_are_flagged():
    out = physicochemical_features("GPA")
    assert out[0, 7] == 1.0
    assert out[1, 8] == 1.0
    assert out[2, 7] == 0.0 and out[2, 8] == 0.0


def test_aromatics_are_flagged():
    out = physicochemical_features("FWYHA")
    assert list(out[:, 6]) == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_geometric_features_shape(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    out = geometric_features(structure.residues)
    assert out.shape == (structure.n_residues, N_GEOMETRIC_FEATURES)
    assert np.isfinite(out).all()


def test_geometric_features_on_an_empty_chain():
    assert geometric_features([]).shape == (0, N_GEOMETRIC_FEATURES)


def test_glycine_gets_a_virtual_cb():
    """Without a reconstructed CB every glycine would have undefined
    side-chain direction and become a special case of zeros."""
    pdb = (
        "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  GLY A   1       1.458   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  C   GLY A   1       2.009   1.420   0.000  1.00 20.00           C\n"
        "ATOM      4  N   ALA A   2       3.300   1.600   0.000  1.00 20.00           N\n"
        "ATOM      5  CA  ALA A   2       4.000   2.800   0.000  1.00 20.00           C\n"
        "ATOM      6  C   ALA A   2       5.500   2.700   0.000  1.00 20.00           C\n"
        "ATOM      7  CB  ALA A   2       3.600   3.800   1.000  1.00 20.00           C\n"
    )
    residues, _, _ = __import__(
        "bindsite.structure", fromlist=["parse_pdb"]
    ).parse_pdb(pdb, "T")
    out = geometric_features(residues)
    assert np.isfinite(out).all()
    # The glycine's side-chain direction is defined, not zero-filled.
    assert out[0, 3] != 0.0


def test_ca_distance_matrix_is_symmetric_with_zero_diagonal(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    matrix = ca_distance_matrix(structure.residues)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_adjacent_ca_distance_is_physical():
    """A real structure's consecutive CA atoms sit ~3.8 A apart."""
    from bindsite.structure import load_structure
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data/pdb/1STP.pdb"
    if not path.is_file():
        pytest.skip("reference structure not cached")
    structure = load_structure(path, identifier="1STP")
    matrix = ca_distance_matrix(structure.residues)
    assert np.mean(np.diag(matrix, 1)) == pytest.approx(3.8, abs=0.15)


def test_featurise_produces_every_block(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    features = featurise(structure)
    assert features.sequence_features.shape[1] == N_SEQUENCE_FEATURES
    assert features.physicochemical.shape[1] == N_PHYSICOCHEMICAL_FEATURES
    assert features.geometric.shape[1] == N_GEOMETRIC_FEATURES
    assert features.labels.size == structure.n_residues
    assert features.distance_matrix is not None


def test_modality_blocks_partition_the_features(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    features = featurise(structure)
    assert features.sequence_block.shape[1] == (
        N_SEQUENCE_FEATURES + N_PHYSICOCHEMICAL_FEATURES
    )
    assert features.structure_block.shape[1] == N_GEOMETRIC_FEATURES


def test_featurise_rejects_an_empty_structure():
    from bindsite.structure import ProteinStructure

    empty = ProteinStructure(identifier="E", chain="A", residues=[])
    with pytest.raises(DataError, match="no residues"):
        featurise(empty)


def test_feature_names_match_the_dimensions():
    names = feature_names()
    assert len(names["sequence"]) == N_SEQUENCE_FEATURES
    assert len(names["physicochemical"]) == N_PHYSICOCHEMICAL_FEATURES
    assert len(names["geometric"]) == N_GEOMETRIC_FEATURES


def test_features_do_not_see_the_ligand():
    """The central guarantee: a pocket is a concavity, so any geometric
    feature computed with the ligand present would partly encode the answer."""
    pdb = synthetic_pdb(n_residues=40, ligand="ATP", ligand_centre=(2.3, 0.0, 20.0))
    with_ligand = featurise(structure_from_pdb_text(pdb, "with"))
    stripped = "\n".join(
        line for line in pdb.splitlines() if not line.startswith("HETATM")
    )
    without = featurise(
        structure_from_pdb_text(stripped, "without", require_ligand=False)
    )

    assert np.array_equal(with_ligand.sequence_features, without.sequence_features)
    assert np.array_equal(with_ligand.physicochemical, without.physicochemical)
    assert np.array_equal(with_ligand.geometric, without.geometric)
    assert np.array_equal(with_ligand.distance_matrix, without.distance_matrix)
    # The labels, and only the labels, differ.
    assert with_ligand.labels.sum() > 0
    assert without.labels.sum() == 0


def test_featurise_record_serialises(helix_pdb):
    features = featurise(structure_from_pdb_text(helix_pdb, "T"))
    record = features.to_dict()
    assert record["n_residues"] == features.n_residues
    assert 0.0 <= record["binding_fraction"] <= 1.0
