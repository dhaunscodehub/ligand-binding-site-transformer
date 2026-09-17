"""Structure parsing, ligand curation and binding-residue labelling."""

from __future__ import annotations

import numpy as np
import pytest

from bindsite.exceptions import NoLigandError, StructureError
from bindsite.structure import (
    ADDITIVES, CONTACT_CUTOFF, EXCLUDED_HETERO, IONS, THREE_TO_ONE, WATER,
    curate_ligands, label_binding_residues, parse_pdb, structure_from_pdb_text,
)
from bindsite.testing import synthetic_pdb


def test_parses_the_expected_number_of_residues(helix_pdb):
    residues, het, counts = parse_pdb(helix_pdb, "T")
    assert len(residues) == 40
    assert counts  # het groups were seen


def test_backbone_atoms_are_assigned(helix_pdb):
    residues, _, _ = parse_pdb(helix_pdb, "T")
    assert all(r.has_backbone for r in residues)
    assert all(r.cb is not None for r in residues)


def test_sequence_is_recovered(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    assert structure.sequence == "A" * 40


def test_hydrogens_are_dropped():
    """Most crystal structures have no hydrogens; including them where
    present would make contacts depend on the depositor."""
    pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  HA  ALA A   1       1.500   1.000   0.000  1.00 20.00           H\n"
    )
    residues, _, _ = parse_pdb(pdb, "T")
    assert residues[0].atom_coords.shape[0] == 2


def test_only_the_first_model_is_read():
    """An NMR ensemble's later models are conformers of the same molecule."""
    pdb = (
        "MODEL        1\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      2  CA  ALA A   2      10.000   0.000   0.000  1.00 20.00           C\n"
        "ENDMDL\n"
    )
    residues, _, _ = parse_pdb(pdb, "T")
    assert len(residues) == 1


def test_altlocs_resolve_to_highest_occupancy():
    pdb = (
        "ATOM      1  CA AALA A   1       0.000   0.000   0.000  0.30 20.00           C\n"
        "ATOM      2  CA BALA A   1       5.000   0.000   0.000  0.70 20.00           C\n"
    )
    residues, _, _ = parse_pdb(pdb, "T")
    assert residues[0].atom_coords.shape[0] == 1
    assert residues[0].ca[0] == pytest.approx(5.0)


def test_truncated_record_is_rejected():
    with pytest.raises(StructureError, match="truncated"):
        parse_pdb("ATOM      1  CA  ALA A   1    \n", "T")


def test_unparseable_coordinates_are_rejected():
    with pytest.raises(StructureError, match="unparseable"):
        parse_pdb(
            "ATOM      1  CA  ALA A   1       abc.def   0.000   0.000  1.00 20.00\n",
            "T",
        )


def test_no_polymer_is_rejected():
    with pytest.raises(StructureError, match="no standard polymer"):
        parse_pdb("HETATM    1  O   HOH A   1       0.000   0.000   0.000\n", "T")


def test_water_is_never_a_ligand(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    assert "HOH" in structure.excluded_hetero
    assert all(l.residue_name != "HOH" for l in structure.ligands)


def test_sulfate_is_never_a_ligand(helix_pdb):
    """A crystallisation additive contact is not a binding site."""
    structure = structure_from_pdb_text(helix_pdb, "T")
    assert "SO4" in structure.excluded_hetero
    assert all(l.residue_name != "SO4" for l in structure.ligands)


def test_the_real_ligand_is_kept(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    assert [l.residue_name for l in structure.ligands] == ["ATP"]


@pytest.mark.parametrize("name", ["HOH", "SO4", "GOL", "EDO", "ZN", "NA", "CL", "PEG"])
def test_solvent_ions_and_additives_are_in_the_exclusion_list(name):
    assert name in EXCLUDED_HETERO


def test_exclusion_categories_are_disjoint_from_amino_acids():
    """A standard residue must never be treated as an excluded het group."""
    standard = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE "
                   "PRO SER THR TRP TYR VAL".split())
    assert not (EXCLUDED_HETERO & standard)


def test_water_ions_and_additives_are_distinct_sets():
    assert not (WATER & IONS)
    assert not (WATER & ADDITIVES)


def test_a_structure_with_only_solvent_is_rejected():
    pdb = synthetic_pdb(n_residues=20, ligand_atoms=0, include_water=True)
    with pytest.raises(NoLigandError, match="no het group survives"):
        structure_from_pdb_text(pdb, "T", require_ligand=True)


def test_rejection_message_explains_why():
    pdb = synthetic_pdb(n_residues=20, ligand_atoms=0)
    with pytest.raises(NoLigandError) as info:
        structure_from_pdb_text(pdb, "T")
    assert "crystallisation artefact" in str(info.value)


def test_small_het_groups_are_rejected_as_fragments():
    """Below the heavy-atom floor a het group cannot define a pocket."""
    pdb = synthetic_pdb(n_residues=20, ligand="LIG", ligand_atoms=3,
                        include_water=False, include_sulfate=False)
    with pytest.raises(NoLigandError):
        structure_from_pdb_text(pdb, "T", min_heavy_atoms=6)


def test_the_heavy_atom_floor_is_configurable():
    pdb = synthetic_pdb(n_residues=20, ligand="LIG", ligand_atoms=3,
                        ligand_centre=(2.3, 0.0, 10.0),
                        include_water=False, include_sulfate=False)
    structure = structure_from_pdb_text(pdb, "T", min_heavy_atoms=3)
    assert len(structure.ligands) == 1


def test_modified_residues_are_polymer_not_ligand():
    """MSE is a HETATM record but it is a selenomethionine, not a ligand."""
    pdb = synthetic_pdb(
        n_residues=20, ligand="ATP", ligand_atoms=8,
        ligand_centre=(2.3, 0.0, 10.0), include_modified_residue=True,
    )
    structure = structure_from_pdb_text(pdb, "T")
    assert all(l.residue_name != "MSE" for l in structure.ligands)


def test_modified_residues_map_to_their_parent_letter():
    assert THREE_TO_ONE["MSE"] == "M"
    assert THREE_TO_ONE["SEP"] == "S"
    assert THREE_TO_ONE["PTR"] == "Y"


def test_binding_residues_are_within_the_cutoff(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T", cutoff=4.0)
    for residue in structure.residues:
        if residue.is_binding:
            assert residue.min_ligand_distance <= 4.0
        else:
            assert residue.min_ligand_distance > 4.0


def test_a_larger_cutoff_labels_more_residues(helix_pdb):
    counts = [
        structure_from_pdb_text(helix_pdb, "T", cutoff=c).n_binding
        for c in (3.0, 4.0, 6.0, 8.0)
    ]
    assert counts == sorted(counts)


def test_distant_ligand_labels_nothing():
    pdb = synthetic_pdb(n_residues=20, ligand_centre=(500.0, 500.0, 500.0))
    structure = structure_from_pdb_text(pdb, "T")
    assert structure.n_binding == 0


def test_labelling_uses_all_heavy_atoms_not_just_ca():
    """A lysine side chain reaches ~6 A from its CA; a CA-only cutoff would
    make a label depend on residue size rather than on contacts."""
    from bindsite.structure import Ligand, Residue

    residue = Residue(
        name="LYS", one_letter="K", chain="A", residue_seq=1, insertion="",
        # CA far from the ligand, a side-chain atom close to it.
        atom_coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0]]),
        ca=np.array([0.0, 0.0, 0.0]),
    )
    ligand = Ligand(
        residue_name="LIG", chain="A", residue_seq=900, insertion="",
        coords=np.array([[0.0, 0.0, 11.0]]), elements=["C"],
    )
    label_binding_residues([residue], [ligand], cutoff=4.0)
    assert residue.is_binding
    assert residue.min_ligand_distance == pytest.approx(2.0)


def test_no_ligand_leaves_everything_unlabelled(helix_pdb):
    residues, _, _ = parse_pdb(helix_pdb, "T")
    label_binding_residues(residues, [], cutoff=4.0)
    assert not any(r.is_binding for r in residues)
    assert all(np.isinf(r.min_ligand_distance) for r in residues)


def test_non_positive_cutoff_is_rejected(helix_pdb):
    residues, _, _ = parse_pdb(helix_pdb, "T")
    with pytest.raises(ValueError, match="positive"):
        label_binding_residues(residues, [], cutoff=0.0)


def test_labels_are_recomputed_not_accumulated(helix_pdb):
    """Calling twice must not leave stale labels from the first call."""
    residues, het, _ = parse_pdb(helix_pdb, "T")
    ligands, _ = curate_ligands(het)
    label_binding_residues(residues, ligands, cutoff=8.0)
    many = sum(r.is_binding for r in residues)
    label_binding_residues(residues, ligands, cutoff=3.0)
    few = sum(r.is_binding for r in residues)
    assert few < many


def test_default_cutoff_is_the_conventional_value():
    assert CONTACT_CUTOFF == 4.0


def test_structure_record_reports_curation_mode(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    assert "exclusion lists" in structure.to_dict()["curation"]


def test_binding_fraction_is_consistent(helix_pdb):
    structure = structure_from_pdb_text(helix_pdb, "T")
    assert structure.binding_fraction == pytest.approx(
        structure.n_binding / structure.n_residues
    )
