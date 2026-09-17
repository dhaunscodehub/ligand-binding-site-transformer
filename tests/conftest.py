"""Shared fixtures.

Synthetic fixtures are session-scoped: generating them is cheap, but the
pairwise alignments in the homology tests are not, so anything reusable is
built once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def helix_pdb() -> str:
    """A valid PDB file: helix, one ATP, plus water and sulfate to be dropped."""
    from bindsite.testing import synthetic_pdb

    return synthetic_pdb(n_residues=40, ligand="ATP", ligand_centre=(2.3, 0.0, 20.0))


@pytest.fixture(scope="session")
def features_signal():
    """Feature records whose labels are a clean function of the features."""
    from bindsite.testing import synthetic_features

    return synthetic_features(n_proteins=24, signal=1.0, seed=0)


@pytest.fixture(scope="session")
def features_noise():
    """Feature records whose labels carry no information about the features."""
    from bindsite.testing import synthetic_features

    return synthetic_features(n_proteins=24, signal=0.0, seed=1)


@pytest.fixture(scope="session")
def family_sequences():
    """Sequences in five known families of four members each."""
    from bindsite.testing import synthetic_families

    return synthetic_families(n_families=5, per_family=4, length=120, seed=0)


@pytest.fixture(scope="session")
def family_clustering(family_sequences):
    from bindsite.homology import cluster_connected_components

    return cluster_connected_components(family_sequences, identity=0.30)


@pytest.fixture(scope="session")
def family_features():
    """Homologous families sharing sequence, structure and binding site.

    The fixture the leakage controls need: independent random proteins each
    form their own cluster, so a homology split and a random split coincide
    and nothing can leak.
    """
    from bindsite.testing import synthetic_family_features

    return synthetic_family_features(n_families=12, per_family=5, seed=0)


@pytest.fixture(scope="session")
def family_feature_clustering(family_features):
    from bindsite.homology import cluster_connected_components

    _, sequences = family_features
    return cluster_connected_components(sequences, identity=0.30)
