"""Per-residue features: sequence identity, physicochemistry and backbone geometry.

Three feature blocks, kept separate because the model's cross-attention treats
sequence and structure as distinct modalities and the ablation in
``validation`` needs to switch them independently.

**Sequence** — one-hot residue identity (20 amino acids + unknown).

**Physicochemical** — published per-residue scales. Every scale here is cited;
none is invented or tuned. Values are normalised to roughly unit range so that
no single scale dominates the initial gradients purely through magnitude.

**Backbone geometry** — what can be computed from N, CA, C, CB alone:
relative solvent exposure by neighbour counting, local curvature, and the
direction the side chain points relative to the protein's centre of mass.

The geometric features deliberately avoid any use of ligand coordinates.
That sounds obvious, but it is the easiest way to leak in this task: a pocket
is a concave region, and a "concavity" feature computed with the ligand still
in the structure measures where the ligand is, not where a pocket is. The
ligand is stripped before any feature is computed, and a test asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .exceptions import DataError
from .structure import ProteinStructure, Residue

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNKNOWN_INDEX = len(AMINO_ACIDS)
N_SEQUENCE_FEATURES = len(AMINO_ACIDS) + 1

# Kyte & Doolittle hydropathy, J Mol Biol 157:105 (1982). Range -4.5..4.5.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# Maximum accessible surface area, Tien et al., PLoS One 8:e80635 (2013),
# theoretical values. Used to turn a neighbour count into a relative measure.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0,
    "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0,
    "M": 224.0, "F": 240.0, "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0,
    "Y": 263.0, "V": 174.0,
}

# Formal charge at physiological pH. His is given 0.1: its pKa near 6.0 means
# it is mostly neutral at pH 7.4 but titratable, and assigning a full +1
# would overstate it.
CHARGE = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
    **{aa: 0.0 for aa in "ACFGILMNPQSTVWY"},
}

# Side-chain volume in cubic angstroms, Zamyatnin, Prog Biophys Mol Biol
# 24:107 (1974).
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# Hydrogen-bond donor and acceptor counts on the side chain only. Backbone
# donors/acceptors are identical for every residue and carry no signal.
HBOND_DONORS = {
    "R": 3, "K": 1, "W": 1, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1,
    "C": 1, **{aa: 0 for aa in "ADEFGILMPV"},
}
HBOND_ACCEPTORS = {
    "D": 2, "E": 2, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1,
    **{aa: 0 for aa in "ACFGIKLMPRVW"},
}

# Aromaticity: aromatic side chains dominate ligand stacking interactions.
AROMATIC = frozenset("FWYH")

# Radius within which neighbouring CB atoms are counted for a burial estimate.
# 10 A is the conventional half-sphere-exposure radius (Hamelryck,
# Proteins 59:38, 2005).
NEIGHBOUR_RADIUS = 10.0

PHYSICOCHEMICAL_NAMES = (
    "hydropathy", "charge", "volume", "max_asa", "hbond_donors",
    "hbond_acceptors", "aromatic", "is_glycine", "is_proline",
)
N_PHYSICOCHEMICAL_FEATURES = len(PHYSICOCHEMICAL_NAMES)

GEOMETRIC_NAMES = (
    "neighbour_count", "relative_burial", "depth_from_centroid",
    "sidechain_outwardness", "local_curvature", "ca_ca_prev", "ca_ca_next",
    "terminal_proximity",
)
N_GEOMETRIC_FEATURES = len(GEOMETRIC_NAMES)


@dataclass
class ResidueFeatures:
    """Feature matrices for one protein, one row per residue."""

    identifier: str
    sequence: str
    sequence_features: np.ndarray        # (L, 21)
    physicochemical: np.ndarray          # (L, 9)
    geometric: np.ndarray                # (L, 8)
    labels: np.ndarray                   # (L,) bool
    residue_keys: list[tuple[str, int, str]] = field(default_factory=list)
    # Pairwise CA-CA distances in angstroms, (L, L). Supplied here rather than
    # recomputed in the model so the structural attention bias and the
    # geometric features come from one source of truth.
    distance_matrix: np.ndarray | None = None

    @property
    def n_residues(self) -> int:
        return int(self.sequence_features.shape[0])

    @property
    def sequence_block(self) -> np.ndarray:
        """Sequence + physicochemical, the "sequence modality"."""
        return np.hstack([self.sequence_features, self.physicochemical])

    @property
    def structure_block(self) -> np.ndarray:
        """Backbone geometry, the "structure modality"."""
        return self.geometric

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier, "n_residues": self.n_residues,
            "n_binding": int(self.labels.sum()),
            "binding_fraction": float(self.labels.mean()) if self.n_residues else 0.0,
            "n_sequence_features": int(self.sequence_features.shape[1]),
            "n_physicochemical_features": int(self.physicochemical.shape[1]),
            "n_geometric_features": int(self.geometric.shape[1]),
        }


def sequence_one_hot(sequence: str) -> np.ndarray:
    """One-hot encode a sequence, with a dedicated column for unknowns."""
    out = np.zeros((len(sequence), N_SEQUENCE_FEATURES), dtype=np.float32)
    for position, residue in enumerate(sequence):
        out[position, AA_INDEX.get(residue, UNKNOWN_INDEX)] = 1.0
    return out


def physicochemical_features(sequence: str) -> np.ndarray:
    """Published per-residue scales, normalised to roughly unit range.

    Normalisation is by a fixed published constant rather than by dataset
    statistics. Using dataset means would make the features depend on which
    proteins happen to be in the split, so the same residue would get
    different features in train and test.
    """
    out = np.zeros((len(sequence), N_PHYSICOCHEMICAL_FEATURES), dtype=np.float32)
    for position, residue in enumerate(sequence):
        out[position] = (
            KYTE_DOOLITTLE.get(residue, 0.0) / 4.5,
            CHARGE.get(residue, 0.0),
            VOLUME.get(residue, 140.0) / 227.8,
            MAX_ASA.get(residue, 200.0) / 285.0,
            HBOND_DONORS.get(residue, 0) / 3.0,
            HBOND_ACCEPTORS.get(residue, 0) / 2.0,
            1.0 if residue in AROMATIC else 0.0,
            1.0 if residue == "G" else 0.0,
            1.0 if residue == "P" else 0.0,
        )
    return out


def _pseudo_cb(residue: Residue) -> np.ndarray | None:
    """CB position, reconstructed from the backbone for glycine.

    Glycine has no CB, so the side-chain direction features would be undefined
    for every glycine. The standard reconstruction places a virtual CB from N,
    CA and C using ideal tetrahedral geometry, which keeps glycine comparable
    to every other residue instead of being a special case with zeros.
    """
    if residue.cb is not None:
        return residue.cb
    if not residue.has_backbone:
        return residue.ca
    n, ca, c = residue.n, residue.ca, residue.c
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    # Coefficients for ideal CB placement (Gly virtual CB), standard values.
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def geometric_features(residues: Sequence[Residue]) -> np.ndarray:
    """Backbone-derived geometry. Never sees ligand coordinates.

    ``neighbour_count`` and ``relative_burial`` are the informative pair: a
    binding pocket is a concave region, which shows up as a residue that is
    partially buried yet still solvent-accessible. Fully buried core residues
    and fully exposed surface residues are both usually non-binding, so a
    monotonic burial measure alone would not separate them — which is why the
    raw count and the size-normalised version are both provided.
    """
    count = len(residues)
    out = np.zeros((count, N_GEOMETRIC_FEATURES), dtype=np.float32)
    if count == 0:
        return out

    cb_positions = np.array(
        [
            (_pseudo_cb(r) if _pseudo_cb(r) is not None else np.zeros(3))
            for r in residues
        ],
        dtype=float,
    )
    ca_positions = np.array(
        [(r.ca if r.ca is not None else cb_positions[i]) for i, r in enumerate(residues)],
        dtype=float,
    )
    centroid = ca_positions.mean(axis=0)

    # Pairwise CB distances for the neighbour count.
    deltas = cb_positions[:, None, :] - cb_positions[None, :, :]
    distances = np.sqrt((deltas ** 2).sum(axis=-1))
    np.fill_diagonal(distances, np.inf)
    neighbours = (distances < NEIGHBOUR_RADIUS).sum(axis=1)

    radial = np.sqrt(((ca_positions - centroid) ** 2).sum(axis=1))
    max_radial = radial.max() if radial.max() > 0 else 1.0

    for index, residue in enumerate(residues):
        # Side-chain direction relative to outward radial direction: +1 points
        # away from the core, -1 points inward. Pocket-lining residues
        # frequently point their side chains inward into the cavity.
        outward = ca_positions[index] - centroid
        norm = np.linalg.norm(outward)
        sidechain = cb_positions[index] - ca_positions[index]
        sidechain_norm = np.linalg.norm(sidechain)
        outwardness = (
            float(np.dot(outward / norm, sidechain / sidechain_norm))
            if norm > 1e-6 and sidechain_norm > 1e-6 else 0.0
        )

        previous = ca_positions[index - 1] if index > 0 else ca_positions[index]
        following = (
            ca_positions[index + 1] if index + 1 < count else ca_positions[index]
        )
        # Angle at this CA between its neighbours: a proxy for local backbone
        # curvature, which distinguishes loops (where pockets usually sit)
        # from regular helix and strand.
        left, right = previous - ca_positions[index], following - ca_positions[index]
        left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
        curvature = (
            float(np.dot(left / left_norm, right / right_norm))
            if left_norm > 1e-6 and right_norm > 1e-6 else 0.0
        )

        out[index] = (
            neighbours[index] / 50.0,
            # Normalised by the residue's own maximum ASA so a large residue
            # with many neighbours is not automatically "more buried".
            neighbours[index] / (MAX_ASA.get(residue.one_letter, 200.0) / 10.0),
            radial[index] / max_radial,
            outwardness,
            curvature,
            float(np.linalg.norm(ca_positions[index] - previous)) / 4.0,
            float(np.linalg.norm(following - ca_positions[index])) / 4.0,
            # Distance to the nearer chain terminus, normalised. Termini are
            # flexible and rarely form pockets.
            min(index, count - 1 - index) / max(count / 2.0, 1.0),
        )
    return out


def featurise(structure: ProteinStructure) -> ResidueFeatures:
    """Compute every feature block for one labelled structure.

    Features are computed from the polymer only. The ligand is used to derive
    labels and for nothing else — a geometric feature computed with the ligand
    present would encode the answer.
    """
    if structure.n_residues == 0:
        raise DataError(f"{structure.identifier}: no residues to featurise")

    sequence = structure.sequence
    return ResidueFeatures(
        identifier=structure.identifier,
        sequence=sequence,
        sequence_features=sequence_one_hot(sequence),
        physicochemical=physicochemical_features(sequence),
        geometric=geometric_features(structure.residues),
        labels=structure.labels,
        residue_keys=[r.key for r in structure.residues],
        distance_matrix=ca_distance_matrix(structure.residues),
    )


def ca_distance_matrix(residues: Sequence[Residue]) -> np.ndarray:
    """Pairwise CA-CA distances in angstroms.

    A residue with no resolved CA falls back to its pseudo-CB so the matrix
    stays complete; a missing row would make the attention bias undefined for
    that residue and silently exclude it from structural context.
    """
    if not residues:
        return np.zeros((0, 0))
    coords = np.array(
        [
            (
                r.ca if r.ca is not None
                else (_pseudo_cb(r) if _pseudo_cb(r) is not None else np.zeros(3))
            )
            for r in residues
        ],
        dtype=float,
    )
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((deltas ** 2).sum(axis=-1))


def feature_names() -> dict[str, tuple[str, ...]]:
    """Names of every feature, for interpretation and for tests."""
    return {
        "sequence": tuple(AMINO_ACIDS) + ("X",),
        "physicochemical": PHYSICOCHEMICAL_NAMES,
        "geometric": GEOMETRIC_NAMES,
    }
