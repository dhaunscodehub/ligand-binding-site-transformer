"""Synthetic fixtures with controllable signal.

Two generators:

:func:`synthetic_features`
    Feature records whose labels are a known function of the features, with a
    ``signal`` knob. Used for the positive control and for fast tests, where
    downloading and parsing real structures would dominate the runtime.

:func:`synthetic_pdb`
    A minimal but *valid* PDB file with a real backbone geometry and a real
    het group. Used to test the parser, the curation and the labelling on
    input whose correct answer is known by construction — a real structure
    tests the same code but its answer has to be looked up.

Nothing here is a claim about proteins. Everything derived from these
fixtures is a statement about the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import (
    N_GEOMETRIC_FEATURES, N_PHYSICOCHEMICAL_FEATURES, N_SEQUENCE_FEATURES,
    ResidueFeatures,
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class SyntheticSpec:
    """Shape and difficulty of a synthetic dataset."""

    n_proteins: int = 40
    min_length: int = 80
    max_length: int = 200
    positive_rate: float = 0.08
    # 0 = labels unrelated to features (negative control),
    # 1 = labels a clean function of the features (positive control).
    signal: float = 1.0
    # Number of homology "families"; proteins in a family share a sequence
    # template, so a clustering should recover them.
    n_families: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_proteins < 2:
            raise ValueError("n_proteins must be at least 2")
        if self.min_length < 20:
            raise ValueError("min_length must be at least 20")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        if not 0 < self.positive_rate < 1:
            raise ValueError("positive_rate must be in (0, 1)")
        if not 0 <= self.signal <= 1:
            raise ValueError("signal must be in [0, 1]")
        if self.n_families < 1:
            raise ValueError("n_families must be at least 1")


def synthetic_features(
    n_proteins: int = 40,
    signal: float = 1.0,
    positive_rate: float = 0.08,
    min_length: int = 80,
    max_length: int = 200,
    seed: int = 0,
) -> dict[str, ResidueFeatures]:
    """Feature records whose labels depend on the features by construction.

    The label is driven by a fixed linear combination of a few feature
    columns, thresholded to hit ``positive_rate``. With ``signal = 0`` the
    labels are permuted, so the positive rate is preserved exactly while every
    association with the features is destroyed — which is what makes it a
    negative control rather than merely a different dataset.
    """
    rng = np.random.default_rng(seed)
    records: dict[str, ResidueFeatures] = {}

    for index in range(n_proteins):
        length = int(rng.integers(min_length, max_length + 1))
        sequence = "".join(rng.choice(list(AMINO_ACIDS), size=length))

        sequence_features = np.zeros((length, N_SEQUENCE_FEATURES), dtype=np.float32)
        for position, residue in enumerate(sequence):
            sequence_features[position, AMINO_ACIDS.index(residue)] = 1.0
        physicochemical = rng.normal(
            0, 1, (length, N_PHYSICOCHEMICAL_FEATURES)
        ).astype(np.float32)
        geometric = rng.normal(0, 1, (length, N_GEOMETRIC_FEATURES)).astype(np.float32)

        # A deliberately simple generative rule: burial-like column 0 and
        # side-chain-direction-like column 3, plus hydrophobicity column 0 of
        # the physicochemical block.
        drive = (
            1.5 * geometric[:, 0] - 1.2 * geometric[:, 3] + 0.8 * physicochemical[:, 0]
        )
        drive = drive + rng.normal(0, 1.0 - signal + 1e-9, length)
        n_positive = max(1, int(round(length * positive_rate)))
        labels = np.zeros(length, dtype=bool)
        labels[np.argsort(-drive)[:n_positive]] = True
        if signal <= 0.0:
            labels = rng.permutation(labels)

        # CA positions along a smooth curve so distances are physically sane.
        t = np.linspace(0, 4 * np.pi, length)
        coords = np.stack(
            [10 * np.cos(t), 10 * np.sin(t), np.linspace(0, 3.8 * length / 3, length)],
            axis=1,
        )
        deltas = coords[:, None, :] - coords[None, :, :]

        name = f"SYN{index:03d}"
        records[name] = ResidueFeatures(
            identifier=name, sequence=sequence,
            sequence_features=sequence_features,
            physicochemical=physicochemical, geometric=geometric,
            labels=labels,
            residue_keys=[("A", i + 1, "") for i in range(length)],
            distance_matrix=np.sqrt((deltas ** 2).sum(axis=-1)),
        )
    return records


def synthetic_family_features(
    n_families: int = 12,
    per_family: int = 5,
    length: int = 120,
    member_noise: float = 0.35,
    shared_signal: float = 0.45,
    positive_rate: float = 0.08,
    seed: int = 0,
) -> tuple[dict[str, "ResidueFeatures"], dict[str, str]]:
    """Homologous families that share sequence, structure AND binding site.

    This fixture makes homology leakage measurable, and getting it right takes
    some care. Two designs that do **not** work:

    * *Independent random proteins.* Every protein becomes its own cluster, so
      a homology-separated split and a random protein split are the same
      split. Nothing can leak.
    * *Shared sequence but independent features.* Family members cluster
      together, but if their per-residue features are drawn independently then
      nothing in the model's input identifies the family. A model cannot apply
      a family-specific rule it has no way to detect, so again nothing leaks.

    What actually leaks in the PDB is **near-duplication**: homologous entries
    have similar sequences, similar structures, and their ligands sit in the
    same pocket. So here each family has a template of per-residue features
    *and* a template binding site, and members are that template plus noise.

    A random protein split then puts near-duplicates of a training protein
    into the test set, and the model scores well by recognising them. A
    cluster-level split holds out whole families, so it must generalise from
    the weaker ``shared_signal`` rule that applies across families. The gap
    between the two is homology leakage.

    ``member_noise`` controls how close family members are — 0 makes them
    identical, large values dissolve the family. ``shared_signal`` controls
    how much genuinely generalisable signal exists.

    Returns (feature records, sequences).
    """
    rng = np.random.default_rng(seed)
    alphabet = list(AMINO_ACIDS)
    n_features = N_PHYSICOCHEMICAL_FEATURES + N_GEOMETRIC_FEATURES

    # One rule shared across all families: the generalisable part of the task.
    shared_weights = rng.normal(0, 1, n_features)

    records: dict[str, ResidueFeatures] = {}
    sequences: dict[str, str] = {}

    for family in range(n_families):
        template_sequence = list(rng.choice(alphabet, size=length))
        # The family's structural template: per-residue features every member
        # inherits, which is what makes members recognisable as relatives.
        template_features = rng.normal(0, 1, (length, n_features))
        # The family's pocket: which positions are binding, shared by members.
        template_site = rng.permutation(length)[
            : max(1, int(round(length * positive_rate)))
        ]

        for member in range(per_family):
            variant = list(template_sequence)
            for position in rng.choice(
                length, size=max(1, length // 12), replace=False
            ):
                variant[position] = rng.choice(alphabet)
            sequence = "".join(variant)

            block = template_features + rng.normal(
                0, member_noise, (length, n_features)
            )
            physicochemical = block[:, :N_PHYSICOCHEMICAL_FEATURES].astype(np.float32)
            geometric = block[:, N_PHYSICOCHEMICAL_FEATURES:].astype(np.float32)

            sequence_features = np.zeros(
                (length, N_SEQUENCE_FEATURES), dtype=np.float32
            )
            for position, residue in enumerate(sequence):
                sequence_features[position, AMINO_ACIDS.index(residue)] = 1.0

            # The label is the family's own pocket, plus a generalisable rule
            # that holds across families.
            drive = shared_signal * (block @ shared_weights)
            drive[template_site] += 3.0
            n_positive = max(1, int(round(length * positive_rate)))
            labels = np.zeros(length, dtype=bool)
            labels[np.argsort(-drive)[:n_positive]] = True

            index = np.arange(length)
            angle = np.deg2rad(100.0 * index)
            coords = np.stack(
                [2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * index], axis=1
            )
            deltas = coords[:, None, :] - coords[None, :, :]

            name = f"F{family:02d}M{member}"
            sequences[name] = sequence
            records[name] = ResidueFeatures(
                identifier=name, sequence=sequence,
                sequence_features=sequence_features,
                physicochemical=physicochemical, geometric=geometric,
                labels=labels,
                residue_keys=[("A", i + 1, "") for i in range(length)],
                distance_matrix=np.sqrt((deltas ** 2).sum(axis=-1)),
            )
    return records, sequences


def synthetic_families(
    n_families: int = 8, per_family: int = 5, length: int = 120, seed: int = 0
) -> dict[str, str]:
    """Sequences in known families, for testing clustering and splits.

    Each family has a template; members are the template with a small number
    of substitutions, so within-family identity is high and between-family
    identity is at background level. A clustering at 30% identity should
    recover the families exactly, which is what makes this a test of the
    clustering rather than a demonstration.
    """
    rng = np.random.default_rng(seed)
    alphabet = list(AMINO_ACIDS)
    sequences: dict[str, str] = {}

    for family in range(n_families):
        template = list(rng.choice(alphabet, size=length))
        for member in range(per_family):
            variant = list(template)
            # ~8% of positions substituted: members stay well above 30%
            # identity to each other and far below it to other families.
            for position in rng.choice(
                length, size=max(1, length // 12), replace=False
            ):
                variant[position] = rng.choice(alphabet)
            sequences[f"F{family}M{member}"] = "".join(variant)
    return sequences


def synthetic_pdb(
    n_residues: int = 40,
    ligand: str = "ATP",
    ligand_atoms: int = 8,
    ligand_centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
    include_water: bool = True,
    include_sulfate: bool = True,
    include_modified_residue: bool = False,
) -> str:
    """A valid PDB file with a known geometry and known contacts.

    The protein is a helix; the ligand sits at ``ligand_centre``, so which
    residues are within the contact cutoff is determined by the geometry and
    can be computed independently of the parser.

    ``include_water``, ``include_sulfate`` and ``include_modified_residue``
    add het groups that **must not** be labelled as ligands, so a test can
    confirm the curation drops them rather than assuming it does.
    """
    lines = ["HEADER    SYNTHETIC TEST STRUCTURE"]
    serial = 1
    # Ideal alpha helix: 1.5 A rise, 100 degrees per residue, 2.3 A radius.
    for index in range(n_residues):
        angle = np.deg2rad(100.0 * index)
        radius, rise = 2.3, 1.5
        ca = (radius * np.cos(angle), radius * np.sin(angle), rise * index)
        # N and C placed either side along the helix axis.
        n_atom = (ca[0], ca[1], ca[2] - 0.7)
        c_atom = (ca[0], ca[1], ca[2] + 0.7)
        # CB pointing outward from the helix axis.
        cb = (ca[0] * 1.6, ca[1] * 1.6, ca[2])
        for name, element, coord in (
            ("N", "N", n_atom), ("CA", "C", ca), ("C", "C", c_atom),
            ("CB", "C", cb),
        ):
            lines.append(
                f"ATOM  {serial:>5} {name:<4} ALA A{index + 1:>4}    "
                f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
                f"  1.00 20.00          {element:>2}"
            )
            serial += 1

    if include_modified_residue:
        # A selenomethionine: HETATM records that are polymer, not a ligand.
        for offset in range(4):
            lines.append(
                f"HETATM{serial:>5} CA   MSE A{n_residues + 1:>4}    "
                f"{50.0 + offset:>8.3f}{50.0:>8.3f}{50.0:>8.3f}"
                f"  1.00 20.00           C"
            )
            serial += 1

    centre = np.array(ligand_centre, dtype=float)
    for offset in range(ligand_atoms):
        coord = centre + np.array([offset * 0.6, 0.0, 0.0])
        lines.append(
            f"HETATM{serial:>5} C{offset:<3} {ligand:<3} A{900:>4}    "
            f"{coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
            f"  1.00 20.00           C"
        )
        serial += 1

    if include_water:
        for offset in range(5):
            lines.append(
                f"HETATM{serial:>5} O    HOH A{800 + offset:>4}    "
                f"{1.0:>8.3f}{1.0:>8.3f}{float(offset):>8.3f}"
                f"  1.00 20.00           O"
            )
            serial += 1

    if include_sulfate:
        # A sulfate close to the protein: must be excluded as an additive.
        for offset, (name, element) in enumerate(
            (("S", "S"), ("O1", "O"), ("O2", "O"), ("O3", "O"), ("O4", "O"))
        ):
            lines.append(
                f"HETATM{serial:>5} {name:<4} SO4 A{700:>4}    "
                f"{2.5:>8.3f}{2.5:>8.3f}{float(offset):>8.3f}"
                f"  1.00 20.00          {element:>2}"
            )
            serial += 1

    lines.append("TER   ")
    lines.append("END   ")
    return "\n".join(lines) + "\n"
