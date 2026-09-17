"""Splits over proteins, grouped by homology cluster.

The unit of splitting is the **cluster**, never the protein and never the
residue. Three levels of granularity are possible and only one is valid:

``residue``
    Splitting residues would put residues of the *same protein* in train and
    test. Neighbouring residues share features almost entirely, so this is
    the most severe form of leakage available in this task. Implemented as
    :func:`residue_split` solely so the control experiment can quantify it.

``protein``
    Splitting proteins keeps each protein whole but allows a protein and its
    close homolog to land on opposite sides. Because the PDB is highly
    redundant — many entries are the same protein with a different ligand —
    this is the leak that published binding-site numbers most often contain.
    :func:`random_split`.

``cluster``
    Splitting homology clusters is the only level at which a test protein is
    genuinely from an unseen family. :func:`homology_split`.

Every split carries its strategy, and :func:`assert_no_homology_leakage`
checks the invariant directly against the sequences rather than trusting the
clustering it was built from.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..exceptions import DataError, LeakageError
from ..homology import Clustering, pairwise_identity

# A homology-separated split needs at least this many clusters: with fewer,
# there is no way to form disjoint train, validation and test groups.
MIN_CLUSTERS = 3


@dataclass
class Split:
    """Protein names per fold, plus the clusters behind them."""

    name: str
    train: list[str]
    val: list[str]
    test: list[str]
    strategy: str
    train_clusters: list[int] = field(default_factory=list)
    val_clusters: list[int] = field(default_factory=list)
    test_clusters: list[int] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "name": self.name, "strategy": self.strategy,
            "n_train": len(self.train), "n_val": len(self.val),
            "n_test": len(self.test),
            "n_train_clusters": len(self.train_clusters),
            "n_val_clusters": len(self.val_clusters),
            "n_test_clusters": len(self.test_clusters),
            **self.extra,
        }


def assert_no_homology_leakage(
    split: Split,
    clusters: Clustering | None = None,
    sequences: dict[str, str] | None = None,
    max_identity: float | None = None,
    matrix=None,
) -> dict:
    """Assert that no cluster, and no near-identical sequence, spans folds.

    Two checks, because they can fail independently:

    1. **Cluster disjointness.** No cluster id appears in more than one fold.
       Cheap, and catches a bug in how the split was assembled.

    2. **Direct identity.** The highest pairwise identity between train and
       test sequences is below ``max_identity``. This is the check that
       matters, because it does not trust the clustering: an approximate
       clustering backend can miss a homologous pair, and then check 1 passes
       while the split still leaks.

    Check 2 costs O(n_train x n_test) alignments when computed from
    ``sequences`` — several minutes on a few hundred proteins, which is enough
    that it would end up switched off. Pass a precomputed
    :class:`~bindsite.homology.IdentityMatrix` as ``matrix`` instead and it
    becomes a table lookup, so it can run on every split.
    """
    report: dict = {"strategy": split.strategy, "checks": []}

    if clusters is not None:
        folds = {
            "train": {clusters.labels[p] for p in split.train if p in clusters.labels},
            "val": {clusters.labels[p] for p in split.val if p in clusters.labels},
            "test": {clusters.labels[p] for p in split.test if p in clusters.labels},
        }
        for left, right in (("train", "test"), ("train", "val"), ("val", "test")):
            shared = folds[left] & folds[right]
            if shared:
                members = clusters.members()
                offending = sorted(members[c] for c in sorted(shared))[:3]
                raise LeakageError(
                    f"homology cluster(s) {sorted(shared)} appear in both "
                    f"{left} and {right}. Example members: {offending}. "
                    "A test protein from a cluster the model trained on is not "
                    "an unseen family, and every metric computed on this split "
                    "would be optimistic."
                )
        report["checks"].append(
            {"check": "cluster disjointness", "passed": True,
             "n_clusters": clusters.n_clusters}
        )

    if sequences is not None or matrix is not None:
        threshold = (
            max_identity
            if max_identity is not None
            else (clusters.identity_threshold if clusters else 0.30)
        )
        worst, left_name, right_name = 0.0, "", ""
        for a in split.train:
            for b in split.test:
                if matrix is not None:
                    if not matrix.covers([a, b]):
                        continue
                    score, _ = matrix.get(a, b)
                elif a in sequences and b in sequences:
                    score, _ = pairwise_identity(sequences[a], sequences[b])
                else:
                    continue
                if score > worst:
                    worst, left_name, right_name = score, a, b
        # Strictly greater: a pair exactly at the clustering threshold is the
        # boundary case the clustering itself allows.
        if worst > threshold + 1e-9:
            raise LeakageError(
                f"train sequence {left_name!r} and test sequence {right_name!r} "
                f"share {worst:.1%} identity, above the {threshold:.0%} threshold "
                "this split claims to enforce. The clustering backend missed "
                "this pair."
            )
        report["checks"].append({
            "check": "max train-test identity", "passed": True,
            "max_identity": worst, "threshold": threshold,
            "worst_pair": [left_name, right_name],
            "source": "precomputed matrix" if matrix is not None else "alignments",
            "n_pairs_checked": len(split.train) * len(split.test),
        })

    report["passed"] = True
    return report


def _allocate(
    items: Sequence, test_fraction: float, val_fraction: float, seed: int
) -> tuple[list, list, list]:
    """Shuffle and allocate into train/val/test by fraction."""
    rng = np.random.default_rng(seed)
    order = list(items)
    rng.shuffle(order)
    total = len(order)
    n_test = max(1, int(round(total * test_fraction))) if test_fraction > 0 else 0
    n_val = max(1, int(round(total * val_fraction))) if val_fraction > 0 else 0
    if n_test + n_val >= total:
        raise DataError(
            f"test ({n_test}) and validation ({n_val}) take all {total} items, "
            "leaving nothing to train on; lower the fractions or add data"
        )
    return order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]


def homology_split(
    clusters: Clustering,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
    name: str = "homology",
) -> Split:
    """Split by homology cluster: the only level that measures generalisation.

    Clusters are allocated, then every member follows its cluster. Cluster
    sizes differ, so the resulting protein-level fractions will not match the
    requested fractions exactly; the realised fractions are recorded in
    ``extra`` rather than being silently different from what was asked for.
    """
    if clusters.n_sequences == 0:
        raise DataError("clustering contains no sequences")
    if clusters.n_clusters < MIN_CLUSTERS:
        sizes = clusters.cluster_sizes()
        raise DataError(
            f"a homology-separated split needs at least {MIN_CLUSTERS} clusters, "
            f"found {clusters.n_clusters} (sizes {sizes}). Every protein here is "
            "homologous to the others, so no split of this dataset can present "
            "the model with an unseen family. Add more diverse proteins, or "
            "report the result as within-family."
        )

    members = clusters.members()
    train_c, val_c, test_c = _allocate(
        sorted(members), test_fraction, val_fraction, seed
    )
    expand = lambda group: sorted(p for c in group for p in members[c])  # noqa: E731

    train, val, test = expand(train_c), expand(val_c), expand(test_c)
    total = len(train) + len(val) + len(test)
    return Split(
        name=name, train=train, val=val, test=test, strategy="homology_cluster",
        train_clusters=sorted(train_c), val_clusters=sorted(val_c),
        test_clusters=sorted(test_c),
        extra={
            "identity_threshold": clusters.identity_threshold,
            "clustering_backend": clusters.backend,
            "requested_test_fraction": test_fraction,
            "realised_test_fraction": len(test) / total if total else 0.0,
            "realised_val_fraction": len(val) / total if total else 0.0,
            "shared_clusters_train_test": 0,
        },
    )


def random_split(
    proteins: Sequence[str],
    clusters: Clustering | None = None,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
    name: str = "random_protein",
) -> Split:
    """Split proteins at random, ignoring homology.

    Provided **only** as the comparison arm for the control experiment. It
    records how many homology clusters it split across folds, so its own
    output states the problem with it.
    """
    train, val, test = _allocate(
        sorted(proteins), test_fraction, val_fraction, seed
    )
    extra: dict = {"note": "homology is ignored; for comparison only"}
    if clusters is not None:
        train_c = {clusters.labels[p] for p in train if p in clusters.labels}
        test_c = {clusters.labels[p] for p in test if p in clusters.labels}
        extra["shared_clusters_train_test"] = len(train_c & test_c)
    return Split(
        name=name, train=train, val=val, test=test, strategy="random_protein",
        extra=extra,
    )


def residue_split(
    proteins: Sequence[str],
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
    name: str = "random_residue",
) -> Split:
    """Every protein in every fold: residues are split, not proteins.

    The most severe leak available here, and included only so the control
    experiment can measure it. Adjacent residues share nearly all their
    features, so a model can interpolate a held-out residue from its own
    protein's neighbours without learning anything generalisable.

    The fold lists are identical by construction; the residue-level allocation
    is carried in ``extra`` for the loader to apply.
    """
    everything = sorted(proteins)
    return Split(
        name=name, train=everything, val=everything, test=everything,
        strategy="random_residue",
        extra={
            "residue_level": True,
            "test_fraction": test_fraction,
            "val_fraction": val_fraction,
            "seed": seed,
            "note": (
                "residues of the same protein appear in train and test; the "
                "most severe leak in this task, included for comparison only"
            ),
        },
    )


def cluster_kfold(
    clusters: Clustering, n_folds: int = 5, seed: int = 0
) -> list[Split]:
    """K-fold cross-validation with whole clusters held out per fold."""
    if n_folds < 2:
        raise DataError(f"n_folds must be at least 2, got {n_folds}")
    if clusters.n_clusters < n_folds:
        raise DataError(
            f"{n_folds} folds requested but only {clusters.n_clusters} clusters "
            "exist; a fold would have no cluster to hold out"
        )

    members = clusters.members()
    rng = np.random.default_rng(seed)
    order = sorted(members)
    rng.shuffle(order)
    chunks = [list(c) for c in np.array_split(np.array(order), n_folds)]

    splits: list[Split] = []
    for index, held_out in enumerate(chunks):
        train_c = [c for c in order if c not in set(held_out)]
        expand = lambda group: sorted(p for c in group for p in members[c])  # noqa: E731
        splits.append(
            Split(
                name=f"fold{index}", train=expand(train_c), val=[],
                test=expand(held_out), strategy="homology_cluster",
                train_clusters=sorted(train_c), test_clusters=sorted(held_out),
                extra={
                    "fold": index, "n_folds": n_folds,
                    "identity_threshold": clusters.identity_threshold,
                },
            )
        )
    return splits


def label_balance(
    labels_per_protein: dict[str, np.ndarray], split: Split
) -> dict:
    """Residue counts and positive rate per fold.

    The positive rate is the number that determines whether AUROC is a
    reasonable headline metric. At a few percent positives, AUROC stays high
    for a model with poor precision, so it is reported per fold here and
    AUPRC is reported alongside it in :mod:`bindsite.evaluate`.
    """
    out: dict = {}
    for fold, names in (("train", split.train), ("val", split.val), ("test", split.test)):
        arrays = [labels_per_protein[n] for n in names if n in labels_per_protein]
        if not arrays:
            out[f"{fold}_n_residues"] = 0
            out[f"{fold}_positive_rate"] = None
            continue
        stacked = np.concatenate(arrays)
        out[f"{fold}_n_proteins"] = len(arrays)
        out[f"{fold}_n_residues"] = int(stacked.size)
        out[f"{fold}_n_binding"] = int(stacked.sum())
        out[f"{fold}_positive_rate"] = float(stacked.mean())
    return out
