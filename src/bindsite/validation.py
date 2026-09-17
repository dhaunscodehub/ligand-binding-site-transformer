"""Control experiments that establish what the metrics mean.

An AUROC on its own does not say whether a binding-site model learned pocket
geometry, memorised protein families, or was handed a task it could not fail.
These controls pin the scale and quantify the specific failure modes.

The controls:

1. **Label shuffle (negative control).** Binding labels are permuted within
   each protein, destroying any relationship to structure while preserving the
   positive rate exactly. A correct pipeline scores at chance: AUROC 0.5 and
   AUPRC equal to the positive rate. Anything above that means information is
   leaking from labels into features.

2. **Positive control.** A synthetic task where the label is a known function
   of the features. A correct pipeline recovers it. Failure here means a poor
   result elsewhere is uninformative.

3. **Homology leakage, three splits.** The same data split at residue,
   protein and cluster level. Residue-level and protein-level splits should
   score higher than the cluster-level split, and the differences are the
   quantitative case for homology separation.

4. **Direct identity verification.** The homology split's train-test maximum
   pairwise identity is measured against the sequences themselves, not
   inferred from the clustering.

5. **AUROC versus AUPRC under imbalance.** A deliberately mediocre predictor
   is scored both ways to show the gap. This is not a bug demonstration — it
   is why this repository does not report AUROC alone.

6. **Leakage guard fires.** The homology assertion rejects a deliberately
   corrupted split.

7. **Ligand-blind features.** Features computed with and without the ligand
   present are identical, proving no ligand coordinate reaches the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.splits import (
    Split, assert_no_homology_leakage, homology_split, random_split,
)
from .exceptions import LeakageError
from .homology import Clustering, cluster_greedy
from .models.baselines import fit_baseline

# A control that should sit at chance is allowed this much away from it,
# applied to the MEAN over replicate seeds.
CHANCE_TOLERANCE_AUROC = 0.03
# Seeds the shuffle control is averaged over. A single shuffle of ~1800
# test residues with ~150 positives has an AUROC standard error near
# 0.024, and individual seeds measured from -0.063 to +0.038; asserting
# per-seed would test the draw rather than the claim.
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)
# The positive control must reach at least this AUPRC lift over chance.
POSITIVE_CONTROL_MIN_LIFT = 2.0


def _stack(features_by_name, names, geometric_only: bool = False):
    """Stack per-residue features and labels for the given proteins."""
    blocks, labels = [], []
    for name in names:
        record = features_by_name[name]
        block = (
            record.structure_block if geometric_only
            else np.hstack([record.sequence_block, record.structure_block])
        )
        blocks.append(block)
        labels.append(record.labels.astype(int))
    return np.vstack(blocks), np.concatenate(labels)


def _feature_names(record, geometric_only: bool = False) -> list[str]:
    from .features import feature_names

    names = feature_names()
    if geometric_only:
        return list(names["geometric"])
    return list(names["sequence"]) + list(names["physicochemical"]) + list(
        names["geometric"]
    )


def _fit_score(
    features_by_name, split: Split, model: str = "logistic", seed: int = 0
) -> dict:
    """Fit on train, score on test, return residue-level metrics."""
    from .evaluate import residue_metrics

    names = _feature_names(features_by_name[split.train[0]])
    train_x, train_y = _stack(features_by_name, split.train)
    test_x, test_y = _stack(features_by_name, split.test)
    fitted = fit_baseline(model, train_x, train_y, names, seed=seed)
    metrics = residue_metrics(test_y, fitted.predict_proba(test_x))
    return {
        "auroc": metrics.auroc, "auprc": metrics.auprc,
        "auprc_lift": metrics.auprc_lift,
        "positive_rate": metrics.positive_rate,
        "best_f1": metrics.best_f1,
        "n_test_residues": metrics.n_residues,
        "n_train_residues": int(train_y.size),
    }


def _shuffle_replicate(features_by_name, clusters: Clustering, seed: int) -> dict:
    """One replicate: permute labels within each protein, then fit and score."""
    import copy

    rng = np.random.default_rng(seed)
    shuffled = {}
    for name, record in features_by_name.items():
        clone = copy.copy(record)
        clone.labels = rng.permutation(record.labels)
        shuffled[name] = clone

    split = homology_split(clusters, test_fraction=0.25, val_fraction=0.0, seed=seed)
    return {"seed": seed, **_fit_score(shuffled, split, seed=seed)}


def label_shuffle_control(
    features_by_name,
    clusters: Clustering,
    seed: int = 0,
    seeds: tuple[int, ...] = SHUFFLE_SEEDS,
) -> dict:
    """Permuting labels within each protein must reduce the model to chance.

    Shuffling **within** each protein rather than globally is deliberate: it
    preserves every protein's positive rate exactly, so the only thing
    destroyed is the association between a residue's features and its label.
    A global shuffle would also change per-protein rates, confounding the two.

    Averaged over replicate seeds, because a single shuffle is a noisy
    estimate of chance. The per-seed spread is reported alongside the mean.
    """
    replicates = [_shuffle_replicate(features_by_name, clusters, s) for s in seeds]
    aurocs = np.array(
        [r["auroc"] if r["auroc"] is not None else 0.5 for r in replicates]
    )
    auprcs = np.array([r["auprc"] or 0.0 for r in replicates])
    lifts = np.array([r["auprc_lift"] or 0.0 for r in replicates])
    excess = float(aurocs.mean() - 0.5)

    return {
        "name": "label shuffle (negative control): features carry no signal",
        "expectation": (
            f"mean AUROC within {CHANCE_TOLERANCE_AUROC} of 0.5 and mean AUPRC "
            "near the positive rate"
        ),
        "passed": bool(abs(excess) <= CHANCE_TOLERANCE_AUROC),
        "auroc": float(aurocs.mean()),
        "auroc_sd": float(aurocs.std()),
        "auroc_excess_over_chance": excess,
        "auprc": float(auprcs.mean()),
        "auprc_lift": float(lifts.mean()),
        "positive_rate": replicates[0]["positive_rate"],
        "n_test_residues": replicates[0]["n_test_residues"],
        "n_train_residues": replicates[0]["n_train_residues"],
        "n_replicates": len(replicates),
        "seeds": list(seeds),
        "per_seed_auroc": [round(float(a), 4) for a in aurocs],
        "split_strategy": "homology_cluster",
    }


def positive_control(n_proteins: int = 40, seed: int = 0) -> dict:
    """A label that is a known function of the features must be recoverable."""
    from .testing import synthetic_features

    records = synthetic_features(
        n_proteins=n_proteins, signal=1.0, seed=seed
    )
    names = sorted(records)
    cut = int(len(names) * 0.75)
    split = Split(
        name="synthetic", train=names[:cut], val=[], test=names[cut:],
        strategy="synthetic",
    )
    scores = _fit_score(records, split, seed=seed)
    return {
        "name": "positive control: label is a known function of the features",
        "expectation": f"AUPRC lift over chance >= {POSITIVE_CONTROL_MIN_LIFT}",
        "passed": bool((scores["auprc_lift"] or 0.0) >= POSITIVE_CONTROL_MIN_LIFT),
        "split_strategy": "synthetic",
        **scores,
    }


def leakage_by_split_level(
    features_by_name, clusters: Clustering, seed: int = 0,
    model: str = "random_forest",
) -> dict:
    """Score the same data under residue-, protein- and cluster-level splits.

    The residue-level arm splits residues within each protein, so a protein
    contributes to both train and test. The protein-level arm keeps proteins
    whole but ignores homology. The cluster-level arm is the only one where a
    test protein is from an unseen family.

    A **high-capacity** model is used (random forest by default), and that is
    the point rather than an incidental choice. Leakage is exploited by
    capacity: a single linear model cannot condition its rule on which family
    a residue belongs to, so it shows only a small gap between the splits. A
    model that can memorise family-specific structure — a forest, or a
    transformer — shows a large one. Measuring the leak with a low-capacity
    model would understate the risk for the models people actually deploy.
    """
    from .evaluate import residue_metrics

    names = sorted(features_by_name)
    all_names = _feature_names(features_by_name[names[0]])
    rng = np.random.default_rng(seed)

    # --- residue level: split rows, not proteins ---
    pooled_x, pooled_y = _stack(features_by_name, names)
    order = rng.permutation(pooled_y.size)
    cut = int(pooled_y.size * 0.75)
    train_index, test_index = order[:cut], order[cut:]
    fitted = fit_baseline(
        model, pooled_x[train_index], pooled_y[train_index], all_names, seed=seed
    )
    residue_level = residue_metrics(
        pooled_y[test_index], fitted.predict_proba(pooled_x[test_index])
    )

    # --- protein level: whole proteins, homology ignored ---
    protein_split = random_split(
        names, clusters, test_fraction=0.25, val_fraction=0.0, seed=seed
    )
    protein_level = _fit_score(
        features_by_name, protein_split, model=model, seed=seed
    )

    # --- cluster level: unseen families ---
    cluster_split = homology_split(
        clusters, test_fraction=0.25, val_fraction=0.0, seed=seed
    )
    assert_no_homology_leakage(cluster_split, clusters)
    cluster_level = _fit_score(
        features_by_name, cluster_split, model=model, seed=seed
    )

    arms = {
        "random_residue": {
            "auroc": residue_level.auroc, "auprc": residue_level.auprc,
            "auprc_lift": residue_level.auprc_lift,
            "positive_rate": residue_level.positive_rate,
        },
        "random_protein": {
            "auroc": protein_level["auroc"], "auprc": protein_level["auprc"],
            "auprc_lift": protein_level["auprc_lift"],
            "positive_rate": protein_level["positive_rate"],
            "shared_clusters_train_test": protein_split.extra.get(
                "shared_clusters_train_test"
            ),
        },
        "homology_cluster": {
            "auroc": cluster_level["auroc"], "auprc": cluster_level["auprc"],
            "auprc_lift": cluster_level["auprc_lift"],
            "positive_rate": cluster_level["positive_rate"],
            "shared_clusters_train_test": 0,
        },
    }

    honest = arms["homology_cluster"]["auprc"] or 0.0
    return {
        "name": "leakage by split level: residue vs protein vs homology cluster",
        "model": model,
        "expectation": (
            "a residue-level split scores highest, a homology-separated split "
            "lowest; the differences quantify the leak"
        ),
        # The ordering is the claim. Both leaking arms must exceed the honest one.
        "passed": bool(
            (arms["random_residue"]["auprc"] or 0.0) > honest
            and (arms["random_protein"]["auprc"] or 0.0) >= honest
        ),
        "arms": arms,
        "residue_split_auprc_inflation": float(
            (arms["random_residue"]["auprc"] or 0.0) - honest
        ),
        "protein_split_auprc_inflation": float(
            (arms["random_protein"]["auprc"] or 0.0) - honest
        ),
    }


def identity_verification(
    features_by_name, clusters: Clustering, sequences: dict[str, str], seed: int = 0
) -> dict:
    """Verify the homology split against the sequences, not the clustering."""
    split = homology_split(clusters, test_fraction=0.25, val_fraction=0.0, seed=seed)
    report = assert_no_homology_leakage(
        split, clusters, sequences=sequences,
    )
    identity_check = next(
        (c for c in report["checks"] if c["check"] == "max train-test identity"),
        None,
    )
    return {
        "name": "homology split verified directly against the sequences",
        "expectation": (
            f"maximum train-test identity at or below the "
            f"{clusters.identity_threshold:.0%} threshold"
        ),
        "passed": bool(identity_check is not None and identity_check["passed"]),
        "max_train_test_identity": (
            identity_check["max_identity"] if identity_check else None
        ),
        "threshold": clusters.identity_threshold,
        "worst_pair": identity_check["worst_pair"] if identity_check else None,
        "n_train": len(split.train), "n_test": len(split.test),
    }


def auroc_versus_auprc(positive_rate: float = 0.10, n: int = 20000, seed: int = 0) -> dict:
    """Show the AUROC/AUPRC gap on a mediocre predictor under imbalance.

    Constructs scores with a modest real signal, then reports both metrics.
    The point is not that either is wrong but that AUROC alone is not
    informative at a 10% positive rate: it stays high while precision is poor.
    """
    from .evaluate import residue_metrics

    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < positive_rate).astype(int)
    # Positives drawn from a slightly higher-mean distribution: a real but
    # unimpressive separation, which is what a working model looks like here.
    scores = rng.normal(0.0, 1.0, n) + labels * 1.2
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    metrics = residue_metrics(labels, probabilities)

    return {
        "name": "AUROC overstates a mediocre predictor under class imbalance",
        "expectation": (
            "AUROC well above 0.5 while AUPRC and precision stay modest, "
            "demonstrating why AUROC is not reported alone"
        ),
        "passed": bool(
            metrics.auroc is not None and metrics.auprc is not None
            and metrics.auroc > 0.75 and metrics.auprc < 0.55
        ),
        "auroc": metrics.auroc, "auprc": metrics.auprc,
        "auprc_chance": metrics.positive_rate,
        "auprc_lift": metrics.auprc_lift,
        "best_f1": metrics.best_f1,
        "precision_at_best_f1": metrics.precision_at_best_f1,
        "interpretation": (
            f"AUROC {metrics.auroc:.3f} looks strong, but the best achievable F1 "
            f"is {metrics.best_f1:.3f} with precision "
            f"{metrics.precision_at_best_f1:.3f}. The ROC false-positive-rate "
            "denominator is the large negative class, so many false positives "
            "barely move it."
        ),
    }


def leakage_guard_fires(clusters: Clustering, seed: int = 0) -> dict:
    """The homology assertion must reject a corrupted split."""
    split = homology_split(clusters, test_fraction=0.3, val_fraction=0.0, seed=seed)
    if not split.train or not split.test:
        return {
            "name": "leakage guard rejects a corrupted homology split",
            "passed": False,
            "message": "could not build a split to corrupt",
        }

    # Move one training protein into the test fold. Its cluster is now on both
    # sides, which is exactly the corruption the guard exists to catch.
    corrupted = Split(
        name=split.name, train=list(split.train[1:]), val=list(split.val),
        test=list(split.test) + [split.train[0]], strategy=split.strategy,
        train_clusters=list(split.train_clusters),
        val_clusters=list(split.val_clusters),
        test_clusters=list(split.test_clusters),
    )
    try:
        assert_no_homology_leakage(corrupted, clusters)
    except LeakageError as error:
        return {
            "name": "leakage guard rejects a corrupted homology split",
            "expectation": "raises LeakageError when a cluster spans train and test",
            "passed": True,
            "message": str(error)[:300],
        }
    return {
        "name": "leakage guard rejects a corrupted homology split",
        "expectation": "raises LeakageError when a cluster spans train and test",
        "passed": False,
        "message": (
            "the guard accepted a split in which a homology cluster appears in "
            "both train and test; no leakage check in this package can be relied on"
        ),
    }


def features_are_ligand_blind(pdb_path: str) -> dict:
    """Features must be identical whether or not the ligand is in the file.

    A pocket is a concavity, so any geometric feature computed with the ligand
    still present would partly measure where the ligand is — the answer. This
    strips every HETATM record and checks the features are bit-identical.
    """
    from pathlib import Path

    from .features import featurise
    from .structure import structure_from_pdb_text

    text = Path(pdb_path).read_text()
    with_ligand = structure_from_pdb_text(text, "with_ligand", require_ligand=True)

    stripped = "\n".join(
        line for line in text.splitlines() if not line.startswith("HETATM")
    )
    without = structure_from_pdb_text(
        stripped, "without_ligand", require_ligand=False
    )

    left, right = featurise(with_ligand), featurise(without)
    identical = (
        left.n_residues == right.n_residues
        and np.array_equal(left.sequence_features, right.sequence_features)
        and np.array_equal(left.physicochemical, right.physicochemical)
        and np.allclose(left.geometric, right.geometric, atol=0, rtol=0)
        and np.allclose(left.distance_matrix, right.distance_matrix, atol=0, rtol=0)
    )
    return {
        "name": "features are identical with and without the ligand present",
        "expectation": "no ligand coordinate influences any feature",
        "passed": bool(identical),
        "n_residues": left.n_residues,
        "n_binding_with_ligand": int(left.labels.sum()),
        "n_binding_without_ligand": int(right.labels.sum()),
        "max_geometric_difference": float(
            np.abs(left.geometric - right.geometric).max()
        ) if left.n_residues == right.n_residues else None,
    }


def run_controls(
    features_by_name=None,
    clusters: Clustering | None = None,
    sequences: dict[str, str] | None = None,
    reference_pdb: str | None = None,
    seed: int = 0,
) -> dict:
    """Run every control that the supplied inputs permit.

    The synthetic controls always run. The dataset-dependent controls need a
    real featurised dataset and its clustering; when absent they are recorded
    as skipped rather than silently omitted.
    """
    checks: list[dict] = [
        positive_control(seed=seed),
        auroc_versus_auprc(seed=seed),
    ]
    skipped: dict[str, str] = {}

    if features_by_name and clusters is not None:
        checks.append(label_shuffle_control(features_by_name, clusters, seed))
        checks.append(leakage_by_split_level(features_by_name, clusters, seed))
        checks.append(leakage_guard_fires(clusters, seed))
        if sequences:
            checks.append(
                identity_verification(features_by_name, clusters, sequences, seed)
            )
        else:
            skipped["identity_verification"] = "no sequences supplied"
    else:
        for name in (
            "label_shuffle_control", "leakage_by_split_level",
            "leakage_guard_fires", "identity_verification",
        ):
            skipped[name] = "no featurised dataset and clustering supplied"

    if reference_pdb:
        checks.append(features_are_ligand_blind(reference_pdb))
    else:
        skipped["features_are_ligand_blind"] = "no reference structure supplied"

    levels = next(
        (c for c in checks if c["name"].startswith("leakage by split level")), None
    )
    return {
        "seed": seed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "skipped": skipped,
        "headline": (
            {
                "residue_split_auprc_inflation": levels["residue_split_auprc_inflation"],
                "protein_split_auprc_inflation": levels["protein_split_auprc_inflation"],
                "homology_separated_auprc": levels["arms"]["homology_cluster"]["auprc"],
            }
            if levels else None
        ),
    }
