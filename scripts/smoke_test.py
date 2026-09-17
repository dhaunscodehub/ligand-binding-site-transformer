#!/usr/bin/env python3
"""Fast end-to-end smoke test: every module exercised, no network needed.

Answers one question — can this installation produce a result at all — and is
the first thing to run after cloning.

    python scripts/smoke_test.py

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))


def main() -> int:  # noqa: PLR0915 - a flat list of checks is the point
    from bindsite.chemcomp import ComponentCache, classify
    from bindsite.config import config_from_dict
    from bindsite.data.splits import (
        assert_no_homology_leakage, homology_split, random_split,
    )
    from bindsite.evaluate import compare_models, evaluate_predictions, residue_metrics
    from bindsite.exceptions import (
        ConfigError, DataError, LeakageError, NoLigandError,
    )
    from bindsite.features import featurise, feature_names
    from bindsite.homology import (
        cluster_connected_components, cluster_from_matrix, identity_matrix,
        pairwise_identity,
    )
    from bindsite.io_utils import write_json
    from bindsite.models.baselines import BASELINE_MODELS, fit_baseline
    from bindsite.structure import structure_from_pdb_text
    from bindsite.testing import (
        synthetic_families, synthetic_family_features, synthetic_features,
        synthetic_pdb,
    )
    from bindsite.validation import (
        auroc_versus_auprc, features_are_ligand_blind, positive_control,
    )

    # ---- structure parsing and curation ----------------------------------
    pdb = synthetic_pdb(n_residues=40, ligand="ATP", ligand_centre=(2.3, 0.0, 20.0))
    structure = structure_from_pdb_text(pdb, "SMOKE")
    check(
        "a valid PDB file parses into residues",
        structure.n_residues == 40,
        f"{structure.n_residues} residues",
    )
    check(
        "the real ligand is kept",
        [l.residue_name for l in structure.ligands] == ["ATP"],
    )
    check(
        "water and sulfate are excluded as non-ligands",
        "HOH" in structure.excluded_hetero and "SO4" in structure.excluded_hetero,
        f"excluded {sorted(structure.excluded_hetero)}",
    )
    check(
        "binding residues are within the contact cutoff",
        all(
            r.min_ligand_distance <= structure.contact_cutoff
            for r in structure.residues if r.is_binding
        ),
        f"{structure.n_binding} binding ({structure.binding_fraction:.1%})",
    )
    check(
        "a larger cutoff labels more residues",
        structure_from_pdb_text(pdb, "S", cutoff=6.0).n_binding
        >= structure.n_binding,
    )
    try:
        structure_from_pdb_text(
            synthetic_pdb(n_residues=20, ligand_atoms=0), "S", require_ligand=True
        )
        refused = False
    except NoLigandError:
        refused = True
    check("a solvent-only structure is refused", refused)

    # ---- chemical component dictionary -----------------------------------
    cache = ComponentCache(ROOT / "data/chemcomp_cache.json", offline=True)
    if len(cache):
        check(
            "modified residues classify as polymer, not ligand",
            classify("MSE", cache) in ("polymer", "unknown"),
            f"{len(cache)} components cached",
        )
    else:
        check("component cache present (optional)", True, "not built; skipped")

    # ---- features ---------------------------------------------------------
    features = featurise(structure)
    names = feature_names()
    check(
        "every feature block has the declared width",
        features.sequence_features.shape[1] == len(names["sequence"])
        and features.physicochemical.shape[1] == len(names["physicochemical"])
        and features.geometric.shape[1] == len(names["geometric"]),
        f"{features.sequence_block.shape[1]} sequence + "
        f"{features.structure_block.shape[1]} structure",
    )
    check(
        "all features are finite",
        np.isfinite(features.sequence_block).all()
        and np.isfinite(features.structure_block).all(),
    )
    matrix = features.distance_matrix
    check(
        "the CA-CA distance matrix is symmetric with a zero diagonal",
        np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0.0),
    )
    blind = features_are_ligand_blind_inline(pdb)
    check(
        "features are identical with and without the ligand present",
        blind,
        "no ligand coordinate reaches the model",
    )

    # ---- homology ---------------------------------------------------------
    identical, _ = pairwise_identity("ACDEFGHIKL" * 5, "ACDEFGHIKL" * 5)
    unrelated, _ = pairwise_identity("ACDEFGHIKL" * 5, "WWWWWWWWWW" * 5)
    check(
        "identity is 1.0 for identical and low for unrelated sequences",
        identical == 1.0 and unrelated < 0.3,
        f"identical {identical:.2f}, unrelated {unrelated:.2f}",
    )
    sequences = synthetic_families(n_families=5, per_family=4, seed=0)
    clustering = cluster_connected_components(sequences, identity=0.30)
    check(
        "clustering recovers known families",
        clustering.n_clusters == 5,
        f"{clustering.n_clusters} clusters of sizes "
        f"{sorted(clustering.cluster_sizes().values())}",
    )
    violations = [
        (a, b)
        for a in sequences for b in sequences
        if a < b
        and pairwise_identity(sequences[a], sequences[b])[0] >= 0.30
        and pairwise_identity(sequences[a], sequences[b])[1] >= 0.50
        and clustering.labels[a] != clustering.labels[b]
    ]
    check(
        "no pair above the threshold spans two clusters",
        not violations,
        "the invariant a homology split depends on",
    )
    computed = identity_matrix(sequences, mode="local")
    check(
        "clustering from a cached matrix matches clustering from sequences",
        cluster_from_matrix(computed, identity=0.30).labels == clustering.labels,
    )

    # ---- splits -----------------------------------------------------------
    split = homology_split(clustering, test_fraction=0.4, val_fraction=0.0, seed=0)
    check(
        "a homology split shares no cluster between folds",
        not set(split.train_clusters) & set(split.test_clusters),
        f"{len(split.train)} train / {len(split.test)} test proteins",
    )
    report = assert_no_homology_leakage(
        split, clustering, matrix=computed, max_identity=0.30
    )
    worst = next(
        c for c in report["checks"] if c["check"] == "max train-test identity"
    )
    check(
        "the split is verified against the sequences, not the clustering",
        worst["passed"],
        f"max train-test identity {worst['max_identity']:.3f}",
    )
    leaky = random_split(sorted(sequences), clustering, test_fraction=0.4, seed=0)
    check(
        "a random protein split does share clusters (the failure mode)",
        leaky.extra["shared_clusters_train_test"] > 0,
        f"{leaky.extra['shared_clusters_train_test']} shared clusters",
    )
    from bindsite.data.splits import Split

    corrupted = Split(
        name="bad", train=list(split.train[1:]), val=[],
        test=list(split.test) + [split.train[0]], strategy=split.strategy,
    )
    try:
        assert_no_homology_leakage(corrupted, clustering)
        fired = False
    except LeakageError:
        fired = True
    check("the leakage guard rejects a corrupted split", fired)
    try:
        from bindsite.homology import Clustering

        homology_split(
            Clustering(labels={"a": 0, "b": 0}, identity_threshold=0.3,
                       coverage_threshold=0.5, backend="t"),
            test_fraction=0.25,
        )
        refused_few = False
    except DataError:
        refused_few = True
    check("a split with too few clusters is refused", refused_few)

    # ---- models and evaluation -------------------------------------------
    records, family_sequences = synthetic_family_features(
        n_families=8, per_family=4, length=90, seed=0
    )
    all_names = (
        list(names["sequence"]) + list(names["physicochemical"])
        + list(names["geometric"])
    )
    stacked = np.vstack([
        np.hstack([r.sequence_block, r.structure_block]) for r in records.values()
    ])
    labels = np.concatenate([r.labels.astype(int) for r in records.values()])
    fitted = {}
    for name in BASELINE_MODELS:
        try:
            fitted[name] = fit_baseline(name, stacked, labels, all_names, seed=0)
        except DataError as error:
            check(f"baseline {name} fits", False, str(error))
    check(
        "every baseline fits",
        len(fitted) == len(BASELINE_MODELS),
        f"{len(fitted)}/{len(BASELINE_MODELS)}",
    )
    prevalence = residue_metrics(labels, fitted["prevalence"].predict_proba(stacked))
    check(
        "the prevalence baseline has chance AUROC and AUPRC equal to the rate",
        abs(prevalence.auroc - 0.5) < 1e-9
        and abs(prevalence.auprc - prevalence.positive_rate) < 0.02,
        f"AUROC {prevalence.auroc:.3f}, AUPRC {prevalence.auprc:.3f}, "
        f"rate {prevalence.positive_rate:.3f}",
    )
    learned = residue_metrics(labels, fitted["logistic"].predict_proba(stacked))
    check(
        "a learned baseline beats the prevalence floor",
        learned.auprc > prevalence.auprc,
        f"AUPRC {learned.auprc:.3f} vs {prevalence.auprc:.3f}",
    )
    check(
        "AUPRC is reported with its chance level and lift",
        learned.to_dict()["auprc_chance"] is not None
        and learned.auprc_lift is not None,
        f"lift {learned.auprc_lift:.2f}x",
    )
    check(
        "precision in the top L/10 is reported",
        learned.precision_at_top_l10 is not None,
        f"P@L/10 {learned.precision_at_top_l10:.3f}",
    )
    per_protein_labels = {n: r.labels for n, r in list(records.items())[:8]}
    per_protein_scores = {
        n: fitted["logistic"].predict_proba(
            np.hstack([records[n].sequence_block, records[n].structure_block])
        )
        for n in per_protein_labels
    }
    result = evaluate_predictions(
        "logistic", per_protein_labels, per_protein_scores,
        split_strategy="homology_cluster",
    )
    check(
        "both residue-level and protein-level metrics are produced",
        result.protein is not None and result.residue is not None,
        f"{result.protein.n_proteins} proteins scored individually",
    )
    check(
        "the AUROC caveat is always attached",
        any("not a sufficient summary" in c for c in result.caveats),
    )
    leaky_result = evaluate_predictions(
        "logistic", per_protein_labels, per_protein_scores,
        split_strategy="random_protein",
    )
    check(
        "a leaking split's results carry their own warning",
        any("homologous proteins" in c for c in leaky_result.caveats),
    )
    comparison = compare_models([result])
    check(
        "model comparison ranks by AUPRC",
        comparison["ranked_by"] == "auprc",
    )
    undefined = residue_metrics(np.zeros(20, dtype=int), np.random.random(20))
    check(
        "an undefined AUROC is None rather than invented",
        undefined.auroc is None and undefined.auprc is None,
    )

    # ---- controls ---------------------------------------------------------
    control = positive_control(n_proteins=20, seed=0)
    check(
        "the positive control recovers a known function of the features",
        control["passed"],
        f"AUPRC lift {control['auprc_lift']:.2f}x",
    )
    gap = auroc_versus_auprc(positive_rate=0.08, seed=0)
    check(
        "AUROC is shown to overstate a mediocre predictor under imbalance",
        gap["passed"],
        f"AUROC {gap['auroc']:.3f} but AUPRC {gap['auprc']:.3f}, "
        f"F1 {gap['best_f1']:.3f}",
    )

    # ---- transformer ------------------------------------------------------
    try:
        import torch

        from bindsite.models.transformer import (
            TransformerConfig, build_model, select_device,
        )

        config = TransformerConfig(d_model=64, n_heads=4, n_layers=2, max_length=128)
        model = build_model(config)
        logits = model(
            torch.randn(2, 40, 30), torch.randn(2, 40, 8),
            torch.rand(2, 40, 40) * 20, torch.ones(2, 40, dtype=torch.bool),
        )
        check(
            "the transformer produces one logit per residue",
            tuple(logits.shape) == (2, 40) and bool(torch.isfinite(logits).all()),
            f"{model.n_parameters():,} parameters on {select_device()}",
        )
        logits.sum().backward()
        check(
            "every transformer gradient is finite",
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters() if p.requires_grad
            ),
        )
        for use_sequence, use_structure in ((True, False), (False, True)):
            ablated = build_model(
                TransformerConfig(
                    d_model=64, n_heads=4, n_layers=1, max_length=128,
                    use_sequence=use_sequence, use_structure=use_structure,
                )
            )
            out = ablated(
                torch.randn(1, 20, 30) if use_sequence else None,
                torch.randn(1, 20, 8) if use_structure else None,
                torch.rand(1, 20, 20) * 20, torch.ones(1, 20, dtype=torch.bool),
            )
            check(
                f"ablation runs with sequence={use_sequence} "
                f"structure={use_structure}",
                tuple(out.shape) == (1, 20),
            )
    except ImportError:
        check("transformer checks (PyTorch)", True, "torch not installed; skipped")

    # ---- config -----------------------------------------------------------
    check("a minimal config loads", config_from_dict({}).split.strategy == "homology")
    for payload, why in [
        ({"data": {"contact_cutof": 4}}, "misspelled key"),
        ({"nonsense": {}}, "unknown section"),
        ({"homology": {"identity_threshold": 30}}, "percentage not fraction"),
        ({"split": {"strategy": "random"}}, "invalid split strategy"),
        ({"data": {"contact_cutoff": 15}}, "implausible contact cutoff"),
    ]:
        try:
            config_from_dict(payload)
            check(f"config rejects {why}", False, "accepted it")
        except ConfigError:
            check(f"config rejects {why}", True)

    # ---- serialisation ----------------------------------------------------
    with tempfile.TemporaryDirectory() as directory:
        path = write_json(
            Path(directory) / "out.json",
            {"b": np.bool_(True), "i": np.int64(3), "nan": np.float64("nan"),
             "inf": float("inf"), "a": np.arange(3)},
        )
        import json

        text = path.read_text()
        loaded = json.loads(text)
    check(
        "numpy types, NaN and Infinity serialise to standard JSON",
        loaded["b"] is True and loaded["nan"] is None and loaded["inf"] is None
        and "NaN" not in text and "Infinity" not in text,
    )

    width = 66
    print()
    print("SMOKE TEST")
    print("=" * (width + 14))
    for number, (name, passed, detail) in enumerate(CHECKS, start=1):
        print(f"{number:>3}  {'PASS' if passed else 'FAIL'}  {name[:width]:<{width}}  {detail}")
    print("=" * (width + 14))
    n_passed = sum(1 for _, passed, _ in CHECKS if passed)
    print(f"{n_passed}/{len(CHECKS)} checks passed")
    print()
    return 0 if n_passed == len(CHECKS) else 1


def features_are_ligand_blind_inline(pdb_text: str) -> bool:
    """Compare features with and without HETATM records, in memory."""
    from bindsite.features import featurise
    from bindsite.structure import structure_from_pdb_text

    with_ligand = featurise(structure_from_pdb_text(pdb_text, "w"))
    stripped = "\n".join(
        line for line in pdb_text.splitlines() if not line.startswith("HETATM")
    )
    without = featurise(
        structure_from_pdb_text(stripped, "wo", require_ligand=False)
    )
    return bool(
        np.array_equal(with_ligand.geometric, without.geometric)
        and np.array_equal(with_ligand.physicochemical, without.physicochemical)
        and np.array_equal(with_ligand.distance_matrix, without.distance_matrix)
        and with_ligand.labels.sum() > 0
        and without.labels.sum() == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
