"""End-to-end pipeline: build, cluster, split, train, evaluate.

The order is fixed by leakage control:

1. **Build and curate** the dataset. Ligand curation happens here and nowhere
   else, so every downstream label derives from one rule.
2. **Cluster** by sequence identity. Cached, because it is the expensive step
   and it does not depend on anything downstream.
3. **Split by cluster**, then assert the split against the sequences.
4. **Fit** baselines and the transformer on the training fold only.
5. **Evaluate** at residue and protein level on the test fold.

Step 3's assertion runs before any model is fitted. Discovering a leaking
split after training wastes the training, and — worse — invites the
temptation to keep the numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import RunConfig
from .data.dataset import Dataset, build_dataset, search_candidates
from .data.splits import (
    Split, assert_no_homology_leakage, homology_split, label_balance,
    random_split,
)
from .evaluate import EvaluationResult, compare_models, evaluate_predictions
from .exceptions import DataError
from .homology import Clustering, cluster_coherence, cluster_sequences
from .io_utils import provenance, write_json


@dataclass
class PipelineResult:
    """Everything one run produced."""

    name: str
    dataset_statistics: dict
    clustering: dict
    split: dict
    label_balance: dict
    evaluations: list[EvaluationResult] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    training_history: dict | None = None
    warnings: list[str] = field(default_factory=list)
    leakage_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dataset": self.dataset_statistics,
            "clustering": self.clustering,
            "split": self.split,
            "label_balance": self.label_balance,
            "leakage_verification": self.leakage_report,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "comparison": self.comparison,
            "training_history": self.training_history,
            "warnings": self.warnings,
            "provenance": provenance(),
        }


def prepare_dataset(config: RunConfig, progress: bool = False) -> Dataset:
    """Build or load the curated dataset."""
    from .chemcomp import ComponentCache

    cache = ComponentCache(
        Path(config.data.cache_dir).parent / "chemcomp_cache.json"
    )

    if config.data.source == "synthetic":
        raise DataError(
            "the synthetic source produces feature records, not structures; "
            "use bindsite.testing.synthetic_features directly, or the "
            "validate command, which uses it for the positive control"
        )

    if config.data.source == "manifest":
        manifest = json.loads(Path(config.data.manifest).read_text())
        identifiers = sorted(manifest.get("proteins", {}))
        filters = manifest.get("statistics", {}).get("filters", {})
        if not identifiers:
            raise DataError(f"{config.data.manifest}: manifest lists no proteins")
    elif config.data.source == "ids":
        identifiers, filters = list(config.data.pdb_ids), {"source": "explicit ids"}
    else:
        identifiers, filters = search_candidates(
            limit=config.data.n_candidates,
            max_resolution=config.data.max_resolution,
            min_length=config.data.min_length,
            max_length=config.data.max_length,
            pool_size=config.data.pool_size,
            seed=config.data.sample_seed,
        )

    dataset = build_dataset(
        identifiers, cache_dir=config.data.cache_dir,
        cutoff=config.data.contact_cutoff,
        min_ligand_atoms=config.data.min_ligand_heavy_atoms,
        min_length=config.data.min_length, max_length=config.data.max_length,
        min_binding_residues=config.data.min_binding_residues,
        max_proteins=config.data.max_proteins, filters=filters,
        progress=progress, component_cache=cache,
    )
    cache.save()
    dataset.component_cache_report = cache.report()
    if len(dataset) < 10:
        raise DataError(
            f"only {len(dataset)} proteins survived curation, too few to split "
            f"by homology cluster. Rejections: {dataset.statistics()['rejection_reasons']}"
        )
    return dataset


def prepare_identity_matrix(dataset: Dataset, config: RunConfig):
    """Compute or load the pairwise identity matrix.

    Cached as a separate artifact from the clustering because it is the
    expensive part and it does not depend on the threshold: a threshold sweep,
    or the direct split verification, reuses it without realigning anything.
    """
    from .homology import IdentityMatrix, identity_matrix

    sequences = dataset.sequences()
    path = (
        Path(config.homology.cache).with_name("identity_matrix.npz")
        if config.homology.cache else None
    )
    if path and path.is_file():
        try:
            matrix = IdentityMatrix.load(path)
        except (OSError, KeyError, ValueError):
            matrix = None
        # Only reuse a matrix that covers exactly these sequences and was
        # built in the same alignment mode; a stale or partial one would
        # silently weaken the verification.
        if matrix is not None and matrix.covers(sequences) and matrix.mode == "local":
            return matrix

    matrix = identity_matrix(sequences, mode="local")
    if path:
        matrix.save(path)
    return matrix


def prepare_clustering(
    dataset: Dataset, config: RunConfig, measure_coherence: bool = True,
    matrix=None,
) -> Clustering:
    """Cluster the dataset's sequences, using a cache when one is valid."""
    sequences = dataset.sequences()
    cache_path = Path(config.homology.cache) if config.homology.cache else None

    if cache_path and cache_path.is_file():
        payload = json.loads(cache_path.read_text())
        labels = {k: int(v) for k, v in payload.get("labels", {}).items()}
        # A cache is only usable if it covers exactly this dataset at exactly
        # this threshold. Reusing a clustering built at a different threshold
        # would silently change what the split guarantees.
        same_threshold = (
            abs(payload.get("identity_threshold", -1) - config.homology.identity_threshold)
            < 1e-9
            and abs(
                payload.get("coverage_threshold", -1)
                - config.homology.coverage_threshold
            ) < 1e-9
        )
        # The backend and linkage must round-trip too. Without them the
        # reloaded clustering reports the dataclass defaults ("centroid",
        # no edge count), which misdescribes how it was actually produced —
        # and how a clustering was produced is exactly what a reader needs to
        # judge whether the split is sound.
        expected_backend = (
            "connected_components"
            if config.homology.backend in ("auto", "connected_components")
            else config.homology.backend
        )
        same_backend = payload.get("backend") == expected_backend
        if same_threshold and same_backend and set(labels) == set(sequences):
            return Clustering(
                labels=labels,
                identity_threshold=payload["identity_threshold"],
                coverage_threshold=payload["coverage_threshold"],
                backend=payload["backend"],
                representatives={
                    int(k): v for k, v in payload.get("representatives", {}).items()
                },
                coherence=payload.get("coherence"),
                linkage=payload.get("linkage", "unrecorded"),
                n_edges=payload.get("n_edges"),
            )

    if matrix is not None and config.homology.backend in ("auto", "connected_components"):
        from .homology import cluster_from_matrix

        clustering = cluster_from_matrix(
            matrix, identity=config.homology.identity_threshold,
            coverage=config.homology.coverage_threshold,
        )
    else:
        clustering = cluster_sequences(
            sequences, identity=config.homology.identity_threshold,
            coverage=config.homology.coverage_threshold,
            backend=config.homology.backend, work_dir=config.homology.work_dir,
            threads=config.homology.threads,
        )
    if measure_coherence:
        clustering.coherence = cluster_coherence(clustering, sequences, seed=0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_path, {
            "labels": clustering.labels,
            "identity_threshold": clustering.identity_threshold,
            "coverage_threshold": clustering.coverage_threshold,
            "backend": clustering.backend,
            "linkage": clustering.linkage,
            "n_edges": clustering.n_edges,
            "representatives": {
                str(k): v for k, v in clustering.representatives.items()
            },
            "coherence": clustering.coherence,
            "report": clustering.to_dict(),
        })
    return clustering


def build_split(
    dataset: Dataset, clustering: Clustering, config: RunConfig, matrix=None
) -> tuple[Split, dict]:
    """Construct the split and verify it before anything is fitted."""
    if config.split.strategy == "homology":
        split = homology_split(
            clustering, test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction, seed=config.split.seed,
        )
        report = assert_no_homology_leakage(
            split, clustering,
            sequences=dataset.sequences() if config.split.verify_identity else None,
            matrix=matrix if config.split.verify_identity else None,
        )
    elif config.split.strategy == "random_protein":
        split = random_split(
            dataset.names, clustering,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction, seed=config.split.seed,
        )
        report = {
            "strategy": "random_protein", "verified": False,
            "note": (
                "no homology verification: this split deliberately ignores "
                "homology and is reported only as a comparison arm"
            ),
            "shared_clusters_train_test": split.extra.get(
                "shared_clusters_train_test"
            ),
        }
    else:
        raise DataError(
            f"split strategy {config.split.strategy!r} is not runnable in the "
            "pipeline; residue-level splitting exists only inside the control "
            "experiments, where its purpose is to be measured, not used"
        )
    return split, report


def _stack(dataset: Dataset, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    blocks, labels = [], []
    for name in names:
        record = dataset.entries[name].features
        blocks.append(np.hstack([record.sequence_block, record.structure_block]))
        labels.append(record.labels.astype(int))
    return np.vstack(blocks), np.concatenate(labels)


def run_pipeline(
    config: RunConfig,
    output_dir: str | Path | None = None,
    progress: bool = False,
) -> PipelineResult:
    """Execute the full pipeline for one configuration."""
    from .features import feature_names
    from .models.baselines import fit_baseline

    warnings: list[str] = []
    dataset = prepare_dataset(config, progress=progress)
    matrix = prepare_identity_matrix(dataset, config)
    clustering = prepare_clustering(dataset, config, matrix=matrix)

    if clustering.coherence and clustering.coherence.get("n_clusters_showing_chaining"):
        chained = clustering.coherence
        warnings.append(
            f"{chained['n_clusters_showing_chaining']} of "
            f"{chained['n_multi_member_clusters']} multi-member clusters contain "
            f"pairs below the {clustering.identity_threshold:.0%} identity "
            f"threshold (minimum observed "
            f"{chained['min_within_cluster_identity']:.3f}). Centroid linkage "
            "compares each sequence only to a cluster representative, so it "
            "over-merges. This is conservative for leakage — unrelated "
            "proteins sharing a fold never split a homologous pair — but it "
            "coarsens the split and reduces the number of independent test "
            "families."
        )

    split, leakage_report = build_split(dataset, clustering, config, matrix=matrix)
    largest = clustering.to_dict().get("largest_cluster_fraction", 0.0)
    if largest > 0.15:
        warnings.append(
            f"the largest homology cluster holds {largest:.1%} of the proteins, "
            "so the realised fold fractions deviate from those requested: a "
            "cluster moves as a unit"
        )

    names = feature_names()
    all_names = (
        list(names["sequence"]) + list(names["physicochemical"])
        + list(names["geometric"])
    )
    train_x, train_y = _stack(dataset, split.train)
    labels_by_protein = {
        name: dataset.entries[name].features.labels for name in split.test
    }

    evaluations: list[EvaluationResult] = []
    for model in config.models.baselines:
        try:
            fitted = fit_baseline(
                model, train_x, train_y, all_names, seed=config.training.seed
            )
        except DataError as error:
            warnings.append(f"baseline {model} skipped: {error}")
            continue
        scores_by_protein = {}
        for name in split.test:
            record = dataset.entries[name].features
            block = np.hstack([record.sequence_block, record.structure_block])
            scores_by_protein[name] = fitted.predict_proba(block)
        evaluations.append(
            evaluate_predictions(
                model, labels_by_protein, scores_by_protein,
                split_strategy=split.strategy,
            )
        )

    history: dict | None = None
    if config.models.transformer:
        try:
            from .train import predict, train_model

            train_features = [dataset.entries[n].features for n in split.train]
            val_features = [dataset.entries[n].features for n in split.val]
            if not val_features:
                raise DataError(
                    "the transformer needs a validation fold for early "
                    "stopping; set split.val_fraction above 0"
                )
            model, training_history = train_model(
                train_features, val_features, config.architecture,
                config.training, verbose=progress,
            )
            history = training_history.to_dict()
            scores = predict(
                model, [dataset.entries[n].features for n in split.test],
                config.architecture, batch_size=config.training.batch_size,
            )
            evaluations.append(
                evaluate_predictions(
                    "transformer", labels_by_protein, scores,
                    split_strategy=split.strategy,
                )
            )
        except ImportError as error:
            warnings.append(
                f"transformer skipped, PyTorch unavailable: {error}"
            )
        except DataError as error:
            warnings.append(f"transformer skipped: {error}")

    if not evaluations:
        raise DataError("no model produced predictions; see warnings")

    result = PipelineResult(
        name=config.name,
        dataset_statistics=dataset.statistics(),
        clustering={
            **clustering.to_dict(),
            "pairwise_identity_distribution": matrix.distribution(),
        },
        split=split.summary(),
        label_balance=label_balance(dataset.labels(), split),
        evaluations=evaluations,
        comparison=compare_models(evaluations),
        training_history=history,
        warnings=warnings,
        leakage_report=leakage_report,
    )
    write_json(
        Path(output_dir or config.output_dir) / f"{config.name}.json",
        result.to_dict(),
    )
    return result
