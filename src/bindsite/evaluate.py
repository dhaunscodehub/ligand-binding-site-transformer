"""Metrics for per-residue binding-site prediction.

**AUROC is the wrong headline metric for this task, and this module is built
to make that visible rather than to argue it.**

Binding residues are roughly 5-15% of a protein. ROC curves use the false
positive rate, whose denominator is the large negative class, so a model can
produce many more false positives than true positives and still show a high
AUROC. A predictor with AUROC 0.82 on a 12%-positive task can have precision
near 0.3. AUPRC uses precision, whose denominator is the model's own positive
predictions, so it does not have this property — and its chance level equals
the positive rate, which is reported alongside every value here.

The binding-site literature reports AUROC widely, so it is computed for
comparability. It is never reported without AUPRC, the positive rate, and the
AUPRC lift over chance next to it.

Two levels, kept separate:

**Residue level, pooled.** All residues from all proteins in one pool. This is
what "0.82 AUROC" normally means. It is dominated by large proteins and by
proteins with many binding residues.

**Protein level.** The metric is computed per protein and then averaged, so
each protein counts once. This is what matters in use — a tool is applied to
one protein at a time — and it is usually lower and more variable. The gap
between the two is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .exceptions import DataError

# Fewer than this many residues, or no positives, makes a per-protein metric
# uninformative; such proteins are counted and excluded rather than averaged in.
MIN_RESIDUES_PER_PROTEIN = 20


@dataclass
class ResidueMetrics:
    """Pooled residue-level metrics."""

    n_residues: int
    n_positive: int
    positive_rate: float
    auroc: float | None
    auprc: float | None
    auprc_lift: float | None
    best_f1: float
    best_f1_threshold: float
    mcc_at_best_f1: float
    precision_at_best_f1: float
    recall_at_best_f1: float
    precision_at_top_l10: float | None = None
    recall_at_top_l10: float | None = None

    def to_dict(self) -> dict:
        return {
            "level": "residue_pooled", "n_residues": self.n_residues,
            "n_positive": self.n_positive, "positive_rate": self.positive_rate,
            "auroc": self.auroc, "auprc": self.auprc,
            "auprc_chance": self.positive_rate,
            "auprc_lift_over_chance": self.auprc_lift,
            "best_f1": self.best_f1,
            "best_f1_threshold": self.best_f1_threshold,
            "precision_at_best_f1": self.precision_at_best_f1,
            "recall_at_best_f1": self.recall_at_best_f1,
            "mcc_at_best_f1": self.mcc_at_best_f1,
            "precision_at_top_L10": self.precision_at_top_l10,
            "recall_at_top_L10": self.recall_at_top_l10,
        }


@dataclass
class ProteinMetrics:
    """Per-protein metrics and their spread."""

    n_proteins: int
    mean_auroc: float | None
    std_auroc: float | None
    mean_auprc: float | None
    std_auprc: float | None
    median_auprc: float | None
    mean_precision_at_top_l10: float | None
    per_protein: dict[str, dict] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "level": "per_protein", "n_proteins": self.n_proteins,
            "mean_auroc": self.mean_auroc, "std_auroc": self.std_auroc,
            "mean_auprc": self.mean_auprc, "std_auprc": self.std_auprc,
            "median_auprc": self.median_auprc,
            "mean_precision_at_top_L10": self.mean_precision_at_top_l10,
            "n_excluded": len(self.excluded),
            "excluded": self.excluded,
            "min_residues_per_protein": MIN_RESIDUES_PER_PROTEIN,
            "per_protein": self.per_protein,
        }


@dataclass
class EvaluationResult:
    """Both levels for one model on one split, plus the caveats that apply."""

    model: str
    split_strategy: str
    residue: ResidueMetrics
    protein: ProteinMetrics | None = None
    caveats: list[str] = field(default_factory=list)

    @property
    def pooled_minus_per_protein_auprc(self) -> float | None:
        """How much pooling flatters the model relative to per-protein AUPRC."""
        if self.protein is None or self.residue.auprc is None:
            return None
        if self.protein.mean_auprc is None:
            return None
        return self.residue.auprc - self.protein.mean_auprc

    def to_dict(self) -> dict:
        return {
            "model": self.model, "split_strategy": self.split_strategy,
            "residue_level": self.residue.to_dict(),
            "protein_level": self.protein.to_dict() if self.protein else None,
            "pooled_minus_per_protein_auprc": self.pooled_minus_per_protein_auprc,
            "caveats": self.caveats,
        }


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC, or ``None`` when it is undefined (one class present)."""
    from sklearn.metrics import roc_auc_score

    if labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(roc_auc_score(labels, scores))


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Average precision, or ``None`` when no positive exists."""
    from sklearn.metrics import average_precision_score

    if labels.sum() == 0:
        return None
    return float(average_precision_score(labels, scores))


def top_l_over_n_metrics(
    labels: np.ndarray, scores: np.ndarray, divisor: int = 10
) -> tuple[float | None, float | None]:
    """Precision and recall in the top L/n scored residues.

    The metric a practitioner actually cares about: given a protein of length
    L, take the L/10 highest-scoring residues and ask how many are truly binding.
    Unlike a fixed threshold it adapts to protein size, and unlike AUROC it
    reflects what you get when you look at a short list.
    """
    length = labels.size
    if length == 0 or labels.sum() == 0:
        return None, None
    k = max(1, length // divisor)
    top = np.argsort(-scores)[:k]
    hits = int(labels[top].sum())
    return hits / k, hits / int(labels.sum())


def residue_metrics(
    labels: Sequence[bool] | np.ndarray, scores: Sequence[float] | np.ndarray
) -> ResidueMetrics:
    """Pooled residue-level metrics, with AUPRC reported against chance."""
    from sklearn.metrics import (
        matthews_corrcoef, precision_recall_curve,
    )

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if labels.size != scores.size:
        raise DataError(f"{labels.size} labels but {scores.size} scores")
    if labels.size == 0:
        raise DataError("no residues to evaluate")
    if not np.isfinite(scores).all():
        raise DataError("scores contain non-finite values")

    rate = float(labels.mean())
    auroc = _safe_auroc(labels, scores)
    auprc = _safe_auprc(labels, scores)

    best_f1, best_threshold = 0.0, 0.5
    precision_at_best, recall_at_best = 0.0, 0.0
    if labels.sum() > 0:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # precision_recall_curve returns one more point than thresholds.
        denominator = precision[:-1] + recall[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(
                denominator > 0,
                2 * precision[:-1] * recall[:-1] / np.maximum(denominator, 1e-12),
                0.0,
            )
        if f1.size:
            position = int(np.argmax(f1))
            best_f1 = float(f1[position])
            best_threshold = float(thresholds[position])
            precision_at_best = float(precision[position])
            recall_at_best = float(recall[position])

    predicted = (scores >= best_threshold).astype(int)
    mcc = (
        float(matthews_corrcoef(labels, predicted))
        if len(set(predicted.tolist())) > 1 and labels.sum() > 0 else 0.0
    )
    precision_top, recall_top = top_l_over_n_metrics(labels, scores)

    return ResidueMetrics(
        n_residues=int(labels.size), n_positive=int(labels.sum()),
        positive_rate=rate, auroc=auroc, auprc=auprc,
        # Lift is the interpretable form: AUPRC 0.30 at a 12% positive rate is
        # 2.5x chance, while the raw 0.30 looks poor next to an AUROC of 0.82.
        auprc_lift=(auprc / rate if auprc is not None and rate > 0 else None),
        best_f1=best_f1, best_f1_threshold=best_threshold,
        mcc_at_best_f1=mcc, precision_at_best_f1=precision_at_best,
        recall_at_best_f1=recall_at_best,
        precision_at_top_l10=precision_top, recall_at_top_l10=recall_top,
    )


def protein_metrics(
    labels_per_protein: dict[str, np.ndarray],
    scores_per_protein: dict[str, np.ndarray],
) -> ProteinMetrics:
    """Per-protein metrics and their spread across proteins."""
    per_protein: dict[str, dict] = {}
    excluded: dict[str, str] = {}

    for name in sorted(labels_per_protein):
        if name not in scores_per_protein:
            excluded[name] = "no scores"
            continue
        labels = np.asarray(labels_per_protein[name]).astype(int)
        scores = np.asarray(scores_per_protein[name], dtype=float)
        if labels.size != scores.size:
            raise DataError(
                f"{name}: {labels.size} labels but {scores.size} scores"
            )
        if labels.size < MIN_RESIDUES_PER_PROTEIN:
            excluded[name] = f"only {labels.size} residues"
            continue
        if labels.sum() == 0:
            excluded[name] = "no binding residues"
            continue
        if labels.sum() == labels.size:
            excluded[name] = "every residue is binding"
            continue

        precision_top, recall_top = top_l_over_n_metrics(labels, scores)
        per_protein[name] = {
            "n_residues": int(labels.size),
            "n_positive": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "auroc": _safe_auroc(labels, scores),
            "auprc": _safe_auprc(labels, scores),
            "precision_at_top_L10": precision_top,
            "recall_at_top_L10": recall_top,
        }

    if not per_protein:
        raise DataError(
            f"no protein has at least {MIN_RESIDUES_PER_PROTEIN} residues with "
            f"both classes present (excluded: {excluded}); per-protein metrics "
            "are not meaningful here"
        )

    def collect(key: str) -> np.ndarray:
        return np.array(
            [v[key] for v in per_protein.values() if v[key] is not None], dtype=float
        )

    aurocs, auprcs = collect("auroc"), collect("auprc")
    tops = collect("precision_at_top_L10")
    return ProteinMetrics(
        n_proteins=len(per_protein),
        mean_auroc=float(aurocs.mean()) if aurocs.size else None,
        std_auroc=float(aurocs.std()) if aurocs.size else None,
        mean_auprc=float(auprcs.mean()) if auprcs.size else None,
        std_auprc=float(auprcs.std()) if auprcs.size else None,
        median_auprc=float(np.median(auprcs)) if auprcs.size else None,
        mean_precision_at_top_l10=float(tops.mean()) if tops.size else None,
        per_protein=per_protein, excluded=excluded,
    )


def evaluate_predictions(
    model: str,
    labels_per_protein: dict[str, np.ndarray],
    scores_per_protein: dict[str, np.ndarray],
    split_strategy: str = "unspecified",
) -> EvaluationResult:
    """Evaluate one model at both levels, attaching the applicable caveats."""
    if not labels_per_protein:
        raise DataError("no proteins to evaluate")

    names = [n for n in sorted(labels_per_protein) if n in scores_per_protein]
    if not names:
        raise DataError("no protein has both labels and scores")

    pooled_labels = np.concatenate([labels_per_protein[n] for n in names])
    pooled_scores = np.concatenate([scores_per_protein[n] for n in names])
    residue = residue_metrics(pooled_labels, pooled_scores)

    protein: ProteinMetrics | None = None
    caveats: list[str] = []
    try:
        protein = protein_metrics(
            {n: labels_per_protein[n] for n in names},
            {n: scores_per_protein[n] for n in names},
        )
    except DataError as error:
        caveats.append(f"per-protein metrics unavailable: {error}")

    caveats.append(
        f"AUROC is reported for comparability with the literature, but at a "
        f"{residue.positive_rate:.1%} positive rate it is not a sufficient "
        "summary: the false-positive-rate denominator is the large negative "
        "class, so a high AUROC is compatible with low precision. AUPRC "
        f"(chance = {residue.positive_rate:.3f}) and precision in the top L/10 "
        "are the metrics that reflect usable output."
    )
    if split_strategy == "random_protein":
        caveats.append(
            "random protein split: homologous proteins may appear in both "
            "train and test, so these numbers are optimistic and are reported "
            "only for comparison against the homology-separated split"
        )
    elif split_strategy == "random_residue":
        caveats.append(
            "random residue split: residues of the same protein appear in "
            "train and test. Neighbouring residues share nearly all features, "
            "so this is the most severe leak available in this task and the "
            "numbers are not an estimate of anything useful"
        )

    return EvaluationResult(
        model=model, split_strategy=split_strategy, residue=residue,
        protein=protein, caveats=caveats,
    )


def compare_models(results: Sequence[EvaluationResult]) -> dict:
    """Rank models by AUPRC and state whether each beats the baselines.

    Ranked by AUPRC rather than AUROC deliberately: AUROC compresses the
    differences that matter under class imbalance, and two models differing by
    0.01 AUROC can differ by 0.10 AUPRC.
    """
    if not results:
        raise DataError("no results to compare")

    rows = [
        {
            "model": r.model, "split_strategy": r.split_strategy,
            "auprc": r.residue.auprc, "auroc": r.residue.auroc,
            "auprc_lift": r.residue.auprc_lift,
            "best_f1": r.residue.best_f1,
            "precision_at_top_L10": r.residue.precision_at_top_l10,
            "per_protein_mean_auprc": (
                r.protein.mean_auprc if r.protein else None
            ),
        }
        for r in results
    ]
    rows.sort(key=lambda row: -(row["auprc"] or 0.0))

    by_name = {r.model: r for r in results}
    floors = {
        name: by_name[name].residue.auprc
        for name in ("prevalence", "burial", "logistic", "random_forest")
        if name in by_name
    }
    beats = {
        r.model: {
            floor: bool((r.residue.auprc or 0.0) > (value or 0.0))
            for floor, value in floors.items()
            if floor != r.model
        }
        for r in results
        if r.model not in floors or r.model != "prevalence"
    }

    return {
        "ranked_by": "auprc",
        "ranking": rows,
        "best_model": rows[0]["model"],
        "baseline_auprc": floors,
        "beats_baselines": beats,
        "note": (
            "ranked by AUPRC: under class imbalance AUROC compresses exactly "
            "the differences that matter, and a model must beat the "
            "per-residue baselines to have justified its context modelling"
        ),
    }
