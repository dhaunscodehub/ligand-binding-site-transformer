"""Metrics, and the AUROC-under-imbalance behaviour they are designed to expose."""

from __future__ import annotations

import numpy as np
import pytest

from bindsite.evaluate import (
    MIN_RESIDUES_PER_PROTEIN, compare_models, evaluate_predictions,
    protein_metrics, residue_metrics, top_l_over_n_metrics,
)
from bindsite.exceptions import DataError


def test_perfect_ranking_scores_one():
    labels = np.array([0, 0, 1, 1])
    metrics = residue_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.auprc == pytest.approx(1.0)
    assert metrics.best_f1 == pytest.approx(1.0)


def test_random_scores_sit_near_chance():
    rng = np.random.default_rng(0)
    labels = (rng.random(4000) < 0.1).astype(int)
    metrics = residue_metrics(labels, rng.random(4000))
    assert abs(metrics.auroc - 0.5) < 0.05
    assert abs(metrics.auprc - metrics.positive_rate) < 0.04


def test_auprc_chance_is_the_positive_rate():
    """The reference point that makes an AUPRC value interpretable."""
    rng = np.random.default_rng(1)
    labels = (rng.random(5000) < 0.2).astype(int)
    metrics = residue_metrics(labels, rng.random(5000))
    assert metrics.to_dict()["auprc_chance"] == pytest.approx(0.2, abs=0.02)


def test_auprc_lift_is_reported():
    labels = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.zeros(90), np.ones(10)])
    metrics = residue_metrics(labels, scores)
    assert metrics.auprc_lift == pytest.approx(1.0 / 0.1, rel=0.05)


def test_auroc_stays_high_while_precision_is_poor():
    """The central point: AUROC's false-positive-rate denominator is the large
    negative class, so many false positives barely move it."""
    rng = np.random.default_rng(2)
    n = 20000
    labels = (rng.random(n) < 0.05).astype(int)
    scores = rng.normal(0, 1, n) + labels * 1.5
    metrics = residue_metrics(labels, 1 / (1 + np.exp(-scores)))
    assert metrics.auroc > 0.8
    assert metrics.auprc < 0.5
    assert metrics.precision_at_best_f1 < 0.6


def test_undefined_auroc_returns_none():
    metrics = residue_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert metrics.auroc is None
    assert metrics.auprc is None


def test_mismatched_lengths_are_rejected():
    with pytest.raises(DataError, match="scores"):
        residue_metrics(np.array([0, 1]), np.array([0.5]))


def test_empty_input_is_rejected():
    with pytest.raises(DataError, match="no residues"):
        residue_metrics(np.array([]), np.array([]))


def test_non_finite_scores_are_rejected():
    with pytest.raises(DataError, match="non-finite"):
        residue_metrics(np.array([0, 1]), np.array([0.5, np.nan]))


def test_top_l_over_n_precision_and_recall():
    labels = np.zeros(100, dtype=int)
    labels[:10] = 1
    scores = np.zeros(100)
    scores[:10] = 1.0  # all positives ranked first
    precision, recall = top_l_over_n_metrics(labels, scores, divisor=10)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_top_l_over_n_adapts_to_length():
    labels = np.zeros(50, dtype=int)
    labels[0] = 1
    scores = np.linspace(1, 0, 50)
    precision, _ = top_l_over_n_metrics(labels, scores, divisor=10)
    assert precision == pytest.approx(1 / 5)


def test_top_l_over_n_is_none_without_positives():
    precision, recall = top_l_over_n_metrics(np.zeros(20, dtype=int), np.zeros(20))
    assert precision is None and recall is None


def test_protein_metrics_report_each_protein():
    labels = {f"P{i}": np.array([1] * 5 + [0] * 25) for i in range(3)}
    scores = {f"P{i}": np.concatenate([np.ones(5), np.zeros(25)]) for i in range(3)}
    report = protein_metrics(labels, scores)
    assert report.n_proteins == 3
    assert report.mean_auprc == pytest.approx(1.0)
    assert report.std_auprc == pytest.approx(0.0)


def test_protein_metrics_exclude_and_list_short_proteins():
    labels = {
        "big": np.array([1] * 5 + [0] * 25),
        "short": np.array([1, 0, 1]),
    }
    scores = {
        "big": np.concatenate([np.ones(5), np.zeros(25)]),
        "short": np.array([0.9, 0.1, 0.8]),
    }
    report = protein_metrics(labels, scores)
    assert report.n_proteins == 1
    assert "short" in report.excluded


def test_protein_metrics_exclude_proteins_without_positives():
    labels = {
        "ok": np.array([1] * 5 + [0] * 25),
        "none": np.zeros(30, dtype=int),
    }
    scores = {k: np.random.default_rng(0).random(v.size) for k, v in labels.items()}
    report = protein_metrics(labels, scores)
    assert report.excluded["none"] == "no binding residues"


def test_protein_metrics_reject_when_nothing_qualifies():
    labels = {"a": np.array([1, 0, 1])}
    scores = {"a": np.array([0.9, 0.1, 0.8])}
    with pytest.raises(DataError, match="no protein has at least"):
        protein_metrics(labels, scores)


def test_protein_metrics_reject_mismatched_lengths():
    with pytest.raises(DataError, match="labels but"):
        protein_metrics(
            {"a": np.zeros(30, dtype=int)}, {"a": np.zeros(20)}
        )


def test_min_residues_per_protein_is_documented():
    assert MIN_RESIDUES_PER_PROTEIN >= 5


def test_evaluate_reports_both_levels():
    labels = {f"P{i}": np.array([1] * 5 + [0] * 25) for i in range(3)}
    scores = {f"P{i}": np.concatenate([np.ones(5), np.zeros(25)]) for i in range(3)}
    result = evaluate_predictions("m", labels, scores, split_strategy="homology_cluster")
    record = result.to_dict()
    assert record["residue_level"]["level"] == "residue_pooled"
    assert record["protein_level"]["level"] == "per_protein"


def test_evaluate_always_attaches_the_auroc_caveat():
    labels = {f"P{i}": np.array([1] * 3 + [0] * 27) for i in range(3)}
    scores = {f"P{i}": np.random.default_rng(i).random(30) for i in range(3)}
    result = evaluate_predictions("m", labels, scores)
    assert any("not a sufficient summary" in c for c in result.caveats)


def test_random_protein_split_gets_its_own_caveat():
    labels = {f"P{i}": np.array([1] * 5 + [0] * 25) for i in range(3)}
    scores = {f"P{i}": np.random.default_rng(i).random(30) for i in range(3)}
    result = evaluate_predictions("m", labels, scores, split_strategy="random_protein")
    assert any("homologous proteins" in c for c in result.caveats)


def test_random_residue_split_gets_the_strongest_caveat():
    labels = {f"P{i}": np.array([1] * 5 + [0] * 25) for i in range(3)}
    scores = {f"P{i}": np.random.default_rng(i).random(30) for i in range(3)}
    result = evaluate_predictions("m", labels, scores, split_strategy="random_residue")
    assert any("most severe leak" in c for c in result.caveats)


def test_pooled_minus_per_protein_is_reported():
    labels = {
        "big": np.array([1] * 20 + [0] * 80),
        "small": np.array([1] * 2 + [0] * 28),
    }
    scores = {
        "big": np.concatenate([np.ones(20), np.zeros(80)]),
        "small": np.concatenate([np.zeros(2), np.ones(28)]),  # wrong on purpose
    }
    result = evaluate_predictions("m", labels, scores)
    assert result.pooled_minus_per_protein_auprc is not None


def test_evaluate_rejects_empty_input():
    with pytest.raises(DataError, match="no proteins"):
        evaluate_predictions("m", {}, {})


def test_evaluate_rejects_when_no_protein_has_both():
    with pytest.raises(DataError, match="both labels and scores"):
        evaluate_predictions("m", {"a": np.array([1, 0])}, {"b": np.array([0.5, 0.5])})


def test_compare_models_ranks_by_auprc():
    labels = {f"P{i}": np.array([1] * 5 + [0] * 25) for i in range(3)}
    good = {f"P{i}": np.concatenate([np.ones(5), np.zeros(25)]) for i in range(3)}
    bad = {f"P{i}": np.concatenate([np.zeros(5), np.ones(25)]) for i in range(3)}
    results = [
        evaluate_predictions("bad", labels, bad),
        evaluate_predictions("good", labels, good),
    ]
    comparison = compare_models(results)
    assert comparison["ranked_by"] == "auprc"
    assert comparison["best_model"] == "good"


def test_compare_models_reports_baseline_auprc():
    labels = {f"P{i}": np.array([1] * 5 + [0] * 25) for i in range(3)}
    scores = {f"P{i}": np.random.default_rng(i).random(30) for i in range(3)}
    results = [
        evaluate_predictions("prevalence", labels, {k: np.full(30, 0.16) for k in labels}),
        evaluate_predictions("transformer", labels, scores),
    ]
    comparison = compare_models(results)
    assert "prevalence" in comparison["baseline_auprc"]


def test_compare_models_rejects_empty():
    with pytest.raises(DataError, match="no results"):
        compare_models([])
