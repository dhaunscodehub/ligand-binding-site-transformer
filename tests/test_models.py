"""Baselines and the transformer."""

from __future__ import annotations

import numpy as np
import pytest

from bindsite.exceptions import ConfigError, DataError
from bindsite.features import feature_names
from bindsite.models.baselines import BASELINE_MODELS, fit_baseline
from bindsite.models.transformer import TransformerConfig

torch = pytest.importorskip("torch")

ALL_NAMES = (
    list(feature_names()["sequence"]) + list(feature_names()["physicochemical"])
    + list(feature_names()["geometric"])
)


def _stacked(records):
    blocks, labels = [], []
    for record in records.values():
        blocks.append(np.hstack([record.sequence_block, record.structure_block]))
        labels.append(record.labels.astype(int))
    return np.vstack(blocks), np.concatenate(labels)


@pytest.mark.parametrize("name", BASELINE_MODELS)
def test_every_baseline_fits_and_scores(name, features_signal):
    features, labels = _stacked(features_signal)
    fitted = fit_baseline(name, features, labels, ALL_NAMES, seed=0)
    scores = fitted.predict_proba(features)
    assert scores.shape == (features.shape[0],)
    assert np.isfinite(scores).all()


def test_prevalence_baseline_is_constant(features_signal):
    features, labels = _stacked(features_signal)
    fitted = fit_baseline("prevalence", features, labels, ALL_NAMES)
    scores = fitted.predict_proba(features)
    assert np.allclose(scores, labels.mean())


def test_prevalence_baseline_has_chance_auroc(features_signal):
    """A constant predictor has no ranking information, so AUROC is 0.5."""
    from bindsite.evaluate import residue_metrics

    features, labels = _stacked(features_signal)
    fitted = fit_baseline("prevalence", features, labels, ALL_NAMES)
    metrics = residue_metrics(labels, fitted.predict_proba(features))
    assert metrics.auroc == pytest.approx(0.5)
    assert metrics.auprc == pytest.approx(metrics.positive_rate, abs=0.02)


def test_burial_baseline_uses_one_feature(features_signal):
    features, labels = _stacked(features_signal)
    fitted = fit_baseline("burial", features, labels, ALL_NAMES)
    assert ALL_NAMES[int(fitted.estimator)] == "neighbour_count"


def test_burial_baseline_needs_its_feature(features_signal):
    features, labels = _stacked(features_signal)
    with pytest.raises(DataError, match="neighbour_count"):
        fit_baseline(
            "burial", features, labels, [f"f{i}" for i in range(features.shape[1])]
        )


def test_learned_baselines_beat_prevalence_on_signal(features_signal):
    from bindsite.evaluate import residue_metrics

    features, labels = _stacked(features_signal)
    floor = residue_metrics(
        labels, fit_baseline("prevalence", features, labels, ALL_NAMES).predict_proba(features)
    ).auprc
    for name in ("logistic", "random_forest"):
        fitted = fit_baseline(name, features, labels, ALL_NAMES, seed=0)
        auprc = residue_metrics(labels, fitted.predict_proba(features)).auprc
        assert auprc > floor, f"{name} did not beat the prevalence floor"


def test_unknown_baseline_is_rejected(features_signal):
    features, labels = _stacked(features_signal)
    with pytest.raises(DataError, match="unknown baseline"):
        fit_baseline("magic", features, labels, ALL_NAMES)


def test_mismatched_rows_are_rejected(features_signal):
    features, labels = _stacked(features_signal)
    with pytest.raises(DataError, match="labels"):
        fit_baseline("logistic", features, labels[:10], ALL_NAMES)


def test_mismatched_columns_are_rejected(features_signal):
    features, labels = _stacked(features_signal)
    with pytest.raises(DataError, match="names"):
        fit_baseline("logistic", features, labels, ALL_NAMES[:5])


def test_single_class_is_rejected(features_signal):
    features, _ = _stacked(features_signal)
    with pytest.raises(DataError, match="all one class"):
        fit_baseline(
            "logistic", features, np.zeros(features.shape[0], dtype=int), ALL_NAMES
        )


def test_baselines_are_reproducible(features_signal):
    """Reproducible to floating-point tolerance, not bit-identical.

    The forest averages per-tree probabilities in parallel (n_jobs=-1) and the
    reduction order is not fixed, so repeated fits differ by about one ULP
    (measured maximum 4.4e-16 over 3449 residues). That is orders of magnitude
    below the precision of any reported metric.
    """
    features, labels = _stacked(features_signal)
    scores = [
        fit_baseline(
            "random_forest", features, labels, ALL_NAMES, seed=3
        ).predict_proba(features)
        for _ in range(2)
    ]
    assert np.allclose(scores[0], scores[1], atol=1e-12, rtol=0)


def test_logistic_baseline_is_bit_identical(features_signal):
    """The single-threaded baseline has no such caveat."""
    features, labels = _stacked(features_signal)
    scores = [
        fit_baseline(
            "logistic", features, labels, ALL_NAMES, seed=3
        ).predict_proba(features)
        for _ in range(2)
    ]
    assert np.array_equal(scores[0], scores[1])


def test_baseline_record_reports_the_training_rate(features_signal):
    features, labels = _stacked(features_signal)
    fitted = fit_baseline("logistic", features, labels, ALL_NAMES)
    record = fitted.to_dict()
    assert record["train_positive_rate"] == pytest.approx(labels.mean())
    assert record["n_features"] == len(ALL_NAMES)


# --- transformer ---------------------------------------------------------

def test_transformer_forward_shape():
    from bindsite.models.transformer import build_model

    config = TransformerConfig(d_model=64, n_heads=4, n_layers=2)
    model = build_model(config)
    batch, length = 3, 50
    logits = model(
        torch.randn(batch, length, config.sequence_dim),
        torch.randn(batch, length, config.structure_dim),
        torch.rand(batch, length, length) * 30,
        torch.ones(batch, length, dtype=torch.bool),
    )
    assert logits.shape == (batch, length)
    assert torch.isfinite(logits).all()


def test_transformer_gradients_are_finite():
    from bindsite.models.transformer import build_model

    config = TransformerConfig(d_model=64, n_heads=4, n_layers=2)
    model = build_model(config)
    logits = model(
        torch.randn(2, 30, 30), torch.randn(2, 30, 8),
        torch.rand(2, 30, 30) * 20, torch.ones(2, 30, dtype=torch.bool),
    )
    logits.sum().backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)


@pytest.mark.parametrize(
    ("use_sequence", "use_structure"),
    [(True, True), (True, False), (False, True)],
)
def test_every_ablation_runs(use_sequence, use_structure):
    from bindsite.models.transformer import build_model

    config = TransformerConfig(
        d_model=64, n_heads=4, n_layers=2,
        use_sequence=use_sequence, use_structure=use_structure,
    )
    model = build_model(config)
    logits = model(
        torch.randn(2, 30, 30) if use_sequence else None,
        torch.randn(2, 30, 8) if use_structure else None,
        torch.rand(2, 30, 30) * 20,
        torch.ones(2, 30, dtype=torch.bool),
    )
    assert logits.shape == (2, 30)


def test_disabling_both_modalities_is_rejected():
    with pytest.raises(ConfigError, match="at least one modality"):
        TransformerConfig(use_sequence=False, use_structure=False)


def test_head_divisibility_is_enforced():
    with pytest.raises(ConfigError, match="divisible"):
        TransformerConfig(d_model=100, n_heads=3)


@pytest.mark.parametrize("field", ["d_model", "n_heads", "n_layers", "max_length"])
def test_non_positive_dimensions_are_rejected(field):
    with pytest.raises(ConfigError, match="positive"):
        TransformerConfig(**{field: 0})


def test_dropout_range_is_enforced():
    with pytest.raises(ConfigError, match="dropout"):
        TransformerConfig(dropout=1.0)


def test_padding_does_not_change_real_positions():
    """A padded batch must give the same logits for the real residues as an
    unpadded one, or batch composition would change predictions."""
    from bindsite.models.transformer import build_model

    torch.manual_seed(0)
    config = TransformerConfig(d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = build_model(config).eval()

    length = 25
    sequence = torch.randn(1, length, 30)
    structure = torch.randn(1, length, 8)
    distances = torch.rand(1, length, length) * 20

    with torch.no_grad():
        unpadded = model(
            sequence, structure, distances,
            torch.ones(1, length, dtype=torch.bool),
        )

    pad = 15
    padded_sequence = torch.zeros(1, length + pad, 30)
    padded_sequence[:, :length] = sequence
    padded_structure = torch.zeros(1, length + pad, 8)
    padded_structure[:, :length] = structure
    padded_distances = torch.zeros(1, length + pad, length + pad)
    padded_distances[:, :length, :length] = distances
    mask = torch.zeros(1, length + pad, dtype=torch.bool)
    mask[:, :length] = True

    with torch.no_grad():
        padded = model(padded_sequence, padded_structure, padded_distances, mask)

    assert torch.allclose(unpadded, padded[:, :length], atol=1e-4)


def test_sequence_longer_than_max_length_is_rejected():
    from bindsite.models.transformer import build_model

    config = TransformerConfig(d_model=64, n_heads=4, n_layers=1, max_length=20)
    model = build_model(config)
    with pytest.raises(ValueError, match="exceeds max_length"):
        model(
            torch.randn(1, 30, 30), torch.randn(1, 30, 8), None,
            torch.ones(1, 30, dtype=torch.bool),
        )


def test_distance_bias_changes_the_output():
    """If the bias had no effect the model would not be structure-aware."""
    from bindsite.models.transformer import build_model

    torch.manual_seed(0)
    config = TransformerConfig(d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = build_model(config).eval()
    # Perturb the bias projection so it is not at its near-zero init.
    with torch.no_grad():
        model.distance_bias.project.weight.normal_(0, 1.0)

    sequence, structure = torch.randn(1, 20, 30), torch.randn(1, 20, 8)
    mask = torch.ones(1, 20, dtype=torch.bool)

    # The distance matrix must VARY across pairs. A constant bias added to
    # every attention logit cancels in the softmax, so a uniform distance
    # matrix correctly produces no change — testing with one would assert
    # that softmax is not shift-invariant, which it is.
    near = torch.cdist(
        torch.arange(20, dtype=torch.float32).reshape(1, 20, 1),
        torch.arange(20, dtype=torch.float32).reshape(1, 20, 1),
    )
    far = near * 5.0
    with torch.no_grad():
        close_out = model(sequence, structure, near, mask)
        far_out = model(sequence, structure, far, mask)
    assert not torch.allclose(close_out, far_out)


def test_parameter_count_grows_with_depth():
    from bindsite.models.transformer import build_model

    shallow = build_model(TransformerConfig(d_model=64, n_heads=4, n_layers=1))
    deep = build_model(TransformerConfig(d_model=64, n_heads=4, n_layers=3))
    assert deep.n_parameters() > shallow.n_parameters()


def test_config_serialises():
    record = TransformerConfig(d_model=64, n_heads=4).to_dict()
    assert record["d_model"] == 64
    assert record["use_distance_bias"] is True
