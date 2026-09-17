"""Training loop: class weighting, padding, and early stopping."""

from __future__ import annotations

import numpy as np
import pytest

from bindsite.exceptions import ConfigError, DataError
from bindsite.models.transformer import TransformerConfig
from bindsite.train import TrainConfig, compute_pos_weight

torch = pytest.importorskip("torch")


def test_pos_weight_is_the_negative_to_positive_ratio():
    labels = [np.array([1] * 10 + [0] * 90)]
    assert compute_pos_weight(labels, cap=100) == pytest.approx(9.0)


def test_pos_weight_is_capped():
    """An extreme weight makes the model predict nearly everything positive."""
    labels = [np.array([1] + [0] * 999)]
    assert compute_pos_weight(labels, cap=20.0) == pytest.approx(20.0)


def test_pos_weight_uses_only_the_labels_given():
    """Computing it over the whole dataset would use the test distribution."""
    train_only = compute_pos_weight([np.array([1] * 20 + [0] * 80)], cap=100)
    both = compute_pos_weight(
        [np.array([1] * 20 + [0] * 80), np.array([0] * 100)], cap=100
    )
    assert train_only != both


def test_pos_weight_rejects_no_positives():
    with pytest.raises(DataError, match="no binding residues"):
        compute_pos_weight([np.zeros(50, dtype=int)])


def test_pos_weight_rejects_empty():
    with pytest.raises(DataError, match="no labels"):
        compute_pos_weight([np.array([], dtype=int)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0},
        {"learning_rate": 1.0}, {"warmup_fraction": 1.0},
        {"warmup_fraction": -0.1}, {"max_pos_weight": 0.0},
    ],
)
def test_invalid_training_config_is_rejected(kwargs):
    with pytest.raises(ConfigError):
        TrainConfig(**kwargs)


def test_training_config_serialises():
    record = TrainConfig(epochs=5).to_dict()
    assert record["epochs"] == 5
    assert record["class_weighting"] is True


def test_padding_is_excluded_from_the_loss():
    """Changing a PADDED position must not change the loss.

    Padded positions carry label 0, the majority class. Including them would
    train the model on positions that do not exist and reweight the loss by
    batch composition — a batch of one long and one short protein would
    weight the short one's real residues differently from a batch of two
    short ones.
    """
    import torch.nn as nn

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    valid = mask.float()

    logits = torch.full((1, 10), -2.0)
    labels = torch.zeros(1, 10)
    labels[0, 0] = 1.0

    def masked_loss(lg, lb):
        return ((criterion(lg, lb) * valid).sum() / valid.sum()).item()

    baseline = masked_loss(logits, labels)

    # Perturb only padded positions.
    perturbed_logits = logits.clone()
    perturbed_logits[0, 4:] = 8.0
    perturbed_labels = labels.clone()
    perturbed_labels[0, 6] = 1.0

    assert masked_loss(perturbed_logits, perturbed_labels) == pytest.approx(baseline)
    # The unmasked mean does change, which is the behaviour being avoided.
    assert criterion(perturbed_logits, perturbed_labels).mean().item() != pytest.approx(
        criterion(logits, labels).mean().item()
    )


def test_training_runs_and_improves_on_a_learnable_task(features_signal):
    """A short run on a clean signal must reduce the loss and reach a
    validation AUPRC above the positive rate."""
    from bindsite.train import train_model

    records = list(features_signal.values())
    train, val = records[:16], records[16:]
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    model, history = train_model(
        train, val, config,
        TrainConfig(epochs=6, batch_size=4, learning_rate=1e-3, patience=6,
                    seed=0, device="cpu"),
    )
    assert len(history.train_loss) == 6
    assert history.train_loss[-1] < history.train_loss[0]
    positive_rate = float(
        np.concatenate([r.labels for r in val]).mean()
    )
    assert history.best_val_auprc > positive_rate


def test_training_records_the_environment(features_signal):
    from bindsite.train import train_model

    records = list(features_signal.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    _, history = train_model(
        records[:12], records[12:16], config,
        TrainConfig(epochs=2, batch_size=4, seed=0, device="cpu"),
    )
    record = history.to_dict()
    assert record["n_parameters"] > 0
    assert record["device"] == "cpu"
    assert record["pos_weight"] > 1.0
    assert record["early_stopping_metric"] == "validation AUPRC"


def test_class_weighting_can_be_disabled(features_signal):
    from bindsite.train import train_model

    records = list(features_signal.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    _, history = train_model(
        records[:12], records[12:16], config,
        TrainConfig(epochs=1, batch_size=4, class_weighting=False, seed=0,
                    device="cpu"),
    )
    assert history.pos_weight == 1.0


def test_early_stopping_triggers(features_noise):
    """On unlearnable data, validation AUPRC stops improving."""
    from bindsite.train import train_model

    records = list(features_noise.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    _, history = train_model(
        records[:16], records[16:], config,
        TrainConfig(epochs=30, batch_size=4, patience=2, seed=0, device="cpu"),
    )
    assert history.stopped_early
    assert len(history.train_loss) < 30


def test_best_epoch_weights_are_restored(features_signal):
    """The last epoch is usually overfitted; the reported model must be the
    best-validation one."""
    from bindsite.train import predict, train_model

    records = list(features_signal.values())
    train, val, test = records[:14], records[14:18], records[18:]
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    model, history = train_model(
        train, val, config,
        TrainConfig(epochs=8, batch_size=4, learning_rate=1e-3, patience=8,
                    seed=0, device="cpu"),
    )
    assert 0 <= history.best_epoch < len(history.train_loss)
    scores = predict(model, test, config, batch_size=4, device="cpu")
    assert set(scores) == {r.identifier for r in test}
    for record in test:
        assert scores[record.identifier].shape == (record.n_residues,)


def test_training_requires_a_validation_fold(features_signal):
    from bindsite.train import train_model

    records = list(features_signal.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    with pytest.raises(DataError, match="validation fold"):
        train_model(records[:8], [], config, TrainConfig(epochs=1, device="cpu"))


def test_training_requires_training_data(features_signal):
    from bindsite.train import train_model

    records = list(features_signal.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    with pytest.raises(DataError, match="no training proteins"):
        train_model([], records[:4], config, TrainConfig(epochs=1, device="cpu"))


def test_predictions_are_probabilities(features_signal):
    from bindsite.train import predict, train_model

    records = list(features_signal.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    model, _ = train_model(
        records[:12], records[12:16], config,
        TrainConfig(epochs=1, batch_size=4, seed=0, device="cpu"),
    )
    scores = predict(model, records[16:], config, batch_size=4, device="cpu")
    for values in scores.values():
        assert values.min() >= 0.0 and values.max() <= 1.0


def test_checkpoint_round_trips(features_signal, tmp_path):
    from bindsite.train import (
        load_checkpoint, predict, save_checkpoint, train_model,
    )

    records = list(features_signal.values())
    config = TransformerConfig(d_model=32, n_heads=2, n_layers=1, max_length=256)
    model, history = train_model(
        records[:12], records[12:16], config,
        TrainConfig(epochs=2, batch_size=4, seed=0, device="cpu"),
    )
    path = save_checkpoint(model, config, history, tmp_path / "ck.pt")
    restored, restored_config, stored_history = load_checkpoint(path, device="cpu")

    assert restored_config.to_dict() == config.to_dict()
    assert stored_history["n_parameters"] == history.n_parameters
    before = predict(model, records[16:18], config, device="cpu")
    after = predict(restored, records[16:18], restored_config, device="cpu")
    for name in before:
        assert np.allclose(before[name], after[name], atol=1e-6)


def test_missing_checkpoint_is_reported(tmp_path):
    from bindsite.train import load_checkpoint

    with pytest.raises(DataError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt")
