"""Training loop for the binding-site transformer.

Class imbalance is handled by **weighting the positive class in the loss**,
using ``pos_weight`` in ``BCEWithLogitsLoss``. The weight is computed from the
**training fold only** — computing it over the whole dataset would use the
test fold's label distribution, which is a small but real leak and an easy one
to commit by accident.

Two details that matter more than they look:

**Padding must be masked out of the loss, not just the attention.** Proteins
have different lengths, so batches are padded. A padded position has a label
of 0, and 0 is the majority class, so including padding in the loss trains the
model to predict "not binding" on positions that do not exist. With variable
lengths that silently reweights the loss by batch composition.

**Early stopping is on validation AUPRC, not loss.** Under class imbalance the
loss is dominated by the negative class, so it can improve while the ranking
of positives gets worse. AUPRC tracks the quantity the model is for.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .exceptions import ConfigError, DataError, OptionalDependencyMissing
from .features import ResidueFeatures
from .models.transformer import TransformerConfig, build_model, select_device


@dataclass
class TrainConfig:
    """Optimisation settings."""

    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.1
    grad_clip: float = 1.0
    patience: int = 8
    class_weighting: bool = True
    # Cap on pos_weight. An extreme weight makes the model predict almost
    # everything positive, which raises recall and destroys precision; the cap
    # keeps the loss from being dominated by a handful of positives.
    max_pos_weight: float = 20.0
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        if not 0 < self.learning_rate < 1:
            raise ConfigError(
                f"learning_rate must be in (0, 1), got {self.learning_rate}"
            )
        if not 0 <= self.warmup_fraction < 1:
            raise ConfigError(
                f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}"
            )
        if self.max_pos_weight < 1:
            raise ConfigError(
                f"max_pos_weight must be at least 1, got {self.max_pos_weight}"
            )

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class TrainingHistory:
    """What happened during training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_auprc: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_auprc: float = -1.0
    stopped_early: bool = False
    pos_weight: float = 1.0
    n_parameters: int = 0
    device: str = "cpu"
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epochs_run": len(self.train_loss),
            "train_loss": self.train_loss, "val_loss": self.val_loss,
            "val_auprc": self.val_auprc, "val_auroc": self.val_auroc,
            "best_epoch": self.best_epoch,
            "best_val_auprc": self.best_val_auprc,
            "stopped_early": self.stopped_early,
            "pos_weight": self.pos_weight,
            "n_parameters": self.n_parameters,
            "device": self.device, "seconds": self.seconds,
            "early_stopping_metric": "validation AUPRC",
        }


def compute_pos_weight(
    labels: Sequence[np.ndarray], cap: float = 20.0
) -> float:
    """Inverse-frequency weight for the positive class, from training labels.

    Returns ``negatives / positives``, capped. The cap is not cosmetic: on a
    protein with 3 binding residues out of 300 the raw weight is 99, and a
    model trained at that weight predicts nearly everything positive.
    """
    stacked = np.concatenate([np.asarray(l).astype(int).ravel() for l in labels])
    if stacked.size == 0:
        raise DataError("no labels supplied for pos_weight")
    positives = int(stacked.sum())
    if positives == 0:
        raise DataError(
            "training labels contain no binding residues; check the contact "
            "cutoff and the ligand curation"
        )
    return float(min((stacked.size - positives) / positives, cap))


def _collate(batch: Sequence[ResidueFeatures], device, torch):
    """Pad a batch and build the mask, the distance matrix and the labels."""
    lengths = [f.n_residues for f in batch]
    longest = max(lengths)
    size = len(batch)

    sequence_dim = batch[0].sequence_block.shape[1]
    structure_dim = batch[0].structure_block.shape[1]

    sequence = torch.zeros(size, longest, sequence_dim)
    structure = torch.zeros(size, longest, structure_dim)
    labels = torch.zeros(size, longest)
    mask = torch.zeros(size, longest, dtype=torch.bool)
    distances = torch.zeros(size, longest, longest)

    for index, features in enumerate(batch):
        length = features.n_residues
        sequence[index, :length] = torch.from_numpy(
            features.sequence_block.astype("float32")
        )
        structure[index, :length] = torch.from_numpy(
            features.structure_block.astype("float32")
        )
        labels[index, :length] = torch.from_numpy(
            features.labels.astype("float32")
        )
        mask[index, :length] = True
        distances[index, :length, :length] = torch.from_numpy(
            _distance_matrix(features).astype("float32")
        )

    return (
        sequence.to(device), structure.to(device), distances.to(device),
        labels.to(device), mask.to(device),
    )


def _distance_matrix(features: ResidueFeatures) -> np.ndarray:
    """The feature record's CA-CA distance matrix.

    Absent only for hand-built records in tests, where a zero matrix makes the
    distance bias contribute nothing and the model still runs.
    """
    if features.distance_matrix is None:
        return np.zeros((features.n_residues, features.n_residues))
    return features.distance_matrix


def train_model(
    train_features: Sequence[ResidueFeatures],
    val_features: Sequence[ResidueFeatures],
    model_config: TransformerConfig,
    train_config: TrainConfig | None = None,
    verbose: bool = False,
):
    """Train the transformer, returning (model, history).

    The returned model has the weights from the best validation epoch, not the
    last: the last epoch is usually overfitted, and restoring the best
    checkpoint is the difference between reporting the model you would deploy
    and the one that happened to finish.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:
        raise OptionalDependencyMissing(
            "PyTorch is required to train: pip install torch"
        ) from error

    from sklearn.metrics import average_precision_score, roc_auc_score

    train_config = train_config or TrainConfig()
    if not train_features:
        raise DataError("no training proteins")
    if not val_features:
        raise DataError(
            "no validation proteins; early stopping needs a validation fold, "
            "and selecting on the training fold would report a fitted value"
        )

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    device = torch.device(select_device(train_config.device))

    model = build_model(model_config).to(device)
    pos_weight = (
        compute_pos_weight(
            [f.labels for f in train_features], train_config.max_pos_weight
        )
        if train_config.class_weighting else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(pos_weight, device=device)
    )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    batches_per_epoch = max(
        1, (len(train_features) + train_config.batch_size - 1) // train_config.batch_size
    )
    total_steps = batches_per_epoch * train_config.epochs
    warmup_steps = max(1, int(total_steps * train_config.warmup_fraction))

    def learning_rate_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return float(0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, learning_rate_at)

    history = TrainingHistory(
        pos_weight=pos_weight, n_parameters=model.n_parameters(),
        device=str(device),
    )
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    since_improvement = 0

    import time

    started = time.time()
    order = np.arange(len(train_features))

    for epoch in range(train_config.epochs):
        model.train()
        np.random.shuffle(order)
        epoch_loss, epoch_residues = 0.0, 0

        for start in range(0, len(order), train_config.batch_size):
            batch = [train_features[i] for i in order[start:start + train_config.batch_size]]
            sequence, structure, distances, labels, mask = _collate(batch, device, torch)

            logits = model(
                sequence if model_config.use_sequence else None,
                structure if model_config.use_structure else None,
                distances, mask,
            )
            per_residue = criterion(logits, labels)
            # Mask padding out of the loss. Padded positions carry label 0,
            # the majority class, so including them would train the model on
            # positions that do not exist and reweight the loss by batch
            # composition.
            valid = mask.float()
            loss = (per_residue * valid).sum() / valid.sum().clamp(min=1.0)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimiser.step()
            scheduler.step()

            n_valid = int(valid.sum().item())
            epoch_loss += float(loss.item()) * n_valid
            epoch_residues += n_valid

        history.train_loss.append(epoch_loss / max(epoch_residues, 1))

        scores, targets, val_loss, val_residues = [], [], 0.0, 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(val_features), train_config.batch_size):
                batch = val_features[start:start + train_config.batch_size]
                sequence, structure, distances, labels, mask = _collate(
                    batch, device, torch
                )
                logits = model(
                    sequence if model_config.use_sequence else None,
                    structure if model_config.use_structure else None,
                    distances, mask,
                )
                per_residue = criterion(logits, labels)
                valid = mask.float()
                val_loss += float((per_residue * valid).sum().item())
                val_residues += int(valid.sum().item())
                probabilities = torch.sigmoid(logits)
                scores.append(probabilities[mask].detach().cpu().numpy())
                targets.append(labels[mask].detach().cpu().numpy())

        pooled_scores = np.concatenate(scores)
        pooled_targets = np.concatenate(targets).astype(int)
        history.val_loss.append(val_loss / max(val_residues, 1))
        auprc = (
            float(average_precision_score(pooled_targets, pooled_scores))
            if pooled_targets.sum() > 0 else 0.0
        )
        auroc = (
            float(roc_auc_score(pooled_targets, pooled_scores))
            if 0 < pooled_targets.sum() < pooled_targets.size else 0.5
        )
        history.val_auprc.append(auprc)
        history.val_auroc.append(auroc)

        if auprc > history.best_val_auprc:
            history.best_val_auprc = auprc
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improvement = 0
        else:
            since_improvement += 1

        if verbose:
            # stderr: stdout is reserved for the JSON result.
            print(
                f"  epoch {epoch + 1:>3}/{train_config.epochs}  "
                f"train {history.train_loss[-1]:.4f}  val {history.val_loss[-1]:.4f}  "
                f"AUPRC {auprc:.4f}  AUROC {auroc:.4f}"
                + ("  *" if since_improvement == 0 else ""),
                file=sys.stderr, flush=True,
            )

        if since_improvement >= train_config.patience:
            history.stopped_early = True
            if verbose:
                print(
                    f"  early stop: no validation AUPRC improvement in "
                    f"{train_config.patience} epochs",
                    file=sys.stderr, flush=True,
                )
            break

    # Restore the best epoch, not the last.
    model.load_state_dict(best_state)
    history.seconds = time.time() - started
    return model, history


def predict(
    model,
    features: Sequence[ResidueFeatures],
    model_config: TransformerConfig,
    batch_size: int = 8,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Per-residue binding probabilities, keyed by protein identifier."""
    import torch

    target = torch.device(device or select_device("auto"))
    model = model.to(target).eval()
    out: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = list(features[start:start + batch_size])
            sequence, structure, distances, _, mask = _collate(batch, target, torch)
            logits = model(
                sequence if model_config.use_sequence else None,
                structure if model_config.use_structure else None,
                distances, mask,
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for index, item in enumerate(batch):
                out[item.identifier] = probabilities[index, : item.n_residues]
    return out


def save_checkpoint(
    model, model_config: TransformerConfig, history: TrainingHistory,
    path: str | Path,
) -> Path:
    """Save weights together with the config that defines their shape."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config.to_dict(),
            "history": history.to_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str | None = None):
    """Load a checkpoint, rebuilding the model from its stored config."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise DataError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TransformerConfig(**payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["state_dict"])
    target = torch.device(device or select_device("auto"))
    return model.to(target).eval(), config, payload.get("history", {})
