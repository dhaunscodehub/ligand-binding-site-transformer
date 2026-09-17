"""Baselines the transformer has to beat.

Four, in increasing order of what they would prove:

``prevalence``
    Predicts the training positive rate for every residue. Its AUROC is 0.5 by
    construction, and its AUPRC equals the positive rate. It exists to make
    the AUPRC scale legible: on a task with 12% positives, an AUPRC of 0.30 is
    2.5x over chance, whereas an AUROC of 0.82 sounds impressive without a
    reference point.

``burial``
    Thresholds a single geometric feature — neighbour count. Binding pockets
    are concave, so this is not nothing, and any learned model that fails to
    beat one hand-picked feature has not justified itself.

``logistic``
    Logistic regression on the per-residue feature vector, no context at all.
    The reference: it isolates how much of the signal is local chemistry
    versus structural context.

``random_forest``
    Non-linear on the same per-residue features. Separates "needs
    non-linearity" from "needs context" — if the forest matches the
    transformer, the gain was non-linearity, not attention.

None of these can see other residues, which is the specific capability the
transformer adds. Comparing against them is how that capability gets measured
instead of asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..exceptions import DataError

BASELINE_MODELS = ("prevalence", "burial", "logistic", "random_forest")


@dataclass
class FittedBaseline:
    """A trained baseline and what it needs to score residues."""

    name: str
    estimator: object
    n_train_residues: int
    positive_rate: float
    feature_names: list[str]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probability of binding, one value per residue."""
        if self.name == "prevalence":
            return np.full(features.shape[0], self.positive_rate, dtype=float)
        if self.name == "burial":
            # The estimator is the column index of the burial feature; the
            # score is that feature, rank-equivalent to a threshold sweep.
            column = int(self.estimator)  # type: ignore[arg-type]
            values = features[:, column]
            spread = values.max() - values.min()
            return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)
        return np.asarray(self.estimator.predict_proba(features))[:, 1]

    def to_dict(self) -> dict:
        return {
            "model": self.name, "n_train_residues": self.n_train_residues,
            "train_positive_rate": self.positive_rate,
            "n_features": len(self.feature_names),
        }


def fit_baseline(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    seed: int = 0,
    balanced: bool = True,
) -> FittedBaseline:
    """Fit one baseline on stacked per-residue features.

    ``balanced`` applies inverse-frequency class weights, matching the
    class-weighted loss the transformer uses. Without it the comparison would
    confound architecture with imbalance handling.
    """
    if name not in BASELINE_MODELS:
        raise DataError(
            f"unknown baseline {name!r}; choose from {list(BASELINE_MODELS)}"
        )
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels).astype(int)
    if features.shape[0] != labels.size:
        raise DataError(
            f"{features.shape[0]} feature rows but {labels.size} labels"
        )
    if features.shape[1] != len(feature_names):
        raise DataError(
            f"{features.shape[1]} feature columns but {len(feature_names)} names"
        )
    if labels.size == 0:
        raise DataError("no residues to fit on")
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        raise DataError(
            f"labels are all one class ({positives}/{labels.size} positive); "
            "a classifier cannot be fitted"
        )

    rate = float(labels.mean())
    names = list(feature_names)

    if name == "prevalence":
        estimator: object = None
    elif name == "burial":
        if "neighbour_count" not in names:
            raise DataError(
                "the burial baseline needs a 'neighbour_count' feature; "
                f"available: {names}"
            )
        estimator = names.index("neighbour_count")
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        weight = "balanced" if balanced else None
        if name == "logistic":
            estimator = Pipeline([
                # A feature with no variance cannot inform a decision, and
                # the unknown-residue one-hot column is constant on any
                # dataset of standard residues. Dropping it keeps the fitted
                # coefficients interpretable.
                #
                # Note: on numpy 2.2.x (Apple Silicon) sklearn's logistic
                # solver emits "divide by zero / overflow / invalid value
                # encountered in matmul" during the fit. Those warnings are
                # spurious — a matmul performs no division — and they are NOT
                # caused by constant columns or by array layout: they persist
                # after this step and after ascontiguousarray, and are absent
                # on numpy 2.5.x. Results are unaffected and were checked
                # rather than assumed: identical AUPRC (0.9830 on the
                # synthetic fixture) and finite coefficients with a maximum
                # magnitude of 1.061 on real data, under both versions. The
                # environment specs therefore ask for numpy >= 2.3.
                ("drop_constant", VarianceThreshold(threshold=0.0)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, class_weight=weight, random_state=seed
                )),
            ])
        else:
            # n_jobs=-1 parallelises the per-tree probability averaging, whose
            # reduction order is not fixed. Predictions are therefore
            # reproducible to about 4e-16 (one ULP, measured) rather than
            # bit-identical. That is far below any reported metric's
            # precision; set n_jobs=1 if bit-exactness is required.
            estimator = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                class_weight=weight, random_state=seed, n_jobs=-1,
            )
        estimator.fit(features, labels)

    return FittedBaseline(
        name=name, estimator=estimator, n_train_residues=int(labels.size),
        positive_rate=rate, feature_names=names,
    )
