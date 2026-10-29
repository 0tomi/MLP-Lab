"""Linear separability check (classification) and linear baseline (regression)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SeparabilityResult:
    separable: bool
    accuracy: float
    model: Pipeline


def check_linear_separability(X: np.ndarray, y: np.ndarray) -> SeparabilityResult:
    """Fit an almost unregularized logistic regression on all the data.

    100% training accuracy proves that a separating hyperplane exists.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=20_000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    accuracy = float((model.predict(X) == y).mean())
    return SeparabilityResult(accuracy == 1.0, accuracy, model)


def linear_regression_baseline(X: np.ndarray, y: np.ndarray) -> float:
    """R² of ordinary least squares on all the data."""
    return float(LinearRegression().fit(X, y).score(X, y))


def linear_regression_line(x: np.ndarray, y: np.ndarray, steps: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line over the range of a single feature, for plotting."""
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    xs = np.linspace(x.min(), x.max(), steps)
    return xs, model.predict(xs.reshape(-1, 1))
