"""Final evaluation metrics, computed with sklearn on original-unit predictions."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(skm.accuracy_score(y, pred)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred, zero_division=0)),
        "f1": float(skm.f1_score(y, pred, zero_division=0)),
        "roc_auc": float(skm.roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan"),
        "log_loss": float(skm.log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(skm.mean_squared_error(y, pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(skm.mean_absolute_error(y, pred)),
        "r2": float(skm.r2_score(y, pred)),
    }


def confusion(y: np.ndarray, proba: np.ndarray, threshold: float) -> np.ndarray:
    return skm.confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1])


def roc(y: np.ndarray, proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return skm.roc_curve(y, proba)
