"""Predictions in original units for raw (unscaled) inputs."""

from __future__ import annotations

import numpy as np

from lab.datasets import Split


def predict(model, split: Split, X_raw: np.ndarray) -> np.ndarray:
    """P(class 1) for classification, target value (original units) for regression."""
    # Eager call (no tf.function): avoids retracing for every new input shape.
    out = np.asarray(model(split.transform_x(np.atleast_2d(X_raw)), training=False)).ravel()
    return out.astype(float) if split.task == "classification" else split.inverse_y(out)
