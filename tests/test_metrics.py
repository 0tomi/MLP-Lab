import numpy as np
import pytest

from lab.metrics import classification_metrics, regression_metrics


def test_classification_metrics_depend_on_threshold():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])

    at_half = classification_metrics(y, proba, threshold=0.5)
    at_low = classification_metrics(y, proba, threshold=0.3)

    assert at_half["accuracy"] == pytest.approx(0.5)
    assert at_low["recall"] == pytest.approx(1.0)
    assert at_half["roc_auc"] == at_low["roc_auc"] == pytest.approx(0.75)
    assert set(at_half) == {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"}


def test_regression_metrics():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 5.0])
    m = regression_metrics(y, pred)
    assert m["mse"] == pytest.approx(4 / 3)
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert set(m) == {"mse", "rmse", "mae", "r2"}
