"""Datasets and train/val/test splitting. Built on sklearn generators and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lab import text as T

Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class Dataset:
    key: str
    task: Task
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    class_names: tuple[str, str] | None = None


def _corners(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    idx = np.arange(n) % 4
    X = corners[idx] + rng.normal(0, 0.05 + 0.15 * noise, size=(n, 2))
    return X, corners[idx]


def _blobs(n, noise, seed):
    X, y = skd.make_blobs(n_samples=n, centers=[[-2, -2], [2, 2]], cluster_std=0.5 + 1.5 * noise, random_state=seed)
    return X, y


def _xor(n, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X + rng.normal(0, 0.5 * noise, size=X.shape), y


def _circles(n, noise, seed):
    return skd.make_circles(n_samples=n, noise=0.3 * noise, factor=0.5, random_state=seed)


def _moons(n, noise, seed):
    return skd.make_moons(n_samples=n, noise=noise, random_state=seed)


def _and(n, noise, seed):
    X, bits = _corners(n, noise, np.random.default_rng(seed))
    return X, (bits[:, 0] * bits[:, 1]).astype(int)


def _or(n, noise, seed):
    X, bits = _corners(n, noise, np.random.default_rng(seed))
    return X, np.maximum(bits[:, 0], bits[:, 1]).astype(int)


def _linear_1d(n, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 1))
    return x, 2.0 * x[:, 0] + 1.0 + rng.normal(0, 3 * noise, size=n)


def _sine_1d(n, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, size=(n, 1))
    return x, np.sin(2 * x[:, 0]) + rng.normal(0, noise, size=n)


_XY = ["x1", "x2"]
_BINARY = T.CLASS_NAMES["default"]

# key -> (generator, feature names, class names)
CLASSIFICATION_DATASETS: dict[str, tuple[Callable, list[str], tuple[str, str]]] = {
    "blobs": (_blobs, _XY, _BINARY),
    "xor": (_xor, _XY, _BINARY),
    "circles": (_circles, _XY, T.CLASS_NAMES["circles"]),
    "moons": (_moons, _XY, _BINARY),
    "and": (_and, _XY, _BINARY),
    "or": (_or, _XY, _BINARY),
    "breast_cancer": (None, [], T.CLASS_NAMES["breast_cancer"]),
}
REGRESSION_DATASETS: dict[str, tuple[Callable, list[str]]] = {
    "linear_1d": (_linear_1d, ["x"]),
    "sine_1d": (_sine_1d, ["x"]),
    "diabetes": (None, []),
}
REAL_DATASETS = {"breast_cancer", "diabetes"}


def make_dataset(key: str, n_samples: int, noise: float, seed: int) -> Dataset:
    """Build a dataset. Real datasets ignore n_samples, noise and seed."""
    if key == "breast_cancer":
        data = skd.load_breast_cancer()
        return Dataset(key, "classification", data.data, data.target.astype(int), list(data.feature_names),
                       T.TARGET_NAMES["breast_cancer"], T.CLASS_NAMES["breast_cancer"])
    if key == "diabetes":
        data = skd.load_diabetes()
        return Dataset(key, "regression", data.data, data.target.astype(float), list(data.feature_names), T.TARGET_NAMES["diabetes"])
    if key in CLASSIFICATION_DATASETS:
        gen, features, classes = CLASSIFICATION_DATASETS[key]
        X, y = gen(n_samples, noise, seed)
        return Dataset(key, "classification", np.asarray(X, float), np.asarray(y, int), list(features), T.TARGET_NAMES["classification"], classes)
    if key in REGRESSION_DATASETS:
        gen, features = REGRESSION_DATASETS[key]
        X, y = gen(n_samples, noise, seed)
        return Dataset(key, "regression", np.asarray(X, float), np.asarray(y, float), list(features), T.TARGET_NAMES["regression"])
    raise KeyError(f"Unknown dataset: {key}")


class CsvError(ValueError):
    """Invalid uploaded CSV; `code` and `params` select the message in lab.text.DATA_CSV_ERRORS."""

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code, self.params = code, params


def dataset_from_dataframe(df: pd.DataFrame, target: str, task: Task, name: str) -> Dataset:
    """Numeric feature columns only; rows with missing values are dropped."""
    features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise CsvError("no_numeric")
    clean = df[features + [target]].dropna()
    X = clean[features].to_numpy(dtype=float)
    if task == "classification":
        encoder = LabelEncoder().fit(clean[target])
        if len(encoder.classes_) != 2:
            raise CsvError("not_binary", n=len(encoder.classes_))
        y = encoder.transform(clean[target]).astype(int)
        return Dataset(name, task, X, y, features, target, tuple(str(c) for c in encoder.classes_))
    if not pd.api.types.is_numeric_dtype(clean[target]):
        raise CsvError("not_numeric_target")
    return Dataset(name, task, X, clean[target].to_numpy(dtype=float), features, target)


@dataclass
class Split:
    """Raw splits plus the scalers fitted on the training part only."""

    task: Task
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler | None
    y_scaler: StandardScaler | None

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (self.x_scaler.transform(X) if self.x_scaler is not None else X).astype("float32")

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        return (self.y_scaler.transform(y) if self.y_scaler is not None else y).astype("float32")

    def inverse_y(self, y_fit: np.ndarray) -> np.ndarray:
        y_fit = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        return (self.y_scaler.inverse_transform(y_fit) if self.y_scaler is not None else y_fit).ravel()

    @property
    def X_train(self):
        return self.transform_x(self.X_train_raw)

    @property
    def X_val(self):
        return self.transform_x(self.X_val_raw)

    @property
    def X_test(self):
        return self.transform_x(self.X_test_raw)

    @property
    def y_train_fit(self):
        return self.transform_y(self.y_train)

    @property
    def y_val_fit(self):
        return self.transform_y(self.y_val)

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Raw X and y for 'train', 'val' or 'test'."""
        return getattr(self, f"X_{name}_raw"), getattr(self, f"y_{name}")


def split_dataset(ds: Dataset, val_size: float, test_size: float, seed: int, standardize: bool) -> Split:
    n = len(ds.y)
    n_test, n_val = max(2, round(test_size * n)), max(2, round(val_size * n))
    stratify = ds.y if ds.task == "classification" else None
    X_rest, X_test, y_rest, y_test = train_test_split(ds.X, ds.y, test_size=n_test, random_state=seed, stratify=stratify)
    stratify = y_rest if ds.task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=n_val, random_state=seed, stratify=stratify)
    x_scaler = StandardScaler().fit(X_train) if standardize else None
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1)) if standardize and ds.task == "regression" else None
    return Split(ds.task, X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler)
