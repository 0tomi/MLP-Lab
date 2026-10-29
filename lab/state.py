"""Session state, cached data building and the active saved model."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from lab import registry
from lab import text as T
from lab.config import MAX_HIDDEN, LayerSpec, TrainConfig
from lab.config_io import DatasetSpec, ImportedConfig
from lab.datasets import (
    REAL_DATASETS,
    CsvError,
    Dataset,
    Split,
    dataset_from_dataframe,
    make_dataset,
    split_dataset,
)
from lab.separability import SeparabilityResult, check_linear_separability

_UNITS = (8, 4, 4, 4)
_ACTS = ("tanh", "relu", "relu", "relu")

DEFAULTS: dict = {
    "task": "classification",
    "dataset_classification": "xor",
    "dataset_regression": "sine_1d",
    "n_samples": 300,
    "noise": 0.15,
    "seed": 0,
    "val_pct": 15,
    "test_pct": 15,
    "standardize": True,
    "n_hidden": 2,
    **{f"units_{i}": _UNITS[i - 1] for i in range(1, MAX_HIDDEN + 1)},
    **{f"act_{i}": _ACTS[i - 1] for i in range(1, MAX_HIDDEN + 1)},
    **{f"drop_{i}": 0.0 for i in range(1, MAX_HIDDEN + 1)},
    "optimizer": "adam",
    "lr": 0.01,
    "momentum": 0.9,
    "batch_size": 32,
    "epochs": 300,
    "l2": 0.0,
    "monitor_classification": "val_loss",
    "monitor_regression": "val_loss",
    "patience": 40,
    "early_stopping": True,
    "refresh": 10,
    "threshold": 0.5,
    "results_split": "test",
}


def init_state() -> None:
    """Seed defaults and keep widget values alive even when their widget is not drawn."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = st.session_state.get(key, value)
    if "pending_active" in st.session_state:
        st.session_state["active_run_id"] = st.session_state.pop("pending_active")
    if "pending_import" in st.session_state:
        apply_imported(st.session_state.pop("pending_import"))


def apply_imported(imported: ImportedConfig) -> None:
    """Write an imported config into the widget state. Call before any widget is drawn."""
    s, c, d = st.session_state, imported.config, imported.dataset
    s["task"] = c.task
    s[f"dataset_{c.task}"] = d.name
    if d.n_samples is not None:
        s["n_samples"] = d.n_samples
    if d.noise is not None:
        s["noise"] = d.noise
    s["seed"] = d.seed
    s["val_pct"], s["test_pct"] = round(d.val * 100), round(d.test * 100)
    s["standardize"] = d.standardize
    s["n_hidden"] = len(c.layers)
    for i, layer in enumerate(c.layers, start=1):
        s[f"units_{i}"], s[f"act_{i}"], s[f"drop_{i}"] = layer.units, layer.activation, layer.dropout
    s["optimizer"], s["lr"], s["batch_size"] = c.optimizer, c.learning_rate, c.batch_size
    if c.optimizer == "sgd":
        s["momentum"] = c.momentum
    s["epochs"], s["l2"], s["patience"] = c.epochs, c.l2, c.patience
    s[f"monitor_{c.task}"] = c.monitor
    s["early_stopping"] = c.early_stopping


def dataset_spec() -> DatasetSpec:
    """The Data widgets as the exportable dataset section."""
    s = st.session_state
    key = dataset_key()
    synthetic = key not in REAL_DATASETS and key != "csv"
    upload = s.get("csv_file") if key == "csv" else None
    return DatasetSpec(
        name=key,
        n_samples=s["n_samples"] if synthetic else None,
        noise=s["noise"] if synthetic else None,
        seed=int(s["seed"]),
        val=s["val_pct"] / 100,
        test=s["test_pct"] / 100,
        standardize=s["standardize"],
        csv_file=upload.name if upload is not None else None,
        csv_target=s.get("csv_target") if key == "csv" else None,
    )


# ---------------------------------------------------------------- data
@dataclass
class DataContext:
    dataset: Dataset
    split: Split
    data_id: tuple


@st.cache_data(show_spinner=False)
def _builtin(key: str, n: int, noise: float, seed: int) -> Dataset:
    return make_dataset(key, n, noise, seed)


@st.cache_data(show_spinner=False)
def _from_csv(data: bytes, name: str, target: str, task: str) -> Dataset:
    return dataset_from_dataframe(pd.read_csv(io.BytesIO(data)), target, task, name)


@st.cache_data(show_spinner=False)
def read_csv_columns(data: bytes) -> list[str]:
    return list(pd.read_csv(io.BytesIO(data), nrows=5).columns)


@st.cache_data(show_spinner=False)
def _split(_ds: Dataset, data_id: tuple, val: float, test: float, seed: int, standardize: bool) -> Split:
    return split_dataset(_ds, val, test, seed, standardize)


@st.cache_resource(show_spinner=False, max_entries=16)
def separability(_ds: Dataset, data_id: tuple) -> SeparabilityResult:
    return check_linear_separability(_ds.X, _ds.y)


def dataset_key() -> str:
    return st.session_state[f"dataset_{st.session_state['task']}"]


def current_data() -> DataContext | None:
    """Dataset and split described by the Data widgets; None while a CSV is pending."""
    s = st.session_state
    task, key = s["task"], dataset_key()
    if key == "csv":
        upload = s.get("csv_file")
        target = s.get("csv_target")
        if upload is None or target is None:
            return None
        data = upload.getvalue()
        digest = hashlib.sha1(data).hexdigest()[:12]
        try:
            ds = _from_csv(data, upload.name, target, task)
        except CsvError as err:
            st.session_state["csv_error"] = T.DATA_CSV_ERRORS[err.code].format(**err.params)
            return None
        st.session_state.pop("csv_error", None)
        data_id = ("csv", digest, target, task)
    else:
        ds = _builtin(key, s["n_samples"], s["noise"], s["seed"])
        data_id = (key, s["n_samples"], s["noise"], s["seed"])
    val, test = s["val_pct"] / 100, s["test_pct"] / 100
    split = _split(ds, data_id, val, test, s["seed"], s["standardize"])
    return DataContext(ds, split, data_id)


# ---------------------------------------------------------------- architecture / training
def current_layers() -> tuple[LayerSpec, ...]:
    s = st.session_state
    return tuple(
        LayerSpec(int(s[f"units_{i}"]), s[f"act_{i}"], float(s[f"drop_{i}"])) for i in range(1, s["n_hidden"] + 1)
    )


def current_config() -> TrainConfig:
    s = st.session_state
    task = s["task"]
    return TrainConfig(
        task=task,
        layers=current_layers(),
        optimizer=s["optimizer"],
        learning_rate=float(s["lr"]),
        momentum=float(s["momentum"]) if s["optimizer"] == "sgd" else 0.0,
        batch_size=int(s["batch_size"]),
        epochs=int(s["epochs"]),
        l2=float(s["l2"]),
        monitor=s[f"monitor_{task}"],
        patience=int(s["patience"]),
        early_stopping=bool(s["early_stopping"]),
        seed=int(s["seed"]),
    )


# ---------------------------------------------------------------- saved models
@st.cache_resource(show_spinner=False, max_entries=8)
def _load(path: str) -> registry.LoadedRun:
    return registry.load_run(Path(path))


def saved_runs() -> list[registry.RunInfo]:
    return registry.list_runs(models_dir())


def models_dir() -> Path:
    return Path(st.session_state.get("models_dir", registry.MODELS_DIR))


def active_run() -> registry.LoadedRun | None:
    runs = saved_runs()
    if not runs:
        return None
    by_id = {r.meta["id"]: r for r in runs}
    run = by_id.get(st.session_state.get("active_run_id")) or runs[0]
    return _load(str(run.path))
