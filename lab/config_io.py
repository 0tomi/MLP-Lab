"""JSON export/import of the lab configuration, plus a diagnostic export of a trained run.

The same schema is embedded in every saved run's meta.json under "config".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from lab.config import (
    ACTIVATIONS,
    BATCH_SIZES,
    L2_VALUES,
    LEARNING_RATES,
    MAX_HIDDEN,
    MONITOR_OPTIONS,
    OPTIMIZERS,
    LayerSpec,
    TrainConfig,
    output_spec,
)
from lab.datasets import CLASSIFICATION_DATASETS, REGRESSION_DATASETS, Split

SCHEMA_VERSION = 1
DATASET_NAMES = {
    "classification": [*CLASSIFICATION_DATASETS, "csv"],
    "regression": [*REGRESSION_DATASETS, "csv"],
}

_TOP_REQUIRED = ("schema_version", "task", "dataset", "layers", "optimizer", "learning_rate", "batch_size", "epochs",
                 "l2", "monitor", "patience")
_TOP_OPTIONAL = ("momentum", "early_stopping", "output_activation", "loss")
_DATASET_REQUIRED = ("name", "seed", "split", "standardize")
_DATASET_OPTIONAL = ("n_samples", "noise", "csv_file", "csv_target")
_SPLIT_KEYS = ("train", "validation", "test")
_LAYER_KEYS = ("units", "activation", "dropout")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_samples: int | None
    noise: float | None
    seed: int
    val: float
    test: float
    standardize: bool
    csv_file: str | None = None
    csv_target: str | None = None


@dataclass
class ImportedConfig:
    config: TrainConfig
    dataset: DatasetSpec
    adjusted: list[str] = field(default_factory=list)  # fields snapped to the nearest UI option


class ConfigError(ValueError):
    """Invalid config JSON; `code` and `params` select the message in lab.text.CONFIG_ERRORS."""

    def __init__(self, code: str, **params):
        super().__init__(f"{code}: {params}")
        self.code, self.params = code, params


# ---------------------------------------------------------------- export
def config_to_dict(config: TrainConfig, dataset: DatasetSpec) -> dict:
    out_activation, loss = output_spec(config.task)
    data = {
        "name": dataset.name,
        "n_samples": dataset.n_samples,
        "noise": dataset.noise,
        "seed": dataset.seed,
        "split": {
            "train": round(1 - dataset.val - dataset.test, 4),
            "validation": dataset.val,
            "test": dataset.test,
        },
        "standardize": dataset.standardize,
    }
    if dataset.name == "csv":
        data |= {"csv_file": dataset.csv_file, "csv_target": dataset.csv_target}
    return {
        "schema_version": SCHEMA_VERSION,
        "task": config.task,
        "dataset": data,
        "layers": [{"units": l.units, "activation": l.activation, "dropout": l.dropout} for l in config.layers],
        "output_activation": out_activation,
        "loss": loss,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "l2": config.l2,
        "monitor": config.monitor,
        "early_stopping": config.early_stopping,
        "patience": config.patience,
    }


def to_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- import
def parse_config(raw: str | bytes) -> ImportedConfig:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ConfigError("invalid_json", detail=str(err)) from err
    return validate_config(data)


def validate_config(data) -> ImportedConfig:
    adjusted: list[str] = []
    _object(data, "")
    _keys(data, _TOP_REQUIRED, _TOP_OPTIONAL, "")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("version", value=data["schema_version"], expected=SCHEMA_VERSION)
    task = _choice(data, "task", ("classification", "regression"))

    ds = data["dataset"]
    _object(ds, "dataset")
    _keys(ds, _DATASET_REQUIRED, _DATASET_OPTIONAL, "dataset.")
    name = _choice(ds, "name", DATASET_NAMES[task], "dataset.")
    split = ds["split"]
    _object(split, "dataset.split")
    _keys(split, ("validation", "test"), ("train",), "dataset.split.")
    seed = _int(ds, "seed", 0, 9999, "dataset.")
    n_samples = None if ds.get("n_samples") is None else _int(ds, "n_samples", 50, 1000, "dataset.")
    if n_samples is not None:
        n_samples = _snap_step(n_samples, 50, 50, "dataset.n_samples", adjusted, as_int=True)
    noise = None if ds.get("noise") is None else _number(ds, "noise", 0.0, 1.0, "dataset.")
    if noise is not None:
        noise = _snap_step(noise, 0.0, 0.05, "dataset.noise", adjusted)
    val = _snap_step(_number(split, "validation", 0.05, 0.30, "dataset.split."), 0.0, 0.05,
                     "dataset.split.validation", adjusted)
    test = _snap_step(_number(split, "test", 0.05, 0.30, "dataset.split."), 0.0, 0.05, "dataset.split.test", adjusted)
    standardize = _bool(ds, "standardize", "dataset.")
    dataset = DatasetSpec(name, n_samples, noise, seed, val, test, standardize,
                          ds.get("csv_file") if name == "csv" else None, ds.get("csv_target") if name == "csv" else None)

    layers_raw = data["layers"]
    if not isinstance(layers_raw, list):
        raise ConfigError("type", key="layers", expected="list")
    if len(layers_raw) > MAX_HIDDEN:
        raise ConfigError("too_many_layers", max=MAX_HIDDEN)
    layers = []
    for i, layer in enumerate(layers_raw, start=1):
        prefix = f"layers[{i}]."
        _object(layer, prefix[:-1])
        _keys(layer, _LAYER_KEYS, (), prefix)
        units = _int(layer, "units", 1, 64, prefix)
        activation = _choice(layer, "activation", ACTIVATIONS, prefix)
        dropout = _snap_step(_number(layer, "dropout", 0.0, 0.8, prefix), 0.0, 0.05, f"{prefix}dropout", adjusted)
        layers.append(LayerSpec(units, activation, dropout))

    optimizer = _choice(data, "optimizer", OPTIMIZERS)
    momentum = _number(data, "momentum", 0.0, 0.99) if data.get("momentum") is not None else 0.0
    config = TrainConfig(
        task=task,
        layers=tuple(layers),
        optimizer=optimizer,
        learning_rate=_snap_options(_number(data, "learning_rate", 1e-6, 10.0), LEARNING_RATES, "learning_rate",
                                    adjusted, log=True),
        momentum=round(momentum, 2) if optimizer == "sgd" else 0.0,
        batch_size=int(_snap_options(_int(data, "batch_size", 1, 100_000), BATCH_SIZES, "batch_size", adjusted,
                                     log=True)),
        epochs=_int(data, "epochs", 10, 3000),
        l2=_snap_options(_number(data, "l2", 0.0, 1.0), L2_VALUES, "l2", adjusted, log=False),
        monitor=_choice(data, "monitor", tuple(MONITOR_OPTIONS[task])),
        patience=_int(data, "patience", 5, 1000),
        early_stopping=_bool(data, "early_stopping") if "early_stopping" in data else True,
        seed=seed,
    )
    return ImportedConfig(config, dataset, adjusted)


def _object(value, key: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError("not_object" if not key else "type", key=key, expected="object")


def _keys(data: dict, required, optional, prefix: str) -> None:
    for key in required:
        if key not in data:
            raise ConfigError("missing", key=prefix + key)
    allowed = set(required) | set(optional)
    for key in data:
        if key not in allowed:
            raise ConfigError("unknown", key=prefix + key)


def _choice(data: dict, key: str, options, prefix: str = ""):
    if data[key] not in options:
        raise ConfigError("choice", key=prefix + key, value=data[key], options=", ".join(map(str, options)))
    return data[key]


def _number(data: dict, key: str, lo: float, hi: float, prefix: str = "") -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("type", key=prefix + key, expected="number")
    if not lo <= value <= hi:
        raise ConfigError("range", key=prefix + key, min=lo, max=hi)
    return float(value)


def _int(data: dict, key: str, lo: int, hi: int, prefix: str = "") -> int:
    value = data[key]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("type", key=prefix + key, expected="integer")
    if not lo <= value <= hi:
        raise ConfigError("range", key=prefix + key, min=lo, max=hi)
    return value


def _bool(data: dict, key: str, prefix: str = "") -> bool:
    if not isinstance(data[key], bool):
        raise ConfigError("type", key=prefix + key, expected="boolean")
    return data[key]


def _snap_step(value: float, start: float, step: float, key: str, adjusted: list, as_int: bool = False):
    snapped = round(start + round((value - start) / step) * step, 4)
    snapped = int(snapped) if as_int else snapped
    if abs(snapped - value) > 1e-9:
        adjusted.append(key)
    return snapped


def _snap_options(value: float, options, key: str, adjusted: list, log: bool):
    if log and value > 0:
        positive = [o for o in options if o > 0]
        snapped = min(positive, key=lambda o: abs(np.log(o) - np.log(value)))
    else:
        snapped = min(options, key=lambda o: abs(o - value))
    if abs(snapped - value) > 1e-12 * max(1.0, abs(value)):
        adjusted.append(key)
    return snapped


# ---------------------------------------------------------------- diagnostic
def _round(value, digits: int = 4):
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def build_diagnostic(meta: dict, split: Split) -> dict:
    """Config + training outcome + curves: enough to reason about under/overfitting."""
    data = {
        "n_features": int(split.X_train_raw.shape[1]),
        "train_shape": list(split.X_train_raw.shape),
        "validation_shape": list(split.X_val_raw.shape),
        "test_shape": list(split.X_test_raw.shape),
    }
    if split.task == "classification":
        data["positive_rate"] = _round({
            "train": float(np.mean(split.y_train)),
            "validation": float(np.mean(split.y_val)),
            "test": float(np.mean(split.y_test)),
        })
    config = meta["config"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "diagnostic",
        "model_id": meta.get("id"),
        "created": meta.get("created"),
        "config": config,
        "data": data,
        "separability": _round(meta.get("separability") or {}),
        "training": {
            "early_stopping": config.get("early_stopping", True),
            "best_epoch": meta["best_epoch"],
            "epochs_run": meta["epochs_run"],
            "stopped_early": meta["stopped_early"],
            "diverged": meta["diverged"],
        },
        "metrics": _round(meta["metrics"]),
        "history": _round(meta["history"]),
    }
