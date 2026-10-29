"""Rotating on-disk history of trained models (newest MAX_RUNS kept)."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from lab.datasets import Split

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_RUNS = 5
_FILES = ("model.keras", "scaler.joblib", "meta.json", "data.npz")
_ARRAYS = ("X_train_raw", "X_val_raw", "X_test_raw", "y_train", "y_val", "y_test")


@dataclass(frozen=True)
class RunInfo:
    path: Path
    meta: dict


@dataclass
class LoadedRun:
    path: Path
    model: object  # keras.Model
    split: Split
    meta: dict


def save_run(model, split: Split, meta: dict, root: Path = MODELS_DIR, keep: int = MAX_RUNS) -> Path:
    """Save model, scalers, metadata and data splits, then prune older runs."""
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = root / f"{now:%Y%m%d-%H%M%S-%f}_{uuid.uuid4().hex[:6]}"
    tmp = path.with_name(path.name + ".tmp")
    tmp.mkdir()
    model.save(tmp / "model.keras")
    joblib.dump({"x": split.x_scaler, "y": split.y_scaler}, tmp / "scaler.joblib")
    np.savez_compressed(tmp / "data.npz", **{name: getattr(split, name) for name in _ARRAYS})
    full_meta = {"id": path.name, "created": now.isoformat(timespec="seconds"), "task": split.task} | meta
    (tmp / "meta.json").write_text(json.dumps(full_meta, indent=2, default=_json_default))
    tmp.rename(path)
    prune(root, keep)
    return path


def list_runs(root: Path = MODELS_DIR) -> list[RunInfo]:
    """Complete saved runs, newest first."""
    if not root.exists():
        return []
    runs = []
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and not path.name.endswith(".tmp") and all((path / f).exists() for f in _FILES):
            try:
                runs.append(RunInfo(path, json.loads((path / "meta.json").read_text())))
            except (OSError, json.JSONDecodeError):
                continue
    return runs


def prune(root: Path = MODELS_DIR, keep: int = MAX_RUNS) -> None:
    for run in list_runs(root)[keep:]:
        shutil.rmtree(run.path, ignore_errors=True)


def load_run(path: Path) -> LoadedRun:
    from lab.model import keras  # lazy: TensorFlow import is slow

    path = Path(path)
    # Inference only: skip restoring the optimizer state.
    model = keras.models.load_model(path / "model.keras", compile=False)
    # joblib uses pickle: only load runs this app wrote itself into MODELS_DIR.
    scalers = joblib.load(path / "scaler.joblib")
    meta = json.loads((path / "meta.json").read_text())
    with np.load(path / "data.npz") as data:
        arrays = {name: data[name] for name in _ARRAYS}
    split = Split(meta["task"], **arrays, x_scaler=scalers["x"], y_scaler=scalers["y"])
    return LoadedRun(path, model, split, meta)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
