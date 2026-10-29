"""Training loop: Keras fit + EarlyStopping, with a per-epoch hook for live charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from lab.config import MONITOR_OPTIONS
from lab.datasets import Split
from lab.model import TrainConfig, build_model, keras

# hook(epoch_1_based, logs, model)
EpochHook = Callable[[int, dict, keras.Model], None]


@dataclass
class TrainResult:
    model: keras.Model
    history: dict[str, list[float]]
    best_epoch: int  # 1-based, by the monitored metric (weights restored only with early stopping)
    epochs_run: int
    stopped_early: bool
    diverged: bool


class _EpochHook(keras.callbacks.Callback):
    def __init__(self, hook: EpochHook):
        super().__init__()
        self._hook = hook

    def on_epoch_end(self, epoch, logs=None):
        self._hook(epoch + 1, {k: float(v) for k, v in (logs or {}).items()}, self.model)


def train(config: TrainConfig, split: Split, on_epoch: EpochHook | None = None) -> TrainResult:
    X_train, X_val = split.X_train, split.X_val
    model = build_model(X_train.shape[1], config)
    mode = MONITOR_OPTIONS[config.task][config.monitor]
    callbacks = [keras.callbacks.TerminateOnNaN()]
    stopper = None
    if config.early_stopping:
        stopper = keras.callbacks.EarlyStopping(
            monitor=config.monitor, mode=mode, patience=config.patience, restore_best_weights=True
        )
        callbacks.append(stopper)
    if on_epoch:
        callbacks.append(_EpochHook(on_epoch))
    fit = model.fit(
        X_train,
        split.y_train_fit,
        validation_data=(X_val, split.y_val_fit),
        epochs=config.epochs,
        batch_size=min(config.batch_size, len(X_train)),
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    history = {k: [float(v) for v in values] for k, values in fit.history.items()}
    epochs_run = len(history.get("loss", []))
    diverged = any(v != v or v in (float("inf"), float("-inf")) for v in history.get("loss", []))
    return TrainResult(
        model=model,
        history=history,
        best_epoch=best_epoch(history.get(config.monitor, []), mode),
        epochs_run=epochs_run,
        stopped_early=bool(stopper and stopper.stopped_epoch > 0),
        diverged=diverged,
    )


def best_epoch(values: list[float], mode: str) -> int:
    """1-based epoch of the best finite value (first one on ties), matching EarlyStopping's choice."""
    best, best_i = None, 0
    for i, v in enumerate(values):
        if v != v:
            continue
        if best is None or (v < best if mode == "min" else v > best):
            best, best_i = v, i
    return best_i + 1
