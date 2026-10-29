"""Helpers shared by the views."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lab import charts
from lab import text as T
from lab.config import architecture_label, output_spec

PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False}


def show(fig: go.Figure, key: str | None = None, where=None) -> None:
    (where or st).plotly_chart(fig, theme=None, key=key, config=PLOT_CONFIG)


def dataset_label(meta: dict) -> str:
    data = meta["config"].get("dataset", {})
    name = data.get("name", "?")
    if name == "csv":
        return data.get("csv_file") or "CSV"
    return T.DATASETS.get(name, name).split(" · ")[0]


def arch_label(meta: dict) -> str:
    return architecture_label(len(meta["feature_names"]), meta["config"]["layers"])


def key_metric(meta: dict) -> str:
    monitor = meta["config"]["monitor"]
    values = meta["history"].get(monitor, [])
    best = meta["best_epoch"]
    if not values or best > len(values):
        return ""
    return f"{T.MONITORS_SHORT[monitor]} {values[best - 1]:.3f}"


def run_option_label(meta: dict) -> str:
    created = datetime.fromisoformat(meta["created"]).strftime("%d/%m %H:%M")
    return f"{created} · {dataset_label(meta)}"


def is_2d_classification(task: str, n_features: int) -> bool:
    return task == "classification" and n_features == 2


def is_1d_regression(task: str, n_features: int) -> bool:
    return task == "regression" and n_features == 1


def hidden_activations(meta_or_config: dict) -> list[str]:
    config = meta_or_config.get("config", meta_or_config)
    return [layer["activation"] for layer in config["layers"]] + [output_spec(config["task"])[0]]


def predictions_figure(model, split, feature_names, target_name, class_names, point=None, height=440):
    """Decision boundary (2D classification) or fitted curve (1D regression) over all the data."""
    from lab.predict import predict

    X = np.vstack([split.X_train_raw, split.X_val_raw, split.X_test_raw])
    y = np.concatenate([split.y_train, split.y_val, split.y_test])
    if split.task == "classification":
        return charts.decision_boundary(lambda g: predict(model, split, g), X, y, class_names, feature_names,
                                        point=point, height=height)
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    curve = (xs, predict(model, split, xs.reshape(-1, 1)))
    return charts.regression_scatter(X[:, 0], y, feature_names[0], target_name, curve=curve, point=point, height=height)
