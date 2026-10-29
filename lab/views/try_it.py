"""Step 5: feed one input to the active model and look inside."""

from __future__ import annotations

import numpy as np
import streamlit as st

from lab import charts
from lab import text as T
from lab.inspection import dropout_rates, forward_activations, layer_sizes, layer_weights
from lab.predict import predict
from lab.registry import LoadedRun
from lab.views.common import hidden_activations, is_1d_regression, is_2d_classification, predictions_figure, show
from lab.views.results import active_header


def _keys(run: LoadedRun) -> list[str]:
    return [f"try_{run.meta['id']}_{i}" for i in range(len(run.meta["feature_names"]))]


def _reset(keys: list[str], means: np.ndarray) -> None:
    for key, mean in zip(keys, means):
        st.session_state[key] = float(round(mean, 3))


def _inputs(run: LoadedRun) -> np.ndarray:
    names = run.meta["feature_names"]
    X = run.split.X_train_raw
    means, stds = X.mean(axis=0), X.std(axis=0)
    keys = _keys(run)
    for key, mean in zip(keys, means):
        st.session_state.setdefault(key, float(round(mean, 3)))
    st.markdown(f"**{T.TRY_INPUTS}**", help=T.TRY_INPUTS_HELP)
    n_cols = 1 if len(names) <= 4 else 2
    cols = st.columns(n_cols)
    for i, (name, key) in enumerate(zip(names, keys)):
        step = float(max(stds[i] / 10, 1e-3))
        cols[i % n_cols].number_input(name, step=step, format="%.3f", key=key)
    st.button(T.TRY_RESET, on_click=_reset, args=(keys, means), icon=":material/restart_alt:")
    return np.array([[st.session_state[k] for k in keys]], dtype=float)


def render(run: LoadedRun | None) -> None:
    if run is None:
        st.info(T.RESULTS_EMPTY)
        return
    active_header(run)
    meta, model, split = run.meta, run.model, run.split
    left, right = st.columns([1, 2], gap="large")
    with left:
        with st.container(border=True):
            x = _inputs(run)
    output = float(predict(model, split, x)[0])
    with right:
        a, b = st.columns(2)
        if meta["task"] == "classification":
            threshold = st.session_state["threshold"]
            classes = meta["class_names"]
            a.metric(T.TRY_PROBA.format(cls=classes[1]), f"{output:.3f}")
            b.metric(T.TRY_CLASS, classes[int(output >= threshold)], help=f"{T.RESULTS_THRESHOLD}: {threshold:.2f}")
        else:
            a.metric(T.TRY_VALUE, f"{output:.4g}")
        st.markdown(f"**{T.TRY_NETWORK}**", help=T.TRY_NETWORK_HELP)
        values = [layer[0] for layer in forward_activations(model, split.transform_x(x))]
        fig = charts.network_diagram(
            layer_sizes(model), hidden_activations(meta), dropout_rates(model), meta["feature_names"],
            weights=layer_weights(model), node_values=values,
        )
        show(fig, key="try_diagram")

    n_features = len(meta["feature_names"])
    if is_2d_classification(meta["task"], n_features) or is_1d_regression(meta["task"], n_features):
        point = (x[0, 0], x[0, 1]) if n_features == 2 else (x[0, 0], output)
        show(predictions_figure(model, split, meta["feature_names"], meta["target_name"], meta["class_names"],
                                point=point), key="try_boundary")
