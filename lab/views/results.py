"""Step 4: metrics of the active saved model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lab import charts
from lab import text as T
from lab.metrics import classification_metrics, confusion, regression_metrics, roc
from lab.predict import predict
from lab.registry import LoadedRun
from lab.views import config_panel
from lab.views.common import arch_label, dataset_label, show

PARTS = ("train", "val", "test")


def active_header(run: LoadedRun) -> None:
    meta = run.meta
    restored = meta["config"].get("early_stopping", True)
    template, help_text = (
        (T.RESULTS_ACTIVE, T.RESULTS_ACTIVE_HELP) if restored else (T.RESULTS_ACTIVE_LAST, T.RESULTS_ACTIVE_LAST_HELP)
    )
    st.markdown(
        template.format(label=arch_label(meta), dataset=dataset_label(meta), best=meta["best_epoch"],
                        epochs=meta["epochs_run"]),
        help=help_text,
    )


def render(run: LoadedRun | None) -> None:
    if run is None:
        st.info(T.RESULTS_EMPTY)
        return
    active_header(run)
    outputs = {part: (run.split.part(part)[1], predict(run.model, run.split, run.split.part(part)[0])) for part in PARTS}
    if run.meta["task"] == "classification":
        _classification(run, outputs)
    else:
        _regression(outputs)
    with st.expander(T.RESULTS_METRICS_EXPLAIN_TITLE):
        st.markdown(T.RESULTS_METRICS_EXPLAIN[run.meta["task"]])
    config_panel.render_run_exports(run)


def _table(rows: dict[str, dict[str, float]]) -> None:
    df = pd.DataFrame(rows).T.rename(index=T.SPLITS, columns=T.METRICS)
    st.dataframe(df.style.format("{:.3f}"), width="stretch")


def _split_picker() -> str:
    return st.segmented_control(T.RESULTS_SPLIT, PARTS, format_func=T.SPLITS.get, key="results_split", required=True)


def _classification(run: LoadedRun, outputs: dict) -> None:
    a, b = st.columns([2, 1], gap="large")
    threshold = a.slider(T.RESULTS_THRESHOLD, 0.01, 0.99, step=0.01, key="threshold", help=T.HELP["threshold"])
    with b:
        part = _split_picker()
    _table({p: classification_metrics(y, proba, threshold) for p, (y, proba) in outputs.items()})

    y, proba = outputs[part]
    cm = confusion(y, proba, threshold)
    tn, fp, fn, tp = cm.ravel()
    point = (fp / max(fp + tn, 1), tp / max(tp + fn, 1))
    fpr, tpr, _ = roc(y, proba)
    auc = classification_metrics(y, proba, threshold)["roc_auc"]
    left, right = st.columns(2, gap="large")
    with left:
        show(charts.confusion_heatmap(cm, run.meta["class_names"]), key="cm")
    with right:
        show(charts.roc_chart(fpr, tpr, auc, point), key="roc")


def _regression(outputs: dict) -> None:
    _table({p: regression_metrics(y, pred) for p, (y, pred) in outputs.items()})
    part = _split_picker()
    y, pred = outputs[part]
    left, right = st.columns(2, gap="large")
    with left:
        show(charts.predicted_vs_true(y, pred), key="pred_true")
    with right:
        show(charts.residuals_chart(y, pred), key="residuals")
