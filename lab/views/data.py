"""Step 1: pick and inspect the data."""

from __future__ import annotations

import streamlit as st

from lab import charts
from lab import text as T
from lab.datasets import CLASSIFICATION_DATASETS, REAL_DATASETS, REGRESSION_DATASETS
from lab.projection import pca_projection
from lab.separability import linear_regression_baseline, linear_regression_line
from lab.state import DataContext, dataset_key, read_csv_columns, separability
from lab.views.common import show


def render_controls() -> None:
    s = st.session_state
    st.segmented_control(T.DATA_TASK, options=list(T.TASKS), format_func=T.TASKS.get, key="task", required=True,
                         help=T.HELP["task"])
    task = s["task"]
    options = list(CLASSIFICATION_DATASETS if task == "classification" else REGRESSION_DATASETS) + ["csv"]
    st.selectbox(T.DATA_DATASET, options, format_func=T.DATASETS.get, key=f"dataset_{task}")
    key = dataset_key()

    if key == "csv":
        upload = st.file_uploader(T.DATA_CSV, type="csv", key="csv_file")
        if upload is not None:
            st.selectbox(T.DATA_CSV_TARGET, read_csv_columns(upload.getvalue()), key="csv_target")
    elif key in REAL_DATASETS:
        st.caption(T.DATA_REAL_FIXED)
    else:
        st.slider(T.DATA_SAMPLES, 50, 1000, step=50, key="n_samples", help=T.HELP["samples"])
        st.slider(T.DATA_NOISE, 0.0, 1.0, step=0.05, key="noise", help=T.HELP["noise"])

    a, b, c = st.columns(3)
    a.slider(T.DATA_VAL, 5, 30, step=5, key="val_pct", help=T.HELP["val"])
    b.slider(T.DATA_TEST, 5, 30, step=5, key="test_pct", help=T.HELP["test"])
    c.number_input(T.DATA_SEED, 0, 9999, step=1, key="seed", help=T.HELP["seed"])
    st.toggle(T.DATA_STANDARDIZE, key="standardize", help=T.HELP["standardize"])


def render_view(ctx: DataContext | None) -> None:
    if ctx is None:
        st.info(st.session_state.get("csv_error", T.DATA_CSV_WAITING))
        return
    ds, split = ctx.dataset, ctx.split
    st.caption(T.DATA_SPLIT_SUMMARY.format(train=len(split.y_train), val=len(split.y_val), test=len(split.y_test)))
    if ds.task == "classification":
        _classification(ctx)
    else:
        _regression(ctx)


def _classification(ctx: DataContext) -> None:
    ds = ctx.dataset
    result = separability(ds, ctx.data_id)
    if result.separable:
        st.badge(T.SEP_YES, icon=":material/check:", color="green")
    else:
        st.badge(T.SEP_NO, icon=":material/close:", color="orange")
    st.caption(T.SEP_DETAIL.format(acc=result.accuracy))

    if ds.X.shape[1] == 2:
        show(charts.scatter_2d(ds.X, ds.y, ds.class_names, ds.feature_names, linear_score=result.model.decision_function))
    else:
        Z, ratios = pca_projection(ds.X, n_components=2)
        show(charts.pca_scatter(Z, ds.y, ds.class_names, ratios))
        st.caption(T.SEP_PCA_CAVEAT.format(total=ratios.sum()))
    with st.expander(T.SEP_WHAT):
        st.markdown(T.SEP_EXPLAIN)


def _regression(ctx: DataContext) -> None:
    ds = ctx.dataset
    r2 = linear_regression_baseline(ds.X, ds.y)
    st.badge(T.REG_BASELINE.format(r2=r2), color="blue", help=T.REG_BASELINE_HELP)
    if ds.X.shape[1] == 1:
        line = linear_regression_line(ds.X[:, 0], ds.y)
        show(charts.regression_scatter(ds.X[:, 0], ds.y, ds.feature_names[0], ds.target_name, baseline=line))
    else:
        Z, ratios = pca_projection(ds.X, n_components=1)
        show(charts.regression_scatter(Z[:, 0], ds.y, T.PCA_AXES[0].format(pct=ratios[0]), ds.target_name))
        st.caption(T.REG_PC1.format(total=ratios[0]))


def render(ctx_factory) -> DataContext | None:
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        render_controls()
    ctx = ctx_factory()
    with right:
        render_view(ctx)
    return ctx
