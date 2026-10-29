"""Step 3: train with live curves; every finished run is saved to the rotating history."""

from __future__ import annotations

import time

import streamlit as st

from lab import charts
from lab import text as T
from lab.config import BATCH_SIZES, L2_VALUES, LEARNING_RATES, MONITOR_OPTIONS, OPTIMIZERS, architecture_label
from lab.config_io import config_to_dict
from lab.separability import linear_regression_baseline
from lab.state import DataContext, active_run, current_config, dataset_spec, models_dir, separability
from lab.views.common import (
    arch_label,
    dataset_label,
    is_1d_regression,
    is_2d_classification,
    predictions_figure,
    show,
)



def _settings() -> bool:
    s = st.session_state
    task = s["task"]
    with st.container(border=True):
        st.segmented_control(T.TRAIN_OPTIMIZER, OPTIMIZERS, format_func=T.OPTIMIZERS.get, key="optimizer",
                             required=True, help=T.HELP["optimizer"])
        st.select_slider(T.TRAIN_LR, LEARNING_RATES, key="lr", help=T.HELP["lr"])
        if s["optimizer"] == "sgd":
            st.slider(T.TRAIN_MOMENTUM, 0.0, 0.99, step=0.01, key="momentum", help=T.HELP["momentum"])
        st.select_slider(T.TRAIN_BATCH, BATCH_SIZES, key="batch_size", help=T.HELP["batch"])
        st.select_slider(T.TRAIN_L2, L2_VALUES, key="l2", format_func=lambda v: T.TRAIN_L2_OFF if v == 0 else f"{v:g}", help=T.HELP["l2"])
        a, b = st.columns(2)
        a.number_input(T.TRAIN_EPOCHS, 10, 3000, step=10, key="epochs", help=T.HELP["epochs"])
        b.number_input(T.TRAIN_PATIENCE, 5, 1000, step=5, key="patience", help=T.HELP["patience"],
                       disabled=not s["early_stopping"])
        st.checkbox(T.TRAIN_EARLY_STOP, key="early_stopping", help=T.HELP["early_stopping"])
        st.selectbox(T.TRAIN_MONITOR, list(MONITOR_OPTIONS[task]), format_func=T.MONITORS.get, key=f"monitor_{task}",
                     help=T.HELP["monitor"])
        st.slider(T.TRAIN_REFRESH, 1, 50, key="refresh", help=T.HELP["refresh"])
        clicked = st.button(T.TRAIN_BUTTON, type="primary", width="stretch", icon=":material/play_arrow:")
    with st.expander(T.TRAIN_LOSS_EXPLAIN_TITLE):
        st.markdown(T.TRAIN_LOSS_EXPLAIN.format(loss=T.LOSS_NAMES[task]))
    return clicked


def render(ctx: DataContext | None) -> None:
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        clicked = _settings()
    with right:
        if clicked and ctx is not None:
            _train_live(ctx)
        else:
            _show_active()


def _stats_row():
    cols = st.columns(4)
    return [c.empty() for c in cols]


def _heading(title: str, help: str | None = None) -> None:
    st.markdown(f"**{title}**", help=help)


LOSS_HELP = T.HELP["loss_chart"] + "\n\n" + T.HELP["dropout_gap"]


def _train_live(ctx: DataContext) -> None:
    from lab import registry
    from lab.metrics import classification_metrics, regression_metrics
    from lab.predict import predict
    from lab.training import train

    config = current_config()
    split, ds = ctx.split, ctx.dataset
    n_features = ds.X.shape[1]
    mode = MONITOR_OPTIONS[config.task][config.monitor]
    show_boundary = is_2d_classification(config.task, n_features) or is_1d_regression(config.task, n_features)

    progress = st.progress(0.0, text=T.TRAIN_RUNNING)
    stats = _stats_row()
    _heading(T.TRAIN_LOSS_TITLE, LOSS_HELP)
    loss_ph = st.empty()
    metric_ph = None
    if config.monitor != "val_loss":
        _heading(T.MONITORS[config.monitor])
        metric_ph = st.empty()
    boundary_ph = None
    if show_boundary:
        _heading(T.TRAIN_BOUNDARY_TITLE if config.task == "classification" else T.TRAIN_CURVE_TITLE)
        boundary_ph = st.empty()

    history: dict[str, list[float]] = {}
    live = {"last": 0.0, "best": None, "best_epoch": None}

    def hook(epoch: int, logs: dict, model) -> None:
        for k, v in logs.items():
            history.setdefault(k, []).append(v)
        value = logs.get(config.monitor)
        if value is not None and (live["best"] is None or (value < live["best"] if mode == "min" else value > live["best"])):
            live["best"], live["best_epoch"] = value, epoch
        now = time.monotonic()
        last = epoch == config.epochs
        if now - live["last"] > 0.25 or epoch == 1 or last:
            live["last"] = now
            progress.progress(epoch / config.epochs, text=f"{T.TRAIN_RUNNING} {epoch}/{config.epochs}")
            stats[0].metric(T.TRAIN_EPOCH, epoch)
            stats[1].metric(T.TRAIN_LOSS_SHORT, f"{logs.get('loss', float('nan')):.4f}")
            stats[2].metric(T.VAL_LOSS_SHORT, f"{logs.get('val_loss', float('nan')):.4f}")
            best_label = T.TRAIN_BEST_VALUE.format(metric=T.MONITORS_SHORT[config.monitor])
            stats[3].metric(best_label, f"{live['best']:.4f}" if live["best"] is not None else "–",
                            delta=f"{T.TRAIN_EPOCH} {live['best_epoch']}", delta_color="off", delta_arrow="off")
            show(charts.loss_curves(history, live["best_epoch"]), key=f"live_loss_{epoch}", where=loss_ph)
            if metric_ph is not None:
                metric = config.monitor.removeprefix("val_")
                show(charts.metric_curves(history, metric, None, live["best_epoch"]),
                     key=f"live_metric_{epoch}", where=metric_ph)
        if boundary_ph is not None and (epoch == 1 or epoch % st.session_state["refresh"] == 0):
            fig = predictions_figure(model, split, ds.feature_names, ds.target_name, ds.class_names, height=400)
            show(fig, key=f"live_boundary_{epoch}", where=boundary_ph)

    result = train(config, split, on_epoch=hook)

    def scores(part: str) -> dict:
        X, y = split.part(part)
        out = predict(result.model, split, X)
        return classification_metrics(y, out) if config.task == "classification" else regression_metrics(y, out)

    if config.task == "classification":
        sep = separability(ds, ctx.data_id)
        sep_info = {"linearly_separable": sep.separable, "linear_accuracy": sep.accuracy}
    else:
        sep_info = {"linear_r2": linear_regression_baseline(ds.X, ds.y)}
    meta = {
        "feature_names": list(ds.feature_names),
        "target_name": ds.target_name,
        "class_names": list(ds.class_names) if ds.class_names else None,
        "config": config_to_dict(config, dataset_spec()),
        "architecture": architecture_label(n_features, config.layers),
        "separability": sep_info,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "diverged": result.diverged,
        "metrics": {part: scores(part) for part in ("train", "val", "test")},
        "history": result.history,
    }
    path = registry.save_run(result.model, split, meta, root=models_dir())
    st.session_state["pending_active"] = path.name
    st.session_state["train_notice"] = "diverged" if result.diverged else "done"
    st.rerun()


def _show_active() -> None:
    run = active_run()
    if run is None:
        st.info(T.TRAIN_EMPTY)
        return
    meta = run.meta
    notice = st.session_state.pop("train_notice", None)
    if notice == "diverged":
        st.error(T.TRAIN_DIVERGED)
    elif notice == "done":
        if meta["config"].get("early_stopping", True):
            early = T.TRAIN_EARLY if meta["stopped_early"] else ""
            st.success(T.TRAIN_DONE.format(epochs=meta["epochs_run"], early=early, best=meta["best_epoch"]))
        else:
            st.success(T.TRAIN_DONE_FULL.format(epochs=meta["epochs_run"], best=meta["best_epoch"]))
    st.caption(T.TRAIN_SHOWING.format(label=f"{arch_label(meta)} · {dataset_label(meta)}"))

    history, best = meta["history"], meta["best_epoch"]
    monitor = meta["config"]["monitor"]
    cols = st.columns(4)
    cols[0].metric(T.TRAIN_BEST_EPOCH, best, delta=T.TRAIN_BEST_OF.format(epochs=meta["epochs_run"]), delta_color="off",
                   delta_arrow="off")
    cols[1].metric(T.TRAIN_LOSS_SHORT, help=T.TRAIN_LOSS, value=f"{history['loss'][best - 1]:.4f}")
    cols[2].metric(T.VAL_LOSS_SHORT, help=T.VAL_LOSS, value=f"{history['val_loss'][best - 1]:.4f}")
    if monitor != "val_loss":
        cols[3].metric(T.MONITORS_SHORT[monitor], help=T.MONITORS[monitor], value=f"{history[monitor][best - 1]:.4f}")

    _heading(T.TRAIN_LOSS_TITLE, LOSS_HELP)
    show(charts.loss_curves(history, best), key="saved_loss")
    if monitor != "val_loss":
        _heading(T.MONITORS[monitor])
        show(charts.metric_curves(history, monitor.removeprefix("val_"), None, best), key="saved_metric")
    n_features = len(meta["feature_names"])
    if is_2d_classification(meta["task"], n_features) or is_1d_regression(meta["task"], n_features):
        title = T.TRAIN_BOUNDARY_TITLE if meta["task"] == "classification" else T.TRAIN_CURVE_TITLE
        _heading(title)
        show(predictions_figure(run.model, run.split, meta["feature_names"], meta["target_name"], meta["class_names"],
                                height=400), key="saved_boundary")
