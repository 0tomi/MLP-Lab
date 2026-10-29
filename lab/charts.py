"""Plotly figures sharing one calm palette, in a light and a dark variant."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import plotly.graph_objects as go

from lab import text as T

# ---------------------------------------------------------------- palette
@dataclass(frozen=True)
class Palette:
    ink: str  # main text, boundary lines
    muted: str  # secondary text
    grid: str
    bg: str  # page background, also used as marker outline
    neg: str  # class 0, negative weights/activations, validation
    pos: str  # class 1, positive weights/activations, train
    accent: str  # best epoch, user input
    neutral: str  # reference lines, untrained edges
    mid: str  # centre of diverging scales
    badge: str  # dropout badge background
    cm_high: str  # strongest confusion-matrix cell; dark enough for light label text
    extra: tuple[str, str] = ("#9C7BB5", "#C9A646")

    @property
    def diverging(self) -> list:
        return [[0.0, self.neg], [0.5, self.mid], [1.0, self.pos]]


# One place per theme; keep in sync with .streamlit/config.toml.
PALETTES = {
    "light": Palette(ink="#26293B", muted="#7A7E8C", grid="#E7E2D8", bg="#FAF8F4", neg="#D9825B", pos="#3F6E9A",
                     accent="#5E9E7E", neutral="#B9BDC7", mid="#F7F3EC", badge="#EFE8DA",
                     cm_high="#3F6E9A"),
    "dark": Palette(ink="#E6E2DA", muted="#9EA2AE", grid="#2E3240", bg="#16181F", neg="#E8956E", pos="#73A0CC",
                    accent="#7DBE9C", neutral="#5C6274", mid="#262935", badge="#2E3140",
                    cm_high="#3D6A97", extra=("#B79AD0", "#D9BC63")),
}
FONT = "Figtree, system-ui, sans-serif"
_THEME: ContextVar[str] = ContextVar("chart_theme", default="light")


def use_theme(theme: str | None) -> None:
    """Select the palette for figures built in this script run (per session thread)."""
    _THEME.set(theme if theme in PALETTES else "light")


def palette(theme: str | None = None) -> Palette:
    return PALETTES[theme or _THEME.get()]


def plotly_template(theme: str | None = None) -> go.layout.Template:
    p = palette(theme)
    axis = dict(gridcolor=p.grid, zerolinecolor=p.grid, linecolor=p.grid, ticks="", color=p.muted,
                title=dict(font=dict(color=p.ink)))
    return go.layout.Template(
        layout=dict(
            font=dict(family=FONT, color=p.ink, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=[p.pos, p.neg, p.accent, *p.extra],
            margin=dict(l=48, r=16, t=40, b=44),
            xaxis=axis,
            yaxis=axis,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                        font=dict(color=p.ink)),
            hoverlabel=dict(font=dict(family=FONT, color=p.ink), bgcolor=p.bg, bordercolor=p.grid),
            title=dict(font=dict(size=15, color=p.ink), x=0, xanchor="left"),
            coloraxis=dict(colorbar=dict(tickfont=dict(color=p.muted))),
        )
    )


def _fig(*traces) -> go.Figure:
    return go.Figure(list(traces), layout=dict(template=plotly_template()))


PredictFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------- data views
def _class_traces(fig: go.Figure, X: np.ndarray, y: np.ndarray, class_names: Sequence[str], size: int = 7) -> None:
    p = palette()
    for cls in (0, 1):
        mask = y == cls
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers", name=str(class_names[cls]),
                marker=dict(color=(p.neg, p.pos)[cls], size=size, opacity=0.9, line=dict(color=p.bg, width=0.8)),
            )
        )


def _grid(X: np.ndarray, steps: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = (hi - lo) * 0.12 + 1e-9
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], steps)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], steps)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, np.c_[xx.ravel(), yy.ravel()]


def _square(fig: go.Figure, feature_names: Sequence[str], height: int = 440) -> go.Figure:
    fig.update_layout(height=height, xaxis_title=feature_names[0], yaxis_title=feature_names[1])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def scatter_2d(X, y, class_names, feature_names, linear_score: PredictFn | None = None) -> go.Figure:
    """Points coloured by class, optionally with the linear-separability boundary."""
    fig = _fig()
    if linear_score is not None:
        xs, ys, grid = _grid(X)
        z = linear_score(grid).reshape(len(ys), len(xs))
        fig.add_trace(
            go.Contour(
                x=xs, y=ys, z=z, contours=dict(start=0, end=0, size=1, coloring="none"),
                line=dict(color=palette().ink, width=2, dash="dash"), showscale=False, name=T.LEGEND_LINEAR, showlegend=True,
                hoverinfo="skip",
            )
        )
    _class_traces(fig, X, y, class_names)
    return _square(fig, feature_names)


def pca_scatter(Z, y, class_names, ratios) -> go.Figure:
    """Classes on the first two principal components (from lab.projection.pca_projection)."""
    fig = _fig()
    _class_traces(fig, Z, y, class_names)
    return _square(fig, [axis.format(pct=r) for axis, r in zip(T.PCA_AXES, ratios)])


def regression_scatter(x, y, x_name, y_name, curve: tuple | None = None, baseline: tuple | None = None,
                       point: tuple | None = None, height: int = 420) -> go.Figure:
    p = palette()
    fig = _fig(
        go.Scatter(x=x, y=y, mode="markers", showlegend=False,
                   marker=dict(color=p.pos, size=6, opacity=0.6, line=dict(color=p.bg, width=0.6)))
    )
    if baseline is not None:
        fig.add_trace(go.Scatter(x=baseline[0], y=baseline[1], mode="lines", name=T.LEGEND_LINEAR,
                                 line=dict(color=p.ink, width=2, dash="dash")))
    if curve is not None:
        fig.add_trace(go.Scatter(x=curve[0], y=curve[1], mode="lines", name=T.LEGEND_MODEL, line=dict(color=p.neg, width=3)))
    if point is not None:
        fig.add_trace(_point_trace(point))
    fig.update_layout(height=height, xaxis_title=x_name, yaxis_title=y_name)
    return fig


def _point_trace(point) -> go.Scatter:
    return go.Scatter(
        x=[point[0]], y=[point[1]], mode="markers", name=T.TRY_POINT,
        marker=dict(symbol="star", size=20, color=palette().accent, line=dict(color=palette().bg, width=1.5)),
    )


def decision_boundary(predict_proba: PredictFn, X, y, class_names, feature_names, point=None, height=440) -> go.Figure:
    xs, ys, grid = _grid(X, steps=100)
    z = predict_proba(grid).reshape(len(ys), len(xs))
    p = palette()
    fig = _fig(
        go.Heatmap(x=xs, y=ys, z=z, zmin=0, zmax=1, colorscale=p.diverging, opacity=0.8, showscale=True,
                   colorbar=dict(title=dict(text=T.AXIS_PROBA, side="right", font=dict(color=p.ink)), thickness=10,
                                 outlinewidth=0, tickfont=dict(color=p.muted)),
                   hovertemplate="P = %{z:.2f}<extra></extra>")
    )
    fig.add_trace(
        go.Contour(x=xs, y=ys, z=z, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                   line=dict(color=p.ink, width=2), showscale=False, hoverinfo="skip", showlegend=False)
    )
    _class_traces(fig, X, y, class_names, size=6)
    if point is not None:
        fig.add_trace(_point_trace(point))
    return _square(fig, feature_names, height)


# ---------------------------------------------------------------- training
def loss_curves(history: dict, best_epoch: int | None = None, height: int = 300) -> go.Figure:
    return metric_curves(history, "loss", None, best_epoch, height)


def metric_curves(history: dict, key: str, title: str | None, best_epoch: int | None = None, height: int = 300) -> go.Figure:
    p = palette()
    fig = _fig()
    for name, label, color in ((key, T.LEGEND_TRAIN, p.pos), (f"val_{key}", T.LEGEND_VAL, p.neg)):
        values = history.get(name, [])
        fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="lines", name=label,
                                 line=dict(color=color, width=2.5)))
    if best_epoch:
        fig.add_vline(x=best_epoch, line=dict(color=p.accent, width=2, dash="dot"))
        fig.add_annotation(x=best_epoch, y=1, yref="paper", text=T.LEGEND_BEST, showarrow=False,
                           font=dict(color=p.accent, size=12), xanchor="left", yanchor="top", xshift=4)
    fig.update_layout(title=title, height=height, xaxis_title=T.AXIS_EPOCH, hovermode="x unified",
                      margin=dict(t=30 if title is None else 40))
    if len(history.get(key, [])) <= 10:
        fig.update_xaxes(dtick=1)  # no fractional epochs on short runs
    return fig


# ---------------------------------------------------------------- network diagram
def visible_indices(n: int, max_nodes: int, values: np.ndarray | None = None) -> list[int | None]:
    """Neuron indices to draw (None marks the ellipsis node).

    Wide layers show max_nodes - 1 neurons: the first and last ones, or, when activations are given,
    the ones with the largest |activation| (kept in index order, ellipsis last).
    """
    if n <= max_nodes:
        return list(range(n))
    if values is not None:
        top = np.argsort(-np.abs(np.asarray(values).ravel()), kind="stable")[: max_nodes - 1]
        return sorted(int(i) for i in top) + [None]
    head = max_nodes // 2
    tail = max_nodes - head - 1
    return list(range(head)) + [None] + list(range(n - tail, n))


def network_diagram(
    sizes: Sequence[int],
    activations: Sequence[str],
    dropouts: Sequence[float],
    feature_names: Sequence[str] | None = None,
    weights: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    node_values: Sequence[np.ndarray] | None = None,
    max_nodes: int = 12,
    height: int | None = None,
) -> go.Figure:
    """Layers left to right. activations/dropouts describe the hidden layers; output activation is last in activations."""
    n_layers = len(sizes)
    slots = [visible_indices(n, max_nodes, None if node_values is None else node_values[li])
             for li, n in enumerate(sizes)]
    tallest = max(len(s) for s in slots)
    pos: list[dict[int | None, tuple[float, float]]] = []
    for li, layer_slots in enumerate(slots):
        offset = (tallest - len(layer_slots)) / 2
        pos.append({idx: (li, (tallest - 1) - (j + offset)) for j, idx in enumerate(layer_slots)})

    p = palette()
    fig = _fig()
    _edges(fig, slots, pos, weights)
    _nodes(fig, sizes, slots, pos, node_values, feature_names, probability_output=activations[-1] == "sigmoid")

    for li, n in enumerate(sizes):
        if li == 0:
            title, sub = T.DIAGRAM_INPUT, f"{n}"
        elif li == n_layers - 1:
            title, sub = T.DIAGRAM_OUTPUT, f"{n} · {T.ACTIVATIONS.get(activations[-1], activations[-1])}"
        else:
            act = activations[li - 1]
            title, sub = T.DIAGRAM_HIDDEN.format(i=li), f"{n} · {T.ACTIVATIONS.get(act, act)}"
        fig.add_annotation(x=li, y=tallest - 0.1, text=f"<b>{title}</b><br><span style='color:{p.muted}'>{sub}</span>",
                           showarrow=False, yanchor="bottom", font=dict(size=12))
        if 0 < li < n_layers - 1 and dropouts[li - 1] > 0:
            fig.add_annotation(x=li, y=-0.9, text=T.DIAGRAM_DROPOUT.format(rate=dropouts[li - 1]), showarrow=False,
                               bgcolor=p.badge, borderpad=4, font=dict(size=11, color=p.ink), yanchor="top")

    fig.update_layout(
        height=height or max(300, 70 + 46 * tallest), showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=90 if feature_names else 20, r=60, t=20, b=20),
        xaxis=dict(visible=False, range=[-0.5, n_layers - 0.5]),
        yaxis=dict(visible=False, range=[-1.6, tallest + 0.9]),
    )
    return fig


def _edges(fig, slots, pos, weights) -> None:
    p = palette()
    bins: dict[tuple[int, int], tuple[list, list]] = {}
    for li in range(len(slots) - 1):
        src = [a for a in slots[li] if a is not None]
        dst = [b for b in slots[li + 1] if b is not None]
        W = weights[li][0] if weights is not None else None
        # widths relative to the largest edge actually drawn, so truncated wide layers stay readable
        scale = float(np.abs(W[np.ix_(src, dst)]).max()) + 1e-12 if W is not None else 1.0
        for a in src:
            for b in dst:
                if W is None:
                    key = (0, 0)
                else:
                    w = W[a, b]
                    key = (1 if w >= 0 else -1, min(4, int(abs(w) / scale * 5)))
                xs, ys = bins.setdefault(key, ([], []))
                (x0, y0), (x1, y1) = pos[li][a], pos[li + 1][b]
                xs += [x0, x1, None]
                ys += [y0, y1, None]
    for (sign, level), (xs, ys) in bins.items():
        if sign == 0:
            line = dict(color=p.neutral, width=1)
            opacity = 0.6
        else:
            line = dict(color=p.pos if sign > 0 else p.neg, width=0.6 + 1.2 * level)
            opacity = 0.3 + 0.15 * level
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=line, opacity=opacity, hoverinfo="skip"))


def _nodes(fig, sizes, slots, pos, node_values, feature_names, probability_output=False) -> None:
    p = palette()
    last = len(sizes) - 1
    base_colors = [p.accent] + [p.pos] * (last - 1) + [p.neg]
    for li, layer_slots in enumerate(slots):
        idxs = [i for i in layer_slots if i is not None]
        xs = [pos[li][i][0] for i in idxs]
        ys = [pos[li][i][1] for i in idxs]
        if node_values is not None:
            vals = np.asarray(node_values[li]).ravel()
            shown = vals[idxs]
            if li == last and probability_output:
                color = 2 * shown - 1  # probability 0..1 -> -1..1
            else:
                color = shown / (float(np.abs(vals).max()) + 1e-12)
            marker = dict(size=30, color=color, cmin=-1, cmax=1, colorscale=p.diverging,
                          line=dict(color=p.muted, width=1))
            labels = [f"{v:.2f}" for v in shown]
            # strong fills take the page colour as text, weak ones the ink colour: readable in both themes
            text_colors = [p.bg if abs(c) > 0.55 else p.ink for c in np.ravel(color)]
            hover = [f"{T.DIAGRAM_NEURON.format(i=i + 1)}: {v:.4f}" for i, v in zip(idxs, shown)]
        else:
            marker = dict(size=26, color=base_colors[li], line=dict(color=p.bg, width=2))
            labels, hover = None, [T.DIAGRAM_NEURON.format(i=i + 1) for i in idxs]
            text_colors = p.ink
        if li == 0 and feature_names is not None:
            hover = [f"{feature_names[i]}" + (f": {h.split(': ')[1]}" if ": " in h else "") for i, h in zip(idxs, hover)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text" if labels else "markers", marker=marker, text=labels,
                                 textfont=dict(size=9, color=text_colors), hovertext=hover, hoverinfo="text"))
        if li == 0 and feature_names is not None and len(idxs) <= 12:
            fig.add_trace(go.Scatter(x=[x - 0.17 for x in xs], y=ys, mode="text", hoverinfo="skip",
                                     text=[_short(feature_names[i]) for i in idxs], textposition="middle left",
                                     textfont=dict(size=11, color=p.muted)))
        if None in layer_slots:
            x, y = pos[li][None]
            hidden = sizes[li] - len(idxs)
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="text", text=[f"⋮ {T.DIAGRAM_MORE.format(n=hidden)}"],
                                     textfont=dict(size=12, color=p.muted), hoverinfo="skip"))


def _short(name: str, n: int = 14) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


# ---------------------------------------------------------------- results
def confusion_heatmap(cm: np.ndarray, class_names: Sequence[str]) -> go.Figure:
    names = [str(c) for c in class_names]
    total = cm.sum() or 1
    text = [[f"<b>{v}</b><br>{v / total:.0%}" for v in row] for row in cm]
    p = palette()
    # Plotly picks a light or dark label per cell from the cell colour, so text stays readable in both themes.
    fig = _fig(
        go.Heatmap(z=cm, x=names, y=names, text=text, texttemplate="%{text}", colorscale=[[0, p.mid], [1, p.cm_high]],
                   showscale=False, hovertemplate=T.RESULTS_CM_HOVER + "<extra></extra>",
                   textfont=dict(size=15))
    )
    fig.update_layout(title=T.RESULTS_CM, height=360, xaxis_title=T.AXIS_PRED_CLASS, yaxis_title=T.AXIS_TRUE_CLASS)
    fig.update_yaxes(type="category", autorange="reversed", showgrid=False)
    fig.update_xaxes(type="category", showgrid=False, side="bottom")
    return fig


def roc_chart(fpr, tpr, auc: float, operating_point: tuple[float, float]) -> go.Figure:
    p = palette()
    fig = _fig()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=p.neutral, dash="dash"), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=p.pos, width=3), name=f"AUC {auc:.3f}",
                             fill="tozeroy", fillcolor=_alpha(p.pos, 0.12)))
    fig.add_trace(go.Scatter(x=[operating_point[0]], y=[operating_point[1]], mode="markers", name=T.RESULTS_THRESHOLD,
                             marker=dict(size=13, color=p.accent, line=dict(color=p.bg, width=1.5))))
    fig.update_layout(title=T.RESULTS_ROC, height=360, xaxis_title=T.AXIS_FPR, yaxis_title=T.AXIS_TPR)
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig


def predicted_vs_true(y, pred) -> go.Figure:
    lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
    p = palette()
    fig = _fig()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=p.neutral, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=y, y=pred, mode="markers", marker=dict(color=p.pos, size=7, opacity=0.75,
                                                                      line=dict(color=p.bg, width=0.6))))
    fig.update_layout(title=T.RESULTS_PRED_TRUE, height=360, showlegend=False, xaxis_title=T.AXIS_TRUE,
                      yaxis_title=T.AXIS_PRED)
    return fig


def residuals_chart(y, pred) -> go.Figure:
    p = palette()
    fig = _fig(go.Scatter(x=pred, y=y - pred, mode="markers",
                          marker=dict(color=p.neg, size=7, opacity=0.75, line=dict(color=p.bg, width=0.6))))
    fig.add_hline(y=0, line=dict(color=p.neutral, dash="dash"))
    fig.update_layout(title=T.RESULTS_RESIDUALS, height=360, showlegend=False, xaxis_title=T.AXIS_PRED,
                      yaxis_title=T.AXIS_RESIDUAL)
    return fig


def _alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"
