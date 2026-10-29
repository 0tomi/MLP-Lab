"""Step 2: design the network."""

from __future__ import annotations

import streamlit as st

from lab import charts
from lab import text as T
from lab.config import ACTIVATIONS, MAX_HIDDEN, MAX_UNITS, count_parameters, layer_sizes_of, output_spec
from lab.state import DataContext, current_layers
from lab.views import config_panel
from lab.views.common import show


def render(ctx: DataContext | None) -> None:
    task = st.session_state["task"]
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        st.slider(T.ARCH_LAYERS, 0, MAX_HIDDEN, key="n_hidden", help=T.HELP["hidden_layers"])
        for i in range(1, st.session_state["n_hidden"] + 1):
            with st.container(border=True):
                st.markdown(f"**{T.ARCH_LAYER.format(i=i)}**")
                a, b, c = st.columns([1, 1.2, 1.2])
                a.number_input(T.ARCH_UNITS, 1, MAX_UNITS, step=1, key=f"units_{i}", help=T.HELP["units"])
                b.selectbox(T.ARCH_ACTIVATION, ACTIVATIONS, format_func=T.ACTIVATIONS.get, key=f"act_{i}",
                            help=T.HELP["activation"])
                c.slider(T.ARCH_DROPOUT, 0.0, 0.8, step=0.05, key=f"drop_{i}", help=T.HELP["dropout"])
        st.caption(T.ARCH_OUTPUT[task])

    layers = current_layers()
    feature_names = ctx.dataset.feature_names if ctx else None
    sizes = layer_sizes_of(len(feature_names) if feature_names else 2, layers)
    with right:
        total = count_parameters(sizes)
        terms = " + ".join(f"{n}·{prev}+{n}" for prev, n in zip(sizes[:-1], sizes[1:]))
        total_text = f"{total:,}".replace(",", T.THOUSANDS_SEP)
        st.metric(T.ARCH_PARAMS, total_text, help=T.ARCH_FORMULA_HELP)
        st.caption(f"Σ (nℓ·nℓ₋₁ + nℓ) = {terms} = {total_text}")
        activations = [layer.activation for layer in layers] + [output_spec(task)[0]]
        dropouts = [layer.dropout for layer in layers]
        show(charts.network_diagram(sizes, activations, dropouts, feature_names), key="arch_diagram")
    config_panel.render_current()
