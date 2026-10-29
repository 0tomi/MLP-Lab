"""Sidebar: rotating history of saved models and the theme hint."""

from __future__ import annotations

import streamlit as st

from lab import text as T
from lab.state import saved_runs
from lab.views.common import arch_label, key_metric, run_option_label


def render() -> None:
    with st.sidebar:
        _history()
        st.divider()
        st.caption(T.THEME_HINT)


def _history() -> None:
    st.subheader(T.HISTORY_TITLE, help=T.HISTORY_HELP)
    runs = saved_runs()
    if not runs:
        st.caption(T.HISTORY_EMPTY)
        return
    metas = {r.meta["id"]: r.meta for r in runs}
    if st.session_state.get("active_run_id") not in metas:
        st.session_state["active_run_id"] = runs[0].meta["id"]
    st.radio(
        T.HISTORY_TITLE,
        options=list(metas),
        key="active_run_id",
        format_func=lambda run_id: run_option_label(metas[run_id]),
        captions=[f"{arch_label(m)}  \n{key_metric(m)}" for m in metas.values()],
        label_visibility="collapsed",
    )
