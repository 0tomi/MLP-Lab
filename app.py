"""MLP Lab: Streamlit entry point. Run with `streamlit run app.py`."""

import streamlit as st

from lab import text as T
from lab.state import active_run, current_data, init_state
from lab.views import architecture, data, sidebar, train

st.set_page_config(page_title=T.APP_TITLE, page_icon=":material/hub:", layout="wide")
init_state()

st.title(T.APP_TITLE, anchor=False)
st.caption(T.APP_TAGLINE)
sidebar.render()

tab_data, tab_arch, tab_train, tab_results, tab_try = st.tabs(T.TABS, key="step")

with tab_data:
    ctx = data.render(current_data)
with tab_arch:
    architecture.render(ctx)
with tab_train:
    train.render(ctx)

# Results and Try it need the saved Keras model: import TensorFlow only now, after the first tabs are drawn.
from lab.views import results, try_it  # noqa: E402

run = active_run()
with tab_results:
    results.render(run)
with tab_try:
    try_it.render(run)
