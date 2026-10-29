"""Export / import of the configuration as JSON, and the diagnostic export of a trained model."""

from __future__ import annotations

import streamlit as st

from lab import text as T
from lab.config_io import ConfigError, build_diagnostic, config_to_dict, parse_config, to_json, validate_config
from lab.state import current_config, dataset_spec


def error_message(err: ConfigError) -> str:
    params = dict(err.params)
    if "expected" in params:
        params["expected"] = T.CONFIG_TYPES.get(params["expected"], params["expected"])
    return T.CONFIG_INVALID.format(reason=T.CONFIG_ERRORS[err.code].format(**params))


def _notice() -> None:
    notice = st.session_state.pop("import_notice", None)
    if notice is None:
        return
    st.success(T.CONFIG_IMPORTED)
    if notice:
        st.info(T.CONFIG_ADJUSTED.format(fields=", ".join(notice)))


def render_current() -> None:
    """Architecture step: export what the widgets describe; import a JSON into the widgets."""
    with st.expander(T.CONFIG_TITLE, icon=":material/data_object:"):
        st.caption(T.CONFIG_HELP)
        _notice()
        payload = to_json(config_to_dict(current_config(), dataset_spec()))
        left, right = st.columns([1.4, 1], gap="large")
        with left:
            st.download_button(T.CONFIG_DOWNLOAD, payload, file_name="mlp_config.json", mime="application/json",
                               icon=":material/download:")
            st.code(payload, language="json")
        with right:
            upload = st.file_uploader(T.CONFIG_UPLOAD, type="json", key="config_upload")
            if upload is None:
                st.session_state.pop("imported_file_id", None)
                return
            if st.session_state.get("imported_file_id") == upload.file_id:
                return  # already applied; avoid re-importing on every rerun
            try:
                imported = parse_config(upload.getvalue())
            except ConfigError as err:
                st.error(error_message(err))
                return
            st.session_state["imported_file_id"] = upload.file_id
            st.session_state["pending_import"] = imported
            st.session_state["import_notice"] = imported.adjusted
            st.rerun()


def _use_saved(config: dict) -> None:
    imported = validate_config(config)
    st.session_state["pending_import"] = imported
    st.session_state["import_notice"] = imported.adjusted


def render_run_exports(run) -> None:
    """Results step: the active model's config and its diagnostic."""
    meta = run.meta
    diagnostic = to_json(build_diagnostic(meta, run.split))
    config = to_json(meta["config"])
    a, b, c = st.columns(3)
    a.download_button(T.DIAG_DOWNLOAD, diagnostic, file_name=f"mlp_diagnostico_{meta['id']}.json",
                      mime="application/json", icon=":material/download:", help=T.DIAG_HELP, type="primary")
    b.download_button(T.CONFIG_DOWNLOAD, config, file_name=f"mlp_config_{meta['id']}.json", mime="application/json",
                      icon=":material/download:")
    try:
        validate_config(meta["config"])
        c.button(T.CONFIG_USE, on_click=_use_saved, args=(meta["config"],), help=T.CONFIG_USE_HELP,
                 icon=":material/upload:")
    except ConfigError:
        pass  # runs saved with an older schema can still be inspected
    with st.expander(T.DIAG_TITLE, icon=":material/data_object:"):
        st.code(diagnostic, language="json")
