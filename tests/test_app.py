from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from lab import text as T
from lab.registry import list_runs

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(tmp_path):
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["models_dir"] = str(tmp_path)
    return at


def test_app_loads_without_exceptions(app):
    app.run()
    assert not app.exception
    assert [t.label for t in app.tabs][:5] == T.TABS


def test_training_xor_with_mixed_layers_saves_and_shows_the_model(app, tmp_path):
    app.run()
    for key, value in {
        "dataset_classification": "xor",
        "n_hidden": 2,
        "units_1": 8,
        "act_1": "tanh",
        "drop_1": 0.2,
        "units_2": 4,
        "act_2": "relu",
        "epochs": 40,
        "monitor_classification": "val_auc",
    }.items():
        app.session_state[key] = value
    app.run()

    next(b for b in app.button if b.label == T.TRAIN_BUTTON).click()
    app.run()

    assert not app.exception
    runs = list_runs(Path(tmp_path))
    assert len(runs) == 1
    meta = runs[0].meta
    assert meta["architecture"] == "2-8(tanh,d0.2)-4(relu)-1"
    assert meta["config"]["monitor"] == "val_auc"
    assert 1 <= meta["best_epoch"] <= meta["epochs_run"] <= 40
    assert set(meta["metrics"]) == {"train", "val", "test"}
    # Results table and Try it inputs are rendered for the new active model
    assert len(app.dataframe) >= 1
    assert any(n.key and n.key.startswith(f"try_{meta['id']}_") for n in app.number_input)


def test_full_training_without_early_stopping_and_reusing_a_saved_config(app, tmp_path):
    from lab.config_io import validate_config

    app.run()
    for key, value in {"dataset_classification": "moons", "n_hidden": 2, "act_2": "elu", "drop_2": 0.1,
                       "epochs": 15, "patience": 5, "early_stopping": False}.items():
        app.session_state[key] = value
    app.run()
    next(b for b in app.button if b.label == T.TRAIN_BUTTON).click()
    app.run()

    assert not app.exception
    meta = list_runs(Path(tmp_path))[0].meta
    assert meta["epochs_run"] == 15
    assert meta["config"]["early_stopping"] is False
    imported = validate_config(meta["config"])  # meta.json embeds the export schema
    assert imported.config.layers[1].activation == "elu"
    assert "separability" in meta

    app.session_state["n_hidden"] = 1
    app.run()
    next(b for b in app.button if b.label == T.CONFIG_USE).click()
    app.run()
    assert not app.exception
    assert app.session_state["n_hidden"] == 2
    assert app.session_state["drop_2"] == 0.1
