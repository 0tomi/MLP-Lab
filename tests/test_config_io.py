import json

import pytest

from lab.config import LayerSpec, TrainConfig
from lab.config_io import (
    SCHEMA_VERSION,
    ConfigError,
    DatasetSpec,
    build_diagnostic,
    config_to_dict,
    parse_config,
)
from lab.datasets import make_dataset, split_dataset

CONFIG = TrainConfig(
    task="classification",
    layers=(LayerSpec(8, "tanh", 0.2), LayerSpec(4, "relu")),
    optimizer="sgd",
    learning_rate=0.03,
    momentum=0.9,
    batch_size=16,
    epochs=250,
    l2=1e-4,
    monitor="val_auc",
    patience=25,
    early_stopping=False,
    seed=7,
)
DATASET = DatasetSpec(name="xor", n_samples=300, noise=0.15, seed=7, val=0.15, test=0.2, standardize=True)


def _exported() -> dict:
    return json.loads(json.dumps(config_to_dict(CONFIG, DATASET)))


def test_export_is_flat_and_self_describing():
    data = _exported()
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["output_activation"] == "sigmoid"
    assert data["loss"] == "binary_crossentropy"
    assert data["layers"][0] == {"units": 8, "activation": "tanh", "dropout": 0.2}
    assert data["dataset"]["split"] == {"train": 0.65, "validation": 0.15, "test": 0.2}
    assert {"optimizer", "learning_rate", "momentum", "batch_size", "epochs", "l2", "monitor", "patience"} <= set(data)
    assert data["early_stopping"] is False


def test_export_import_round_trip_gives_identical_config():
    imported = parse_config(json.dumps(_exported()))
    assert imported.config == CONFIG
    assert imported.dataset == DATASET
    assert imported.adjusted == []


def test_off_grid_values_snap_to_the_nearest_widget_option():
    data = _exported()
    data["learning_rate"] = 0.004
    data["layers"][0]["dropout"] = 0.33
    imported = parse_config(json.dumps(data))
    assert imported.config.learning_rate == 0.003
    assert imported.config.layers[0].dropout == pytest.approx(0.35)
    assert set(imported.adjusted) == {"learning_rate", "layers[1].dropout"}


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda d: d.pop("layers"), "missing"),
        (lambda d: d["dataset"].pop("seed"), "missing"),
        (lambda d: d.update(colour="blue"), "unknown"),
        (lambda d: d["layers"][0].update(bias=True), "unknown"),
        (lambda d: d.update(schema_version=99), "version"),
        (lambda d: d.update(task="multiclass"), "choice"),
        (lambda d: d["layers"][0].update(activation="softmax"), "choice"),
        (lambda d: d.update(monitor="val_mae"), "choice"),  # regression-only monitor on a classification task
        (lambda d: d.update(epochs="many"), "type"),
        (lambda d: d.update(early_stopping="yes"), "type"),
        (lambda d: d["layers"][1].update(dropout=0.95), "range"),
        (lambda d: d.update(layers=[{"units": 2, "activation": "relu", "dropout": 0}] * 5), "too_many_layers"),
    ],
)
def test_invalid_config_raises_validation_error(mutate, code):
    data = _exported()
    mutate(data)
    with pytest.raises(ConfigError) as err:
        parse_config(json.dumps(data))
    assert err.value.code == code


def test_layers_accept_up_to_512_units_and_reject_513():
    data = _exported()
    data["layers"][0]["units"] = 512
    assert parse_config(json.dumps(data)).config.layers[0].units == 512
    data["layers"][0]["units"] = 513
    with pytest.raises(ConfigError) as err:
        parse_config(json.dumps(data))
    assert err.value.code == "range"


def test_invalid_json_raises_validation_error():
    with pytest.raises(ConfigError) as err:
        parse_config("{not json")
    assert err.value.code == "invalid_json"


def test_diagnostic_export_contains_loss_history_metrics_and_shapes():
    ds = make_dataset("xor", 100, 0.1, 0)
    split = split_dataset(ds, 0.15, 0.15, 0, True)
    history = {"loss": [0.712345, 0.5], "val_loss": [0.70001, 0.55], "auc": [0.5, 0.8], "val_auc": [0.4, 0.7]}
    meta = {
        "id": "run1",
        "created": "2026-10-29T01:00:00",
        "config": config_to_dict(CONFIG, DATASET),
        "best_epoch": 2,
        "epochs_run": 2,
        "stopped_early": False,
        "diverged": False,
        "metrics": {"train": {"accuracy": 0.912345}, "val": {"accuracy": 0.9}, "test": {"accuracy": 0.85}},
        "history": history,
        "separability": {"linearly_separable": False, "linear_accuracy": 0.6},
    }

    diag = build_diagnostic(meta, split)

    assert diag["history"]["loss"] == [0.7123, 0.5]
    assert diag["history"]["val_loss"] == [0.7, 0.55]
    assert diag["history"]["val_auc"] == [0.4, 0.7]
    assert diag["config"] == meta["config"]
    assert diag["training"] == {
        "early_stopping": False, "best_epoch": 2, "epochs_run": 2, "stopped_early": False, "diverged": False,
    }
    assert diag["metrics"]["train"]["accuracy"] == 0.9123
    assert diag["separability"] == {"linearly_separable": False, "linear_accuracy": 0.6}
    assert diag["data"]["train_shape"] == [70, 2]
    assert diag["data"]["test_shape"] == [15, 2]


def test_every_validation_error_has_a_spanish_message():
    from lab import text as T
    from lab.views.config_panel import error_message

    samples = [
        ConfigError("invalid_json", detail="x"), ConfigError("not_object", key="", expected="object"),
        ConfigError("version", value=9, expected=1), ConfigError("missing", key="layers"),
        ConfigError("unknown", key="colour"), ConfigError("type", key="epochs", expected="integer"),
        ConfigError("choice", key="task", value="x", options="a, b"), ConfigError("range", key="l2", min=0, max=1),
        ConfigError("too_many_layers", max=4),
    ]
    assert {e.code for e in samples} == set(T.CONFIG_ERRORS)
    for err in samples:
        message = error_message(err)
        assert message.startswith("No se pudo importar") and "{" not in message
