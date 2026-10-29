from lab.datasets import make_dataset, split_dataset
from lab.model import LayerSpec, TrainConfig
from lab.training import MONITOR_OPTIONS, train


def _split(key="blobs"):
    ds = make_dataset(key, n_samples=200, noise=0.1, seed=0)
    return split_dataset(ds, 0.15, 0.15, seed=0, standardize=True)


def test_training_reports_each_epoch_and_loss_decreases_on_easy_data():
    seen = []
    config = TrainConfig(
        task="classification",
        layers=(LayerSpec(4, "tanh"),),
        optimizer="adam",
        learning_rate=0.05,
        epochs=30,
        patience=30,
        seed=0,
    )

    result = train(config, _split(), on_epoch=lambda epoch, logs, model: seen.append((epoch, logs)))

    assert [e for e, _ in seen] == list(range(1, len(seen) + 1))
    losses = result.history["loss"]
    assert len(losses) == len(seen) == result.epochs_run
    assert losses[-1] < 0.5 * losses[0]
    assert result.history["val_loss"][-1] < result.history["val_loss"][0]
    assert {"loss", "val_loss", "accuracy", "val_accuracy"} <= set(result.history)


def test_early_stopping_respects_patience_and_restores_best_epoch():
    # learning rate 0: the monitored value never improves after epoch 1
    config = TrainConfig(
        task="classification",
        layers=(LayerSpec(3, "relu"),),
        optimizer="sgd",
        learning_rate=0.0,
        epochs=50,
        patience=3,
        monitor="val_loss",
        seed=0,
    )

    result = train(config, _split())

    assert result.best_epoch == 1
    assert result.epochs_run == 1 + 3
    assert result.stopped_early is True


def test_without_early_stopping_all_epochs_run_and_best_epoch_is_still_tracked():
    config = TrainConfig(
        task="classification",
        layers=(LayerSpec(3, "relu"),),
        optimizer="sgd",
        learning_rate=0.0,
        epochs=12,
        patience=3,
        early_stopping=False,
        seed=0,
    )

    result = train(config, _split())

    assert result.epochs_run == 12
    assert result.stopped_early is False
    assert result.best_epoch == 1  # nothing improves with learning rate 0


def test_monitor_options_cover_the_requested_metrics():
    assert set(MONITOR_OPTIONS["classification"]) >= {"val_loss", "val_accuracy", "val_auc", "val_precision", "val_recall"}
    assert MONITOR_OPTIONS["classification"]["val_loss"] == "min"
    assert MONITOR_OPTIONS["classification"]["val_auc"] == "max"
    assert MONITOR_OPTIONS["regression"] == {"val_loss": "min", "val_mae": "min"}


def test_regression_training_runs_with_dropout_and_mae_monitor():
    ds = make_dataset("sine_1d", n_samples=150, noise=0.1, seed=0)
    split = split_dataset(ds, 0.15, 0.15, seed=0, standardize=True)
    config = TrainConfig(
        task="regression",
        layers=(LayerSpec(16, "tanh", 0.2), LayerSpec(8, "relu")),
        learning_rate=0.01,
        epochs=5,
        monitor="val_mae",
        seed=0,
    )

    result = train(config, split)

    assert result.epochs_run == 5
    assert "val_mae" in result.history
