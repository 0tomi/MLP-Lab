import numpy as np

from lab.datasets import make_dataset, split_dataset
from lab.model import LayerSpec, TrainConfig, build_model
from lab.registry import list_runs, load_run, save_run


def _fixture():
    ds = make_dataset("xor", n_samples=80, noise=0.1, seed=0)
    split = split_dataset(ds, 0.15, 0.15, seed=0, standardize=True)
    config = TrainConfig(task="classification", layers=(LayerSpec(4, "tanh", 0.2),), seed=0)
    model = build_model(2, config)
    return split, config, model


def test_only_the_newest_five_runs_are_kept(tmp_path):
    split, config, model = _fixture()

    saved = [save_run(model, split, {"run": i, "config": config.to_dict()}, root=tmp_path) for i in range(7)]

    runs = list_runs(tmp_path)
    assert len(runs) == 5
    assert [r.meta["run"] for r in runs] == [6, 5, 4, 3, 2]  # newest first
    assert {r.path for r in runs} == set(saved[2:])
    assert not saved[0].exists() and not saved[1].exists()
    assert sorted(p.name for p in saved[-1].iterdir()) == ["data.npz", "meta.json", "model.keras", "scaler.joblib"]


def test_load_round_trip_gives_identical_predictions(tmp_path):
    split, config, model = _fixture()
    path = save_run(model, split, {"config": config.to_dict()}, root=tmp_path)

    loaded = load_run(path)

    expected = model.predict(split.X_test, verbose=0)
    np.testing.assert_array_equal(loaded.model.predict(loaded.split.X_test, verbose=0), expected)
    np.testing.assert_array_equal(loaded.split.X_train_raw, split.X_train_raw)
    np.testing.assert_array_equal(loaded.split.y_test, split.y_test)
    assert loaded.meta["config"] == config.to_dict()
    assert TrainConfig.from_dict(loaded.meta["config"]) == config
