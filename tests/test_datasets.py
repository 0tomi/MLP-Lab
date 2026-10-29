import numpy as np
import pandas as pd
import pytest

from lab.datasets import (
    CLASSIFICATION_DATASETS,
    REGRESSION_DATASETS,
    dataset_from_dataframe,
    make_dataset,
    split_dataset,
)

EXPECTED_FEATURES = {
    "blobs": 2,
    "xor": 2,
    "circles": 2,
    "moons": 2,
    "and": 2,
    "or": 2,
    "breast_cancer": 30,
    "linear_1d": 1,
    "sine_1d": 1,
    "diabetes": 10,
}
REAL_SIZES = {"breast_cancer": 569, "diabetes": 442}


@pytest.mark.parametrize("key", list(CLASSIFICATION_DATASETS) + list(REGRESSION_DATASETS))
def test_each_dataset_has_expected_shapes(key):
    ds = make_dataset(key, n_samples=200, noise=0.1, seed=0)

    n_expected = REAL_SIZES.get(key, 200)
    assert ds.X.shape == (n_expected, EXPECTED_FEATURES[key])
    assert ds.y.shape == (n_expected,)
    assert len(ds.feature_names) == EXPECTED_FEATURES[key]
    if ds.task == "classification":
        assert set(np.unique(ds.y)) == {0, 1}
        assert len(ds.class_names) == 2
    else:
        assert ds.y.dtype.kind == "f"


def test_dataset_registries_match_task():
    for key in CLASSIFICATION_DATASETS:
        assert make_dataset(key, 100, 0.1, 0).task == "classification"
    for key in REGRESSION_DATASETS:
        assert make_dataset(key, 100, 0.1, 0).task == "regression"


def test_split_default_proportions_and_scaler():
    ds = make_dataset("moons", n_samples=400, noise=0.2, seed=1)
    split = split_dataset(ds, val_size=0.15, test_size=0.15, seed=1, standardize=True)

    sizes = (len(split.y_train), len(split.y_val), len(split.y_test))
    assert sum(sizes) == 400
    assert sizes == (280, 60, 60)
    np.testing.assert_allclose(split.X_train.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(split.X_train_raw, split.x_scaler.inverse_transform(split.X_train), atol=1e-5)
    assert split.y_scaler is None  # classification targets are never scaled
    # stratified: both classes in every part
    for part in (split.y_train, split.y_val, split.y_test):
        assert set(np.unique(part)) == {0, 1}


def test_regression_split_standardizes_target_and_inverts_it():
    ds = make_dataset("diabetes", n_samples=0, noise=0, seed=0)
    split = split_dataset(ds, 0.15, 0.15, seed=0, standardize=True)

    assert abs(split.y_train_fit.mean()) < 1e-6
    np.testing.assert_allclose(split.inverse_y(split.y_train_fit), split.y_train, rtol=1e-5)


def test_split_without_standardize_keeps_raw_values():
    ds = make_dataset("blobs", n_samples=100, noise=0.1, seed=0)
    split = split_dataset(ds, 0.15, 0.15, seed=0, standardize=False)
    assert split.x_scaler is None
    np.testing.assert_allclose(split.X_train, split.X_train_raw)


def test_dataframe_dataset_keeps_numeric_features_and_encodes_binary_target():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, None],
            "b": [0.5, 0.1, 0.3, 0.2, 0.9],
            "text": ["x", "y", "z", "w", "v"],
            "label": ["no", "yes", "no", "yes", "no"],
        }
    )
    ds = dataset_from_dataframe(df, target="label", task="classification", name="upload.csv")

    assert ds.feature_names == ["a", "b"]
    assert ds.X.shape == (4, 2)  # row with NaN dropped
    assert ds.class_names == ("no", "yes")  # CSV labels are kept as-is
    np.testing.assert_array_equal(ds.y, [0, 1, 0, 1])


def test_dataframe_classification_rejects_non_binary_target():
    df = pd.DataFrame({"a": [1, 2, 3], "label": [0, 1, 2]})
    with pytest.raises(ValueError):
        dataset_from_dataframe(df, target="label", task="classification", name="x.csv")
