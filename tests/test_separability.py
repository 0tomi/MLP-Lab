import pytest

from lab.datasets import make_dataset
from lab.separability import check_linear_separability, linear_regression_baseline


@pytest.mark.parametrize("key", ["blobs", "and", "or"])
def test_linearly_separable_datasets_are_detected(key):
    ds = make_dataset(key, n_samples=200, noise=0.1, seed=0)
    result = check_linear_separability(ds.X, ds.y)
    assert result.separable is True
    assert result.accuracy == 1.0


@pytest.mark.parametrize("key", ["xor", "circles"])
def test_non_separable_datasets_are_detected(key):
    ds = make_dataset(key, n_samples=200, noise=0.1, seed=0)
    result = check_linear_separability(ds.X, ds.y)
    assert result.separable is False
    assert result.accuracy < 1.0


def test_linear_regression_baseline_is_high_for_linear_data_and_low_for_sine():
    linear = make_dataset("linear_1d", n_samples=200, noise=0.1, seed=0)
    sine = make_dataset("sine_1d", n_samples=200, noise=0.05, seed=0)
    assert linear_regression_baseline(linear.X, linear.y) > 0.9
    assert linear_regression_baseline(sine.X, sine.y) < 0.8
