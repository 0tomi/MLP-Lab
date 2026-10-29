import numpy as np

from lab.datasets import make_dataset
from lab.projection import pca_projection


def test_pca_projection_shape_and_explained_variance():
    ds = make_dataset("breast_cancer", n_samples=0, noise=0, seed=0)

    Z, ratios = pca_projection(ds.X, n_components=2)

    assert Z.shape == (len(ds.X), 2)
    assert ratios.shape == (2,)
    assert np.all(np.diff(ratios) <= 0)
    assert 0 < ratios.sum() <= 1
    assert ratios[0] > 0.4  # breast cancer: PC1 holds ~44% of the standardized variance


def test_pca_projection_single_component():
    ds = make_dataset("diabetes", n_samples=0, noise=0, seed=0)
    Z, ratios = pca_projection(ds.X, n_components=1)
    assert Z.shape == (len(ds.X), 1)
    assert 0 < ratios[0] <= 1
