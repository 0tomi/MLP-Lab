"""PCA projection of standardized features, for plotting high-dimensional data."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_projection(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Projection (n, k) and the explained variance ratio of each component (non-increasing, sum <= 1)."""
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    return Z, pca.explained_variance_ratio_
