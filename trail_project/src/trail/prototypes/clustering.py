"""
Clustering algorithms for building behavior prototypes.

Supports K-means (default for MVP-1).
"""

from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans

from trail.utils.logging import get_logger

logger = get_logger(__name__)


class KMeansClustering:
    """K-means clustering for behavior prototype construction."""

    def __init__(self, n_clusters: int = 10, random_state: int = 42, n_init: int = 10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.model: Optional[KMeans] = None

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        logger.info(
            f"Fitting K-means: n_clusters={self.n_clusters}, "
            f"n_samples={X.shape[0]}, n_features={X.shape[1]}"
        )
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.model.fit(X)
        logger.info(f"K-means inertia: {self.model.inertia_:.2f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self.model.cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        return self.model.labels_
