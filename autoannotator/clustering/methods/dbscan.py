from typing import Any, List
import numpy as np
from sklearn.cluster import DBSCAN

from autoannotator.clustering.core.base_clustering import BaseClustering


class ClusteringDBSCAN(BaseClustering):
    """ Basic clustering approach using DBSCAN.
    Currently, only scikit-learn implementation is used
    """
    def __init__(self, type = "sklearn", eps=0.3, min_samples=2) -> None:
        super().__init__()
        self.type = type
        self.eps = eps
        self.min_samples = min_samples
        
    def __call__(self, vectors: List[np.ndarray]) -> List[int]:
        labels = None
        x = np.stack(vectors)
        if self.type == "sklearn":
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit(x)
            labels = db.labels_
        else:
            raise NotImplementedError
        return labels