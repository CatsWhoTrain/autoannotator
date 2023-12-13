import pytest

import numpy as np

from autoannotator.clustering.methods.dbscan import ClusteringDBSCAN


def test_dbscan_sklean():
    dbscan = ClusteringDBSCAN(type="sklearn", eps=0.01, min_samples=2)
    input_data = [
        np.array([1, 2]),
        np.array([1.1, 1.8]),
        np.array([-1, -2]),
        np.array([-1.1, -1.8]),
        np.array([101, 201]),
        np.array([1, -1]),
    ]
    labels = dbscan(input_data)
    assert np.sum(labels == [0, 0, 1, 1, 0, -1]) == len(labels)
