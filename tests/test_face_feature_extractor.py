from pathlib import Path
import numpy as np
import pytest

from autoannotator.feature_extraction.faces.models.model_adaface import (
    ConfigAdaface,
    FaceFeatureExtractorAdaface,
)
from autoannotator.utils.image_alignment import ImageAlignmentSimilarityTransform
from autoannotator.feature_extraction.faces.models.ensemble import (
    FaceFeatureExtractionEnsemle,
)
from autoannotator.feature_extraction.faces.models.model_insightface import (
    FaceFeatureExtractorInsightface,
)
from autoannotator.utils.image_reader import ImageReader


def test_model_adaface():
    model_adaface = FaceFeatureExtractorAdaface()

    reader = ImageReader()
    input_img = reader("assets/images/ms_01.jpg")
    keypoints = [
        [340, 574, 1],
        [478, 503, 1],
        [403, 610, 1],
        [409, 716, 1],
        [527, 657, 1],
    ]

    regressor = ImageAlignmentSimilarityTransform()
    aligned_img = regressor(input_img, keypoints)

    embedding = model_adaface(aligned_img)
    ground_truth = np.load("assets/binaries/ms_01_embedding_adaface.npy")
    np.testing.assert_allclose(embedding, ground_truth, rtol=1e-03, atol=1e-05)


def test_model_insightace():
    model_insightface = FaceFeatureExtractorInsightface()

    reader = ImageReader()
    input_img = reader("assets/images/ms_01.jpg")
    keypoints = [
        [340, 574, 1],
        [478, 503, 1],
        [403, 610, 1],
        [409, 716, 1],
        [527, 657, 1],
    ]

    regressor = ImageAlignmentSimilarityTransform()
    aligned_img = regressor(input_img, keypoints)

    embedding = model_insightface(aligned_img)
    ground_truth = np.load("assets/binaries/ms_01_embedding_insightface.npy")
    np.testing.assert_allclose(embedding, ground_truth, rtol=1e-03, atol=1e-05)


def test_feature_extractor_ensemble():
    adaface_config = ConfigAdaface()
    adaface_config.device = "cuda"
    model_adaface = FaceFeatureExtractorAdaface()
    model_insightface = FaceFeatureExtractorInsightface()
    ensemble = FaceFeatureExtractionEnsemle(models=[model_adaface, model_insightface])
    reader = ImageReader()
    input_img = reader("assets/images/ms_01.jpg")
    keypoints = [
        [340, 574, 1],
        [478, 503, 1],
        [403, 610, 1],
        [409, 716, 1],
        [527, 657, 1],
    ]

    regressor = ImageAlignmentSimilarityTransform()
    aligned_img = regressor(input_img, keypoints)

    embedding = ensemble(aligned_img)
    ground_truth = np.load("assets/binaries/ms_01_embedding_ensemble.npy")
    np.testing.assert_allclose(embedding, ground_truth, rtol=1e-03, atol=1e-05)
