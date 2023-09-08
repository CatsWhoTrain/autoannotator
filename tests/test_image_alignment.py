import pytest


from pathlib import Path
import sys
import numpy as np

# sys.path.append(Path(__file__).absolute().parent.parent.as_posix())
from autoannotator.config.image_alignment import ConfigImageAlignmentBase
from autoannotator.utils.image_alignment import ImageAlignmentRegression
from autoannotator.utils.image_reader import ImageReader

def test_image_alignment():
    reader = ImageReader()
    input_img = reader("assets/images/ms_01.jpg")
    keypoints = [[340, 574, 1],
                [478, 503, 1],
                [403, 610, 1],
                [409, 716, 1],
                [527, 657, 1]]

    config = ConfigImageAlignmentBase()
    regressor = ImageAlignmentRegression(config)
    aligned_img = regressor(input_img, keypoints)
    ground_truth = np.load("assets/binaries/ms_01_aligned.npy")
    assert np.allclose(aligned_img, ground_truth)