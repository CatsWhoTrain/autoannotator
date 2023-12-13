import pytest
import numpy as np

from autoannotator.detection.human.models.uhihcp import (
    UniHCPHuman,
    UniHCPHumanDetectionConfig,
)
from autoannotator.utils.image_reader import ImageReader


def test_face_detection_ensemble():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = UniHCPHuman(UniHCPHumanDetectionConfig())
    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([445, 242, 619, 668])
    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )
