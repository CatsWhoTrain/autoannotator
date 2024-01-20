import pytest
import numpy as np

from autoannotator.detection.human.models.internimage import (
    InternImageHuman,
    InternImageDetectionConfig,
)
from autoannotator.utils.image_reader import ImageReader

def test_internimage_human_detection():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = InternImageHuman(InternImageDetectionConfig())
    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([855, 227, 989, 686])
    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )
