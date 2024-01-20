import pytest
import numpy as np


from autoannotator.detection.human.models.uhihcp import (
    UniHCPHuman,
    UniHCPHumanDetectionConfig,
)
from autoannotator.utils.image_reader import ImageReader


def test_unihcp_human_detection():
    # Pass this test until we re-train the model.
    # Original authors of UniHCP require signing an agreement before using their weights.
    return True

    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = UniHCPHuman(UniHCPHumanDetectionConfig())
    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([450, 241, 617, 669])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )
