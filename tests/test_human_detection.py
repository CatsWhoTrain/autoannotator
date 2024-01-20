import pytest
import numpy as np

from autoannotator.detection.human import HumanDetEnsemble
from autoannotator.detection.human import UniHCPHuman, InternImageHuman, IterDETR, RTDETR
from autoannotator.utils.image_reader import ImageReader


def test_model_unihcp():
    # Pass this test until we re-train the model.
    # Original authors of UniHCP require signing an agreement before using their weights.
    return True
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = UniHCPHuman()

    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([450, 241, 617, 669])
    
    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_model_iterdetr():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = IterDETR()

    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([448, 241, 616, 671])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_model_rtdetr():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    model = RTDETR()

    img = reader(img_file)
    detections = model(img)

    expected_bbox = np.array([449, 240, 616, 670])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )


def test_human_detection_ensemble():
    img_file = "assets/images/people_fullbody_gen_1.jpg"
    reader = ImageReader()

    # TODO: Restore this test when we re-train UniHCP
    # models = [UniHCPHuman(), IterDETR(), RTDETR()]
    # hd_ensemble = HumanDetEnsemble(models=models, model_weights=[0.87, 0.941, 0.87], match_iou_thr=0.5)
    
    models = [IterDETR(), RTDETR()]
    hd_ensemble = HumanDetEnsemble(models=models, model_weights=[0.941, 0.87], match_iou_thr=0.5)

    img = reader(img_file)
    detections, meta, _ = hd_ensemble(img)

    expected_bbox = np.array([448, 240, 616, 670])

    np.testing.assert_allclose(
        expected_bbox, np.array(detections[0].bbox).astype(np.int32)
    )
