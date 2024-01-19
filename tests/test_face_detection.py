import pytest
import numpy as np

from autoannotator.detection.faces.models.ensemble import FaceDetEnsemble
from autoannotator.detection.faces.models.scrfd import SCRFD
from autoannotator.detection.faces.models.yolov7 import YOLOv7
from autoannotator.utils.image_reader import ImageReader


def test_model_scrfd():
    img_file = "assets/images/ms_01.jpg"
    reader = ImageReader()

    model = SCRFD()

    img = reader(img_file)
    faces = model(img)

    expected_bbox = np.array([254, 318, 637, 797])
    expected_landmarks = np.array([[330, 569, 1, 476, 499, 1, 415, 599, 1, 405, 712, 1, 527, 655, 1]])
    np.testing.assert_allclose(expected_bbox, np.array(faces[0].bbox).astype(np.int32))
    np.testing.assert_allclose(expected_landmarks, np.array(faces[0].landmarks).reshape(1, 15).astype(np.int32))


def test_model_yolov7():
    img_file = "assets/images/ms_01.jpg"
    reader = ImageReader()

    model = YOLOv7()

    img = reader(img_file)
    faces = model(img)

    expected_bbox = np.array([256, 311, 640, 796])
    expected_landmarks = np.array([[332, 568, 0, 481, 505, 0, 414, 619, 0, 407, 709, 0, 527, 654, 0]])
    np.testing.assert_allclose(expected_bbox, np.array(faces[0].bbox).astype(np.int32))
    np.testing.assert_allclose(expected_landmarks, np.array(faces[0].landmarks).reshape(1, 15).astype(np.int32))


def test_face_detection_ensemble():
    img_file = "assets/images/ms_01.jpg"
    reader = ImageReader()

    models = [SCRFD(), YOLOv7()]
    fd_ensemble = FaceDetEnsemble(models=models, model_weights=[0.9073, 0.9373], match_iou_thr=0.5)

    img = reader(img_file)
    faces, meta, _ = fd_ensemble(img)
    
    expected_bbox = np.array([255, 314, 639, 797])
    expected_landmarks = np.array([[331, 568, 0, 479, 502, 0, 414, 610, 0, 406, 710, 0, 527, 654, 0]])
    np.testing.assert_allclose(expected_bbox, np.array(faces[0].bbox).astype(np.int32))
    np.testing.assert_allclose(expected_landmarks, np.array(faces[0].landmarks).reshape(1, 15).astype(np.int32))
