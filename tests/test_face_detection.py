import pytest
import numpy as np

from autoannotator.detection.faces.models.ensemble import FaceDetEnsemble
from autoannotator.detection.faces.models.scrfd import SCRFD
from autoannotator.detection.faces.models.yolov7 import YOLOv7
from autoannotator.utils.image_reader import ImageReader

def test_face_detection_ensemble():
    img_file = "assets/images/ms_01.jpg"
    reader = ImageReader()

    models = [SCRFD(), YOLOv7()]
    fd_ensemble = FaceDetEnsemble(models=models)
    

    img = reader(img_file)
    faces = fd_ensemble(img)
    
    expected_bbox = np.array([255, 314, 639, 797])
    expected_landmarks = np.array([[331, 568, 0, 478, 502, 0, 414, 610, 0, 406, 710, 0, 527, 654, 0]])
    np.testing.assert_allclose(expected_bbox, np.array(faces[0].bbox).astype(np.int32))
    np.testing.assert_allclose(expected_landmarks, np.array(faces[0].landmarks).reshape(1, 15).astype(np.int32))
