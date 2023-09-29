import pytest
import numpy as np

import sys
sys.path.append("E:/ИТМО/Code/autoannotator/")

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
    return None
    
    expected_bbox = np.array([255.34286499023438, 314.4675598144531, 639.2444458007812, 797.302734375])
    expected_landmarks = np.array([[331.6590881347656, 568.7437133789062, 1.5600734949111938, 478.98736572265625, 502.46636962890625, 1.5603388547897339, 414.7174987792969, 610.7625732421875, 1.5614584684371948, 406.1929626464844, 710.7393798828125, 1.5593634843826294, 527.5408325195312, 654.9186401367188, 1.5474867820739746]])
    np.allclose(expected_bbox, faces[0].bbox)
    np.allclose(expected_landmarks, np.array(faces[0].landmarks).reshape(15))

test_face_detection_ensemble()