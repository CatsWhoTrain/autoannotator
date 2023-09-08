import cv2

import numpy as np
from typing import List, Tuple, Union

from autoannotator.detection.core.base_detector import BaseDetector
from autoannotator.types.base import ImageColorFormat
from autoannotator.types.faces import Face
from autoannotator.utils.image_preprocessing import resize_image, np2onnx, normalize_image
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import get_project_root

__all__ = ['YOLOv7DetectionConfig', 'YOLOv7']


_ROOT = get_project_root()


class YOLOv7DetectionConfig(DetectionConfig):
    """ YOLOv7w6-Face object detector config """
    weights: str = f'{_ROOT}/weights/detection/faces/yolov7-w6-face.onnx'
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/yolov7-w6-face.onnx"
    conf_thresh: float = 0.4
    nms_thresh: float = 0.5
    input_size: Tuple[int, int] = (640, 640)
    mean: Union[Tuple[int, int, int], Tuple[float, float, float]] = (0., 0., 0.)
    std: Union[Tuple[int, int, int], Tuple[float, float, float]] = (1.0, 1.0, 1.0)


class YOLOv7(BaseDetector):
    """
    YOLOv7Face onnx interface. Refer to: https://github.com/derronqi/yolov7-face.
    The onnx model includes nms and detect layer. Supported models: YOLOv7w6

    Arguments:
        config (DetectionConfig): detector config
    """
    def __init__(self, config: DetectionConfig = YOLOv7DetectionConfig()):
        super(YOLOv7, self).__init__(config)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = self.config.input_size

    @property
    def name(self):
        return 'YOLOv7w6-Face'

    def _forward(self, blob):
        net_outs = self.session.run([], {self.input_name: blob})
        return net_outs

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        img, shift, scale = resize_image(img, size=self.input_size, keep_ratio=True, position='center')

        img = normalize_image(img, mean=self.config.mean, std=self.config.std)

        img = np2onnx(img, color_mode=ImageColorFormat.RGB)

        return img, shift, scale

    def _postprocess(self, raw_out, shift=(0, 0), det_scale=1.0) -> List[Face]:
        res = []
        raw_out = raw_out[0]

        for item in raw_out:
            bbs, score, cls_id, kps = item[0:4], item[4], item[5], item[6:]

            bbs[[0, 2]] -= shift[0]
            bbs[[1, 3]] -= shift[1]
            bbs = bbs / det_scale

            kps = kps.reshape((-1, 3))

            kps[:, 0] -= shift[0]
            kps[:, 1] -= shift[1]
            kps /= det_scale

            res.append(Face(
                cls_id=0,   # todo:
                score=score,
                bbox=bbs.tolist(),  # xyxy
                landmarks=kps.tolist()  # 5 x [x, y, conf]
            ))
        return res
