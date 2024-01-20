import numpy as np
from typing import Optional, List, Tuple, Dict

from autoannotator.types.custom_typing import Tuple3f, Tuple3i, Tuple2i, Tuple2f
from autoannotator.detection.core.base_detector import BaseDetector
from autoannotator.types.base import ImageColorFormat, Detection
from autoannotator.utils.image_preprocessing import resize_image, normalize_image, np2onnx
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import get_project_root


__all__ = ['RTDETRDetectionConfig', 'RTDETR']


_ROOT = get_project_root()


class RTDETRDetectionConfig(DetectionConfig):
    """ RTDETR R101 object detector config """
    weights: str = f'{_ROOT}/weights/detection/human/rtdetr_r101_ch_wp_640х640.onnx'
    url: Optional[str] = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/rtdetr_r101_ch_wp_640.640.onnx"
    conf_thresh: float = 0.5
    nms_thresh: float = None
    input_size: Tuple2i = (640, 640)
    mean: Tuple3f | Tuple3i = (0., 0., 0.),
    std: Tuple3f | Tuple3i = (1., 1., 1.),


class RTDETR(BaseDetector):
    """
    RTDETR onnx interface. Refer to: https://github.com/lyuwenyu/RT-DETR
    Supported models: RTDETR_R101
    Custom rtdetr trained on CrowdHuman and WiderPerson datasets
    CrowdHuman metrics: AP@50 = 87.1
    WiderPerson metrics: AP@50 = 81.8


    Arguments:
        config (DetectionConfig): detector config
    """
    def __init__(self, config=RTDETRDetectionConfig()):
        super(RTDETR, self).__init__(config)

        self.color_mode = ImageColorFormat.RGB
        self.input_name = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def name(self):
        return 'RTDETR_R101'

    def _predict(self, img: np.ndarray) -> List[Detection]:
        img0_shape = (img.shape[0], img.shape[1])
        x, pad, scale = self._preprocess(img)
        raw_res = self._forward(x)
        results = self._postprocess(raw_res, pad, scale, img0_shape)
        return results

    def _preprocess(self, img: np.ndarray) -> Tuple[Dict, Tuple2f, Tuple2f]:
        img, pad, scale = resize_image(img, size=self.config.input_size, keep_ratio=True, position='center')
        img = normalize_image(img, mean=self.config.mean, std=self.config.std)

        img1_shape = img.shape[:-1]
        img = np2onnx(img, color_mode=self.color_mode)
        size = np.ascontiguousarray([img1_shape]).astype(np.int32)

        inputs = {
            self.input_name[0]: img,
            self.input_name[1]: size,
        }
        return inputs, pad, scale

    def _postprocess(self, raw_out: np.ndarray, pad: Tuple2f, scale: Tuple2f, img0_shape: Tuple2i) -> List[Detection]:
        h0, w0 = img0_shape
        labels, boxes, scores = raw_out
        bs, num_queries = labels.shape

        results = []
        img_ind = 0

        for i in range(num_queries):
            label = labels[img_ind][i]
            box = boxes[img_ind][i].astype(np.int32).tolist()
            score = scores[img_ind][i]

            if score > self.config.conf_thresh:
                pred = Detection(
                    cls_id=label,
                    score=score,
                    bbox=box,
                )
                pred.shift(x0=-pad[0], y0=-pad[1])
                pred.scale(sx=1 / scale[0], sy=1 / scale[1])
                pred.clip(0, 0, w0-1, h0-1)
                results.append(pred)
        return results

    def _forward(self, inputs):
        out = self.session.run(self.output_names, inputs)
        return out
