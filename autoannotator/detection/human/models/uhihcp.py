import numpy as np
from typing import List, Tuple, Union, Dict

from autoannotator.detection.core.base_detector import BaseDetector
from autoannotator.types.base import Detection, ImageColorFormat
from autoannotator.types.custom_typing import Tuple2f, Tuple2i, Tuple3f, Tuple3i
from autoannotator.utils.image_preprocessing import (
    resize_image,
    np2onnx,
    normalize_image,
)
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import get_project_root

__all__ = ["UniHCPHumanDetectionConfig", "UniHCPHuman"]

_ROOT = get_project_root()


class UniHCPHumanDetectionConfig(DetectionConfig):
    """UniHCP human detector config"""

    # Please request the weights according to the https://github.com/OpenGVLab/UniHCP#pretrained-models
    weights: str = f"{_ROOT}/weights/detection/human/UniHCP_peddet.onnx"
    url: str = None
    conf_thresh: float = 0.4
    nms_thresh: float = None
    input_size: Tuple2i = (800, 1200)
    mean: Tuple3f | Tuple3i = None
    std: Tuple3f | Tuple3i = None


class UniHCPHuman(BaseDetector):
    """
    UniHCP human detection onnx interface. Refer to: https://github.com/OpenGVLab/UniHCP.
    Supported models: UniHCP peddet
    CrowdHuman metrics: AP@50 = 92.5, MR = 41.6

    Arguments:
        config (DetectionConfig): detector config
    """

    def __init__(self, config: DetectionConfig = UniHCPHumanDetectionConfig()):
        super(UniHCPHuman, self).__init__(config)

        self.config = config
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_size = self.config.input_size

    @property
    def name(self):
        return "UniHCP_Human_Detection"

    def _predict(self, img: np.ndarray) -> List[Detection]:
        img0_shape = (img.shape[0], img.shape[1])
        x, shift, scale = self._preprocess(img)
        raw_out = self._forward(x)
        out = self._postprocess(raw_out, shift, scale, img0_shape)
        return out

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple2f, Tuple2f]:
        img, shift, scale = resize_image(
            img, size=self.input_size, keep_ratio=True, position="center", value=127
        )

        if self.config.mean and self.config.std:
            img = normalize_image(img, mean=self.config.mean, std=self.config.std)

        img = np2onnx(img, color_mode=ImageColorFormat.RGB)
        return img, shift, scale

    def _forward(self, inputs):
        out = self.session.run(None, {self.input_name: inputs})
        return out

    def _postprocess(
        self, raw_out: np.ndarray, pad: Tuple2f, scale: Tuple2f, img0_shape: Tuple2i
    ) -> List[Detection]:
        h0, w0 = img0_shape
        results = []
        scores = raw_out[0][0]
        labels = raw_out[1][0]
        boxes = raw_out[2][0]

        scale_fct = np.array(
            [
                self.config.input_size[1],
                self.config.input_size[0],
                self.config.input_size[1],
                self.config.input_size[0],
            ]
        )

        for score, label, box in zip(scores, labels, boxes):
            box = box * scale_fct
            box = box.astype(np.float32).tolist()
            if score > self.config.conf_thresh:
                pred = Detection(
                    cls_id=label,
                    score=score,
                    bbox=box,
                )
                pred.shift(x0=-pad[0], y0=-pad[1])
                pred.scale(sx=1 / scale[0], sy=1 / scale[1])
                pred.clip(0, 0, w0 - 1, h0 - 1)
                results.append(pred)
        return results
