import numpy as np
from typing import Tuple, Dict, List, Optional

from autoannotator.types.custom_typing import Tuple3f, Tuple3i, Tuple2f, Tuple2i
from autoannotator.detection.core.base_detector import BaseDetector
from autoannotator.types.base import ImageColorFormat, Detection
from autoannotator.utils.image_preprocessing import resize_image, normalize_image
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import get_project_root
from autoannotator.utils.algorithms import non_maximum_suppression


__all__ = ['IterDetrDetectionConfig', 'IterDETR']


_ROOT = get_project_root()


def _prepare_iter_detr_inputs(
        image: np.ndarray,
        img0_shape: Tuple2i,
        max_shape: Tuple2i,
        color_mode: ImageColorFormat
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    if color_mode == ImageColorFormat.BGR:
        image = image[:, :, ::-1]

    image = np.transpose(image, (2, 0, 1))

    image = np.ascontiguousarray(image).astype(np.float32)
    size = np.ascontiguousarray(img0_shape).astype(np.int32)

    out_image = np.zeros((3, *max_shape), dtype=np.float32)
    out_image[:, :h, :w] = image
    mask = np.ones(max_shape, dtype=np.int32)
    mask[:h, :w] = False

    out_image = np.expand_dims(out_image, axis=0)
    mask = np.expand_dims(mask, axis=0)
    size = np.expand_dims(size, axis=0)
    return out_image, mask, size


class IterDetrDetectionConfig(DetectionConfig):
    """ Iter Deformable DETR object detector config """
    weights: str = f'{_ROOT}/weights/detection/human/iter_detr_swinl_800x1300.onnx'
    url: Optional[str] = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/iter_detr_swinl_800x1300.onnx"
    conf_thresh: float = 0.4
    nms_thresh: float = 0.5     # todo: what threshold to choose?
    input_size: Tuple2i = (800, 1300)
    mean: Tuple3f | Tuple3i = (0.485, 0.456, 0.406),
    std: Tuple3f | Tuple3i = (0.229, 0.224, 0.225),


class IterDETR(BaseDetector):
    """
    IterDETR onnx interface. Refer to: https://github.com/zyayoung/Iter-Deformable-DETR.
    Supported models: Iter_DeformableDETR_Swin-L
    CrowdHuman metrics: AP@50 = 94.1, MR = 37.7, JI = 87.1

    Arguments:
        config (DetectionConfig): detector config
    """
    def __init__(self, config: IterDetrDetectionConfig = IterDetrDetectionConfig()):
        super(IterDETR, self).__init__(config)

        self.num_queries = 1000000    # IterDETR SwinL hardcode
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.color_format = ImageColorFormat.RGB

    @property
    def name(self):
        return 'IterDETR_SwinL'

    def _predict(self, img: np.ndarray) -> List[Detection]:
        img0_shape = (img.shape[0], img.shape[1])
        x, shift, scale = self._preprocess(img)
        raw_out = self._forward(x)
        out = self._postprocess(raw_out, shift, scale, img0_shape)
        return out

    def _preprocess(self, img: np.ndarray) -> Tuple[Dict, Tuple2f, Tuple2f]:
        img, pad, scale = resize_image(img, size=self.config.input_size, keep_ratio=True, position='center')
        img = normalize_image(img, mean=self.config.mean, std=self.config.std)
        img, mask, size = _prepare_iter_detr_inputs(
            img,
            img0_shape=self.config.input_size,
            max_shape=self.config.input_size,
            color_mode=self.color_format
        )

        inputs = {
            'images': img,
            'mask': mask,
            'orig_target_sizes': size,
        }
        return inputs, pad, scale

    def _postprocess(self, raw_out: np.ndarray, pad: Tuple2f, scale: Tuple2f, img0_shape: Tuple2i) -> List[Detection]:
        h0, w0 = img0_shape
        results = []
        for i in range(self.num_queries):
            score = raw_out[0][i]
            label = raw_out[1][i]
            box = raw_out[2][i].astype(np.int32).tolist()

            if score > self.config.conf_thresh:

                pred = Detection(
                    cls_id=0,
                    score=score,
                    bbox=box,
                )
                pred.shift(x0=-pad[0], y0=-pad[1])
                pred.scale(sx=1/scale[0], sy=1/scale[1])
                pred.clip(0, 0, w0 - 1, h0 - 1)
                results.append(pred)

        keep_indices = non_maximum_suppression(results, iou_thresh=self.config.nms_thresh)
        results = [results[i] for i in keep_indices]
        return results

    def _forward(self, inputs):
        out = self.session.run(self.output_names, inputs)
        return out
