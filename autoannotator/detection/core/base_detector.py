import cv2
import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from typing import List, Tuple
from autoannotator.types.base import Detection
from autoannotator.config.detection import DetectionConfig
from autoannotator.utils.misc import attempt_download_onnx


class BaseDetector(ABC):
    """
    This is a base abstract detector class

    Arguments:
        config (DetectionConfig): detector config
    """

    def __init__(self, config: DetectionConfig):
        self.config = config

        self.session = None
        self._init_session()

    def __repr__(self):
        cls_name = self.__class__.__name__
        kwargs = ', '.join([f'{k}={v}' for k, v in self.config.dict().items()])
        return f'{cls_name}({kwargs})'

    def __call__(self, img: np.ndarray) -> List[Detection]:
        """
        Detect objects on image

        Arguments:
            img (np.ndarray): The input RGB image, HxWx3

        Returns:
            List[Detection]: List of detected objects
        """
        x, shift, scale = self._preprocess(img)
        raw_out = self._forward(x)
        out = self._postprocess(raw_out, shift, scale)
        return out

    def _init_session(self):
        """ Init onnx runtime session """
        if self.session is None:
            import onnxruntime
            if self.config.device == 'cpu':
                assert onnxruntime.get_device() == 'CPU'
                providers = ['CPUExecutionProvider']
            else:
                assert onnxruntime.get_device() == 'GPU'
                providers = ['CUDAExecutionProvider']

            assert self.config.weights is not None
            if not Path(self.config.weights).is_file():
                attempt_download_onnx(self.config.weights, self.config.url)
                # raise FileNotFoundError(f'No onnx weights found at {self.config.weights}')

            options = onnxruntime.SessionOptions()
            self.session = onnxruntime.InferenceSession(self.config.weights, options, providers=providers)

    @property
    @abstractmethod
    def name(self):
        """ Return detector name """
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        Prepare given image for inference. Apply geometrical transformations and normalization if needed

        Arguments:
            img (np.ndarray): The input RGB image, HxWx3

        Returns:
            np.ndarray: Preprocessed image
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, raw_out, shift=(0, 0), det_scale=1.0) -> List[Detection]:
        """
        Post-process raw onnx model output

        Arguments:
            raw_out: Onnx model output.
            shift (Tuple[float, float]): original to preprocessed image shift
            det_scale (float): original to preprocessed image scale
        Returns:
            List[Detection]: List of detected objects
        """
        raise NotImplementedError

    @abstractmethod
    def _forward(self, img: np.ndarray):
        """
        Onnx graph inference

        Arguments:
            img (np.ndarray): The input image.
        Returns:
            arbitrary object. Don't forget to implement your own post-processing script
        """
        raise NotImplementedError
