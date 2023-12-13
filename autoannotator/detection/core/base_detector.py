import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from typing import List
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
        out = self._predict(img)
        return out

    def _init_session(self):
        """ Init onnx runtime session """
        if self.session is None:
            import onnxruntime
            if self.config.device == 'cpu':
                providers = ['CPUExecutionProvider']
            else:
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
    def _predict(self, img: np.ndarray) -> List[Detection]:
        raise NotImplementedError
