
from abc import ABC, abstractmethod
import numpy as np
from autoannotator.config.feature_extraction import FaceFeatureExtractorConfig
from autoannotator.types.base import Device, ImageColorFormat


class BaseFeatureExtrator(ABC):
    def __init__(self, config: FaceFeatureExtractorConfig) -> None:
        self.config = config
        self.__set_device()

    def __set_device(self):
        match self.config.device:
            case "cpu":
                self.device = Device.CPU
            case "cuda" | "gpu":
                self.device = Device.CUDA
            case "rt":
                self.device = Device.RT
            case _:
                raise ValueError(f"Unknown device type {self.config.device}")
    
    def __call__(self, image: np.ndarray):
        img = self._preprocess(image)
        prediction = self._forward(img)
        embedding = self._postprocess(prediction)
        return embedding

    @abstractmethod
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _forward(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def _postprocess(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError