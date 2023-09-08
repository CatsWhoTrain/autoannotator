
from abc import ABC, abstractmethod
from typing import List
from enum import Enum
import numpy as np
from pydantic import BaseModel

from autoannotator.types.base import Device, ImageColorFormat


class FeatureExtractorConfig(BaseModel):
    onnx_path: str
    name: str
    normalize_mean: List[float] = [0.5, 0.5, 0.5]
    normalize_std: List[float] = [0.5, 0.5, 0.5]
    color_format: ImageColorFormat
    device: str  # cpu, cuda, or rt


class BaseFeatureExtrator(ABC):
    def __init__(self, config: FeatureExtractorConfig) -> None:
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
                raise ValueError(f"Unknown devide type {self.config.device}")
    
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