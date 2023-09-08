
from typing import List

from pydantic import BaseModel
from autoannotator.types.base import ImageColorFormat


class FeatureExtractorConfig(BaseModel):
    onnx_path: str
    url: str
    name: str
    normalize_mean: List[float] = [0.5, 0.5, 0.5]
    normalize_std: List[float] = [0.5, 0.5, 0.5]
    color_format: ImageColorFormat
    device: str  # cpu, cuda, or rt


class ConfigAdaface(FeatureExtractorConfig):
    onnx_path: str = "weights/feature_extraction/faces/adaface_ir101_ms1mv3.onnx"
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/adaface_ir101_ms1mv3.onnx"
    color_format: ImageColorFormat = ImageColorFormat.RGB
    name: str = "AdaFace"
    device: str = "cpu"
    

class ConfigInsightface(FeatureExtractorConfig):
    onnx_path: str = "weights/feature_extraction/faces/if_iresnet100_arc112.onnx"
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/if_iresnet100_arc112.onnx"
    color_format: ImageColorFormat = ImageColorFormat.RGB
    name: str = "InsightFace"
    device: str = "cpu"

