
from typing import List

from pydantic import BaseModel
from autoannotator.types.base import ImageColorFormat


class FaceFeatureExtractorConfig(BaseModel):
    """ Configuration for the feature extractor
    
    Fields:
    onnx_path: Local path to the ONNX model
    url: Download url for the ONNX model. Required only if there is no file at onnx_path
    name: Model name. Should be unique
    normalize_mean: Mean for image normalization. Range: [0, 1] or [0, 255]
    normalize_std: Standard deviation for image normalization.
    color_format: Color format. RGB or BGR or ImageColorFormat class
    device: inference device. CPU, CUDA, or RT
    """
    onnx_path: str
    url: str
    name: str
    normalize_mean: List[float] = [0.5, 0.5, 0.5]
    normalize_std: List[float] = [0.5, 0.5, 0.5]
    color_format: ImageColorFormat
    device: str  # cpu, cuda, or rt


class ConfigAdaface(FaceFeatureExtractorConfig):
    onnx_path: str = "weights/feature_extraction/faces/adaface_ir101_ms1mv3.onnx"
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/adaface_ir101_ms1mv3.onnx"
    color_format: ImageColorFormat = ImageColorFormat.RGB
    name: str = "AdaFace"
    device: str = "cpu"
    

class ConfigInsightface(FaceFeatureExtractorConfig):
    onnx_path: str = "weights/feature_extraction/faces/if_iresnet100_arc112.onnx"
    url: str = "https://github.com/CatsWhoTrain/autoannotator/releases/download/0.0.1/if_iresnet100_arc112.onnx"
    color_format: ImageColorFormat = ImageColorFormat.RGB
    name: str = "InsightFace"
    device: str = "cpu"

