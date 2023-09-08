from pathlib import Path
import sys
sys.path.append(Path(__file__).absolute().parent.parent.parent.parent.parent.as_posix())

from autoannotator.feature_extraction.core.feature_extractor import FeatureExtractorConfig
from autoannotator.feature_extraction.faces.models.face_feature_extractor import FaceFeatureExtractor
from autoannotator.types.base import ImageColorFormat


class ConfigAdaface(FeatureExtractorConfig):
    onnx_path: str = "weights/feature_extraction/faces/adaface_ir101_ms1mv3.onnx"
    color_format: ImageColorFormat = ImageColorFormat.RGB
    name: str = "AdaFace"
    device: str = "cpu"
    

class FaceFeatureExtractorAdaface(FaceFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig=ConfigAdaface()) -> None:
        super().__init__(config)