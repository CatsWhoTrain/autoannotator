

from autoannotator.feature_extraction.core.feature_extractor import FeatureExtractorConfig
from autoannotator.feature_extraction.faces.models.face_feature_extractor import FaceFeatureExtractor
from autoannotator.types.base import ImageColorFormat


class ConfigInsightface(FeatureExtractorConfig):
    onnx_path: str = "weights/feature_extraction/faces/if_iresnet100_arc112.onnx"
    color_format: ImageColorFormat = ImageColorFormat.RGB
    name: str = "InsightFace"
    device: str = "cpu"
    

class FaceFeatureExtractorInsightface(FaceFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig=ConfigInsightface()) -> None:
        super().__init__(config)