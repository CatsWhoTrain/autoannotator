from autoannotator.config.feature_extraction import ConfigAdaface
from autoannotator.feature_extraction.core.feature_extractor import FeatureExtractorConfig
from autoannotator.feature_extraction.faces.models.face_feature_extractor import FaceFeatureExtractor
   

class FaceFeatureExtractorAdaface(FaceFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig=ConfigAdaface()) -> None:
        super().__init__(config)