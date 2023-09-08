from autoannotator.config.feature_extraction import ConfigInsightface
from autoannotator.feature_extraction.core.feature_extractor import FeatureExtractorConfig
from autoannotator.feature_extraction.faces.models.face_feature_extractor import FaceFeatureExtractor


class FaceFeatureExtractorInsightface(FaceFeatureExtractor):
    def __init__(self, config: FeatureExtractorConfig=ConfigInsightface()) -> None:
        super().__init__(config)