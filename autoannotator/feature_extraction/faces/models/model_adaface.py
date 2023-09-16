from autoannotator.config.feature_extraction import ConfigAdaface
from autoannotator.feature_extraction.core.feature_extractor import FaceFeatureExtractorConfig
from autoannotator.feature_extraction.faces.models.face_feature_extractor import FaceFeatureExtractor
   

class FaceFeatureExtractorAdaface(FaceFeatureExtractor):
    """AdaFace model, ir101 trained on ms1mv3. 
    Implelentation is based on AdaFace: Quality Adaptive Margin for Face Recognition,
    https://arxiv.org/abs/2204.00964
    Ptretrained model is taken from https://github.com/mk-minchul/AdaFace
    """
    def __init__(self, config: FaceFeatureExtractorConfig=ConfigAdaface()) -> None:
        super().__init__(config)