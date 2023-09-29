from autoannotator.config.feature_extraction import ConfigInsightface
from autoannotator.feature_extraction.core.feature_extractor import FaceFeatureExtractorConfig
from autoannotator.feature_extraction.faces.models.face_feature_extractor import FaceFeatureExtractor


class FaceFeatureExtractorInsightface(FaceFeatureExtractor):
    """ArcFace model, InsightFace project, IResNet100, Glint360k. 
    Based on ArcFace: Additive Angular Margin Loss for Deep Face Recognition,
    https://arxiv.org/abs/1801.07698
    Ptretrained model is taken from https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
    """
    def __init__(self, config: FaceFeatureExtractorConfig=ConfigInsightface()) -> None:
        super().__init__(config)