import numpy as np
from pathlib import Path
from autoannotator.feature_extraction.core.feature_extractor import BaseFeatureExtrator, FeatureExtractorConfig
from autoannotator.types.base import ImageColorFormat
from autoannotator.utils.image_preprocessing import normalize_image
from autoannotator.utils.onnx_model_handler import OnnxModelHandler

    
class FaceFeatureExtractor(BaseFeatureExtrator):
    def __init__(self, config: FeatureExtractorConfig) -> None:
        super().__init__(config)
        lib_root = Path(__file__).absolute().parent.parent.parent.parent
        self.onnx_path = Path(lib_root, self.config.onnx_path)
        assert self.onnx_path.is_file(), f"Could not find {self.onnx_path.as_posix()}"
        self.model = OnnxModelHandler(self.onnx_path.as_posix(), device=self.device)
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Arrange color channels and normalize the input image

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image
        """
        preprocessed_image = image.copy()
        if self.config.color_format == ImageColorFormat.BGR:
            preprocessed_image = preprocessed_image[...,::-1]
        preprocessed_image = normalize_image(preprocessed_image,
                                             self.config.normalize_mean,
                                             self.config.normalize_std)
        return preprocessed_image
    
    def _forward(self, image: np.ndarray) -> np.ndarray:
        embedding = self.model(image)
        return embedding
    
    def _postprocess(self, tensor: np.ndarray) -> np.ndarray:
        return tensor[0]
