import numpy as np
from pathlib import Path
from autoannotator.feature_extraction.core.feature_extractor import BaseFeatureExtrator, FaceFeatureExtractorConfig
from autoannotator.utils.image_preprocessing import normalize_image, np2onnx
from autoannotator.utils.misc import attempt_download_onnx
from autoannotator.utils.onnx_model_handler import OnnxModelHandler

    
class FaceFeatureExtractor(BaseFeatureExtrator):
    """The base class for face feature extractors.
    The logic consists of 3 parts: preprocessing, inference, and postprocessing.
    It is expected that FeatureExtractorConfig contains URL to download ONNX weights if ONNX file is not available locally.
    """
    def __init__(self, config: FaceFeatureExtractorConfig) -> None:
        super().__init__(config)
        lib_root = Path(__file__).absolute().parent.parent.parent.parent
        self.onnx_path = Path(lib_root, self.config.onnx_path)
        attempt_download_onnx(self.onnx_path.as_posix(), self.config.url)
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
        preprocessed_image = normalize_image(preprocessed_image,
                                             self.config.normalize_mean,
                                             self.config.normalize_std)
        preprocessed_image = np2onnx(preprocessed_image, color_mode=self.config.color_format)
        return preprocessed_image
    
    def _forward(self, image: np.ndarray) -> np.ndarray:
        """Perform inference
        
        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Features (embedding, descriptor)
        """
        embedding = self.model(image)
        return embedding
    
    def _postprocess(self, tensor: np.ndarray) -> np.ndarray:
        """ Process the output of current model
        
        Args:
            image (np.ndarray): Features, output of the model

        Returns:
            np.ndarray: Preprocessed features
        
        """
        return tensor[0]
