
from abc import ABC, ABCMeta, abstractmethod
from typing import List
import cv2

import numpy as np
import skimage

from autoannotator.config.image_alignment import ConfigImageAlignmentSimilarityTransformBase


class ImageAlignmentBase(ABC):
    def __init__(self, config: ConfigImageAlignmentSimilarityTransformBase):
        self.config = config
    
    @abstractmethod
    def __call__(self, image, keypoints):
        pass


class ImageAlignmentSimilarityTransform(ImageAlignmentBase):
    def __init__(self, config: ConfigImageAlignmentSimilarityTransformBase=ConfigImageAlignmentSimilarityTransformBase()):
        super().__init__(config)
        self.ref_pts = np.array(self.config.ref_points, dtype=np.float32)

    def __call__(self, image: np.ndarray, keypoints: List[List[float]]) -> np.ndarray:
        """Rectifies and aligns image for feeding it to a face descriptor extractor

        Args:
            image (np.ndarray): original image in the HxWxC format
            keypoints (List[List[float]]): Facial keypoints in format [x, y, confidence] for each point

        Returns:
            np.ndarray: aligned and resized image
        """
        landmarks = np.array([x[:2] for x in keypoints], dtype=np.float32).reshape((5, 2))
        st = skimage.transform.SimilarityTransform()
        st.estimate(landmarks, self.ref_pts)
        aligned = cv2.warpAffine(image, st.params[0:2, :], self.config.output_size, borderValue=0.0)
        return aligned
