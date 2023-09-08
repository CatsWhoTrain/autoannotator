from typing import List
import numpy as np


def normalize_image(image: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    if image.dtype == np.uint8:
        image = image / 255
    image = (image - mean) * std
    image = np.asarray(image, dtype=np.float32)
    image = np.expand_dims(image, 0)
    image = image.transpose(0, 3, 1, 2)
    return image