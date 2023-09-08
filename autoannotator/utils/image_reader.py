from enum import Enum
from pathlib import Path
import numpy as np
import cv2
from autoannotator.types.base import ImageColorFormat


class ImageReader:
    def __init__(self, color_format: ImageColorFormat=ImageColorFormat.RGB):
       self.color_format = color_format
       
    def __call__(self, path: str | Path) -> np.ndarray:
        return self.read_image(path)

    @staticmethod
    def adjust_suffix(path: Path) -> Path:
        if path.with_suffix('.png').is_file():
            return path.with_suffix('.png')
        elif path.with_suffix('.jpeg').is_file():
            return path.with_suffix('.jpeg') 
        else:
            raise Exception(f"Could not find correct extension for {path}")
            
    def read_image(self, path: str | Path) -> np.ndarray:
        path = Path(path)
        if not path.is_file():
            path = self.adjust_suffix(path)
        
        image = cv2.imread(path.as_posix(), flags=cv2.IMREAD_COLOR)
        if self.color_format is ImageColorFormat.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        assert image is not None, f"Could not read {path}"
        return image