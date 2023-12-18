from pydantic import BaseModel
from typing import List
from enum import Enum
import numpy as np


class Detection(BaseModel):
    cls_id: int
    score: float
    bbox: List[float]  # xyxy

    def shift(self, x0=0, y0=0):
        x1, y1, x2, y2 = self.bbox
        self.bbox = [
            x1 + x0,
            y1 + y0,
            x2 + x0,
            y2 + y0,
        ]

    def scale(self, sx=1.0, sy=1.0):
        x1, y1, x2, y2 = self.bbox
        self.bbox = [
            x1 * sx,
            y1 * sy,
            x2 * sx,
            y2 * sy,
        ]

    def clip(self, x1, y1, x2, y2):
        box = np.array(self.bbox)
        box[0::2] = np.clip(box[0::2], x1, x2)
        box[1::2] = np.clip(box[1::2], y1, y2)
        self.bbox = box.tolist()


class ImageColorFormat(Enum):
    BGR = 1
    RGB = 2
    
    
class Device(Enum):
    CPU = 1
    CUDA = 2
    RT = 3

