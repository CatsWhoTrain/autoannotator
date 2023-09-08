from pydantic import BaseModel
from typing import List
from enum import Enum


class Detection(BaseModel):
    cls_id: int
    score: float
    bbox: List[int]  # xyxy

      
class ImageColorFormat(Enum):
    BGR = 1
    RGB = 2
    
    
class Device(Enum):
    CPU = 1
    CUDA = 2
    RT = 3

