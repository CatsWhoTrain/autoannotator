from pydantic import BaseModel
from typing import List, Tuple
from .base import Detection


class Face(Detection):
    landmarks: List[Tuple[float, float, float]]    # 5 x [x, y, conf]
