from pydantic import BaseModel
from typing import List, Tuple
from .base import Detection


class Face(Detection):
    landmarks: List[Tuple[int, int, int]]    # 5 x [x, y, conf]
