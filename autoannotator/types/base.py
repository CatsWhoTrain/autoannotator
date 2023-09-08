from pydantic import BaseModel
from typing import List


class Detection(BaseModel):
    cls_id: int
    score: float
    bbox: List[int]  # xyxy
