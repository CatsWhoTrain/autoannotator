from typing import List
from pydantic import BaseModel


class ConfigImageAlignmentBase(BaseModel):
    output_size: List[int] = [112, 112]
    ref_points: List[List[float]] = [
        [30.2946+8.0, 51.6963],
        [65.5318+8.0, 51.5014],
        [48.0252+8.0, 71.7366],
        [33.5493+8.0, 92.3655],
        [62.7299+8.0, 92.2041]
    ]