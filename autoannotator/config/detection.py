from typing import Optional, Tuple, Union
from pydantic import BaseModel


class DetectionConfig(BaseModel):
    """ Base object detector inference config """
    weights: str
    conf_thresh: float = 0.5
    nms_thresh: float = 0.5
    device: str = 'cpu'
    # tiles: Optional[Tuple[int, int, float]]     # todo: Not used yet
    input_size: Tuple[int, int]
    mean: Union[Tuple[int, int, int], Tuple[float, float, float]]
    std: Union[Tuple[int, int, int], Tuple[float, float, float]]
