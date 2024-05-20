from typing import List
from pydantic import BaseModel

from autoannotator.types.custom_typing import Tuple2i, Tuple3f, Tuple3i


class DetectionConfig(BaseModel):
    """ Base object detector inference config
    """
    weights: str
    conf_thresh: float = 0.5
    nms_thresh: float = 0.5  # used only in the models with NMS postprocessing
    device: str = 'cpu'
    # tiles: Optional[Tuple[int, int, float]]     # TODO: Not used yet
    input_size: Tuple2i  # height, width
    mean: Tuple3f | Tuple3i
    std: Tuple3f | Tuple3i
    onnx_custom_ops_libraries: List[str] = []  # ONNX custom ops librarires. See https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html
    
