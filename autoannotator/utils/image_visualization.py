from typing import List

import numpy as np
from PIL import Image, ImageDraw

from autoannotator.types.base import Detection


def draw_detections(
    img: np.ndarray,
    detections: List[Detection],
    color="green",
    width=5,
    filename="out.jpg",
):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for det in detections:
        draw.rectangle(
            ((det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3])),
            outline=color,
            width=width,
        )
    img.save(filename, "JPEG")
