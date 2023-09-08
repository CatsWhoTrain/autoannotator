from enum import Enum


class ImageColorFormat(Enum):
    BGR = 1
    RGB = 2
    
    
class Device(Enum):
    CPU = 1
    CUDA = 2
    RT = 3