from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class BaseClustering(ABC):
    """ Base class for clustering approaches
    """
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, vectors: List[np.ndarray]) -> List[Any]:
        pass