from typing import Any, Dict, List

import numpy as np
from autoannotator.feature_extraction.core.feature_extractor import BaseFeatureExtrator


class FaceFeatureExtractionEnsemle:
    def __init__(self, models: List[BaseFeatureExtrator], reduce: str="concat") -> None:
        assert models is not None and len(models) > 0
        self.models = models
        self.__check_model_names_unique()
        self.reduce_type = reduce
        
    def __check_model_names_unique(self):
        known_names = set()
        for model in self.models:
            name = model.config.name
            if name in known_names:
                raise Exception(f"Model with name {name} already exists in the face descriptor extraction ensemble")
            known_names.add(name)
        
    def __call__(self, image: np.ndarray) -> Any:
        embeddings = {}
        for model in self.models:
            embeddings[model.config.name] = model(image)
        return self.reduce(embeddings)
        
    def reduce(self, embeddings: Dict[str, np.ndarray]):
        match self.reduce_type:
            case "concat":
                result_embedding = np.concatenate(list(embeddings.values()))
                return result_embedding
            case _:
                raise ValueError(f"Unknown reduce type {self.reduce}")