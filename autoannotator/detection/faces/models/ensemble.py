import numpy as np
from typing import List, Dict

from autoannotator.detection.utils.wbf import weighted_boxes_fusion
from autoannotator.types.faces import Face
from autoannotator.detection.core.base_detector import BaseDetector


class FaceDetEnsemble:
    """
    This is a face detection ensemble class

    Arguments:
       models (List[BaseDetector]): list of face detectors
    """

    def __init__(self, models: List[BaseDetector], min_score: float = 0.4):
        """
        Constructor
        
        Arguments:
            models (List[BaseDetector]): All the models that should be used in the inference.
            min_score (float): Detections with at least min_score confidence will be added in the results 
        """
        super(FaceDetEnsemble, self).__init__()
        self.models = models
        self.min_score = min_score

    def __call__(self, img: np.ndarray) -> List[Face]:
        """
        Run inference with the ensemble of models on a given image

        Arguments:
            img (np.ndarray): The input image.
            
        Returns:
            (List[Face]): List of detected faces
        """
   
        results = {}
        for model in self.models:
            res = model(img)
            results[model.name] = res
        
        results = self.reduce(results, self.min_score)

        return results

    @staticmethod
    def reduce(results: Dict[str, List[Face]], min_score: float) -> List[Face]:
        """
        Reduces ensemble models predictions into single prediction with Weighted Boxes Fusion Algorithm

        Arguments:
            results (Dict[List[Face]]): Dict of predicted faces {'model1': faces_arr1, 'model2: faces_arr2, ...}
        Returns:
            (List[Face]): Reduced List of detected faces
        """
        boxes_list = []
        scores_list = []
        kp_list = []
        labels_list = []

        for key, model_preds in results.items():
            boxes = []
            scores = []
            kps = []
            labels = []
            for face in model_preds:
                boxes.append(face.bbox)
                scores.append(face.score)
                kps.append(np.array(face.landmarks).flatten().tolist())
                labels.append(face.cls_id)

            boxes_list.append(boxes)
            scores_list.append(scores)
            kp_list.append(kps)
            labels_list.append(labels)

        boxes, scores, kps, labels, _, _, _ = weighted_boxes_fusion(boxes_list, scores_list, kp_list, labels_list)

        out = []
        for box, score, kp, lbl in zip(boxes, scores, kps, labels):
            if score >= min_score:
                out.append(Face(
                    cls_id=lbl,
                    score=score,
                    bbox=box.tolist(),
                    landmarks=kp.reshape(-1, 3).tolist(),
                ))
        return out
