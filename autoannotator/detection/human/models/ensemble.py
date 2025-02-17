import numpy as np
from typing import List, Dict, Tuple

from autoannotator.detection.utils.test_time_augmentation import TestTimeAugmentationBase
from autoannotator.detection.utils.wbf import weighted_boxes_fusion
from autoannotator.types.base import Detection
from autoannotator.detection.core.base_detector import BaseDetector


class HumanDetEnsemble(object):
    """
    This is a human detection ensemble class

    Arguments:
       models (List[BaseDetector]): list of human detectors
    """

    def __init__(self, models: List[BaseDetector], match_iou_thr: float = 0.5, model_weights: List[float] = None, tta: List[TestTimeAugmentationBase] = None):
        """
        Constructor

        Arguments:
            models (List[BaseDetector]): All the models that should be used in the inference.
            match_iou_thr (float): IoU threshold to match Detections, default 0.5
            model_weights (List[float]): model weights that are used to merge predictions into one
        """
        super(HumanDetEnsemble, self).__init__()
        self.models = models
        self.match_iou_thr = match_iou_thr
        self.model_weights = model_weights
        
        self.tta = tta
        self.use_tta = self.tta and len(self.tta) > 0
        if self.use_tta:
            self.model_weights *= (len(self.tta) + 1)
        

    def __call__(self, img: np.ndarray) -> Tuple[List[Detection], List[dict], Dict[str, List[Detection]]]:
        """
        Run inference with the ensemble of models on a given image

        Arguments:
            img (np.ndarray): The input image.

        Returns:
            (List[Detection]): List of detected faces
        """

        results = {}
        for model in self.models:
            res = model(img)
            results[model.name] = res
            
        if self.use_tta:
            for tta in self.tta:
                augmented_img, metadata = tta.augment(img)
                for model in self.models:
                    res = model(augmented_img)
                    res = tta.rectify(res, metadata)
                    results[f"{model.name}_{tta.name}"] = res

        predictions, meta = self.reduce(results)

        return predictions, meta, results

    def reduce(self, results: Dict[str, List[Detection]]) -> Tuple[List[Detection], List[dict]]:
        """
        Reduces ensemble models predictions into single prediction with Weighted Boxes Fusion Algorithm

        Arguments:
            results (Dict[List[Detection]]): Dict of predicted faces {'model1': det_arr1, 'model2: det_arr2, ...}
        Returns:
            (List[Detection]): Reduced List of detected objects
        """
        boxes_list, scores_list, labels_list = [], [], []

        for key, model_preds in results.items():
            boxes, scores, labels = [], [], []
            for detection in model_preds:
                boxes.append(detection.bbox)
                scores.append(detection.score)
                labels.append(detection.cls_id)

            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        annotations = {'labels': labels_list, 'scores': scores_list, 'boxes': boxes_list}
        w_annotations = weighted_boxes_fusion(annotations, weights=self.model_weights, iou_thr=self.match_iou_thr)

        out = []
        meta = []
        for ann in w_annotations:
            out.append(Detection(
                cls_id=ann['label'],
                score=ann['score'],
                bbox=ann['bbox'].tolist(),
            ))
            meta.append(ann['meta'])
        return out, meta
