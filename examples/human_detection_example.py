from typing import List
import sys
from tqdm import tqdm
sys.path.insert(0, '/app')
from autoannotator.detection.utils.test_time_augmentation import TTAColorHistogramEqualization, TTAHorizontalFlip
from autoannotator.types.base import Detection
from autoannotator.utils.image_reader import ImageReader
from autoannotator.detection.human import HumanDetEnsemble
from autoannotator.detection.human import RTDETR, IterDETR, UniHCPHuman
from autoannotator.detection.human import RTDETRDetectionConfig, IterDetrDetectionConfig, UniHCPHumanDetectionConfig


def filter_predictions(results, meta_list, min_score=0.6, min_votes=3, agreement_thr=0.9):
    out_results = []
    kept_preds = 0
    for detection, meta in zip(results, meta_list):

        if detection.score < min_score:
            continue

        if len(meta['unique_models']) < min_votes:
            continue

        out_results.append(detection)
        kept_preds += 1

    agreement_score = kept_preds / len(results)

    should_validate = (agreement_score < agreement_thr)

    return out_results, should_validate


def auto_annotate_humans(img_files: List[str]) -> List[List[Detection]]:
    """
    Face recognition usage example

    Arguments:
        img_files: List of image paths to recognize

    Returns:
        (List[Face]) - List of recognized faces
    """
    # ini image file reader
    reader = ImageReader()

    # init human detector ensemble
    models = [RTDETR(), IterDETR(), UniHCPHuman()]
    hd_ensemble = HumanDetEnsemble(
        models=models,
        model_weights=[0.87, 0.941, 0.925],
        match_iou_thr=0.65,
    )
    results_list = []

    # start processing images
    for ind, img_path in enumerate(tqdm(img_files, desc='Annotating files')):
        img_meta = {'img_path': img_path}
        # read image
        img = reader(img_meta['img_path'])

        # detect faces with ensemble
        results, meta, all_preds = hd_ensemble(img)

        # filter predictions: each prediction should have aggregated score > 0.5 and predicted by at least 2 models
        results, keep_image = filter_predictions(results, meta, min_votes=2, min_score=0.5)

        results_list.append(results)

    return results_list


def auto_annotate_humans_tta(img_files: List[str]) -> List[List[Detection]]:
    """
    Face recognition usage example

    Arguments:
        img_files: List of image paths to recognize

    Returns:
        (List[Face]) - List of recognized faces
    """
    # ini image file reader
    reader = ImageReader()

    # init human detector ensemble
    models = [RTDETR(), IterDETR()]
    hd_ensemble = HumanDetEnsemble(
        models=models,
        model_weights=[0.87, 0.941],
        match_iou_thr=0.5,
        tta=[TTAColorHistogramEqualization(), TTAHorizontalFlip()]
    )
    results_list = []

    # start processing images
    for ind, img_path in enumerate(tqdm(img_files, desc='Annotating files')):
        img_meta = {'img_path': img_path}
        # read image
        img = reader(img_meta['img_path'])

        # detect faces with ensemble
        results, meta, all_preds = hd_ensemble(img)

        # filter predictions: each prediction should have aggregated score > 0.5 and predicted by at least 2 models
        results, keep_image = filter_predictions(results, meta, min_votes=2, min_score=0.5)

        results_list.append(results)

    return results_list