# coding: utf-8
"""
Extended version of the Weighted-Boxes-Fusion https://github.com/ZFTurbo/Weighted-Boxes-Fusion

Changes:
* Added key points support (same behaviour as bboxes)
"""

import numpy as np
from loguru import logger


def prefilter_boxes(boxes, scores, key_points, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            logger.error(f'Length of boxes array and scores array mismatch: {len(boxes[t])} != {len(scores[t])}')
            exit()

        if len(boxes[t]) != len(labels[t]):
            logger.error(f'Length of boxes array and labels array mismatch: {len(boxes[t])} != {len(labels[t])}')
            exit()

        if len(boxes[t]) != len(key_points[t]):
            logger.error(f'Length of boxes array and keypoints array mismatch: {len(boxes[t])} != {len(key_points[t])}')
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])
            kp_part = key_points[t][j]

            # Box data checks
            if x2 < x1:
                # logger.warning('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                # logger.warning('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if any([x1 < 0, x2 < 0, y1 < 0, y2 < 0]):
                # logger.warning(f'Coordinates < 0 were found ({box_part}). Set them to 0.')
                x1, y1, x2, y2 = max(0., x1), max(0., y1), max(0., x2), max(0., y2)
            if (x2 - x1) * (y2 - y1) == 0.0:
                # logger.warning(f'Zero area box skipped: {box_part}.')
                continue

            # [label, score, weight, model index, x1, y1, x2, y2, *keypoints]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2, *kp_part]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x1, y1, x2, y2, *keypoints)
    """

    box = np.zeros(23, dtype=np.float32)
    conf = 0
    conf_list = []
    w = 0
    for b in boxes:
        box[4:8] += (b[1] * b[4:8])
        box[8:] += (b[1] * b[8:])  # key points
        conf += b[1]
        conf_list.append(b[1])
        w += b[2]
    box[0] = boxes[0][0]
    if conf_type in ('avg', 'box_and_model_avg', 'absent_model_aware_avg'):
        box[1] = conf / len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2] = w
    box[3] = -1  # model index field is retained for consistency but is not used.
    box[4:] /= conf
    return box


def find_matching_box_fast(boxes_list, new_box, match_iou):
    """
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.
    """

    def bb_iou_array(boxes, new_box):
        # bb interesection over union
        xA = np.maximum(boxes[:, 0], new_box[0])
        yA = np.maximum(boxes[:, 1], new_box[1])
        xB = np.minimum(boxes[:, 2], new_box[2])
        yB = np.minimum(boxes[:, 3], new_box[3])

        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxBArea = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])

        iou = interArea / (boxAArea + boxBArea - interArea)

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    ious = bb_iou_array(boxes[:, 4:8], new_box[4:8])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou


def compute_bbox_diag(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.linalg.norm([x2 - x1, y2 - y1])


def entropy(p):
    return (-p * np.log2(p)).sum(axis=0)


def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def compute_nme(gt, dt, norm_factor):
    res = []
    for i in range(5):
        if gt[3 * i + 2] == 0:
            res.append(None)
            continue

        dt_x, dt_y = dt[3 * i + 0], dt[3 * i + 1]
        gt_x, gt_y = gt[3 * i + 0], gt[3 * i + 1]

        dx = dt_x - gt_x
        dy = dt_y - gt_y

        res.append(np.linalg.norm([dx, dy]) / norm_factor)

    mean = [v for v in res if v is not None]
    mean = sum(mean) / (len(mean) + 1e-8)

    res.append(mean)
    return res


def compute_cross_nme(bbox_list):
    if len(bbox_list) == 0:
        return []

    nme_list = []
    norm_factor = compute_bbox_diag(bbox_list[0][4:8])
    for j in range(len(bbox_list)):
        for k in range(j + 1, len(bbox_list)):
            nme = compute_nme(bbox_list[j][8:], bbox_list[k][8:], norm_factor)
            mean_nme = nme[-1]
            nme_list.append(mean_nme)
    return nme_list


def weighted_boxes_fusion(
        boxes_list,
        scores_list,
        kp_list,
        labels_list,
        min_votes=1,
        weights=None,
        iou_thr=0.55,
        skip_box_thr=0.0,
        max_nme=0.07,
        conf_type='avg',
        allows_overflow=False
):
    """
    Performs weighted fusion of ensemble predictions.


    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param kp_list: list of key points predictions from each model
    :param min_votes: denotes how many models should agree about the prediction to keep it
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param max_nme: key points nme threshold
    :param conf_type: how to calculate confidence in weighted boxes.
        'avg': average value,
        'max': maximum value,
        'box_and_model_avg': box and model wise hybrid weighted average,
        'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: key points: landmarks coordinates (x1, y1, v1 ... x5, y5, v5)
    :return: labels: boxes labels
    """
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        logger.warning(f'Incorrect # of weights {len(weights)}. Must be: {len(boxes_list)}. Set weights equal to 1.')
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    _conf_types = ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']
    if conf_type not in _conf_types:
        logger.error(f'Unknown conf_type: {conf_type}. Must be one of {_conf_types}')
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, kp_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 15)), np.zeros((0,)), [], [], []

    overall_boxes = []

    reject_status = []
    kp_entr_list = []
    mean_nme_list = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = np.empty((0, 23))

        # Cluster boxes
        for i in range(0, len(boxes)):
            index, best_iou = find_matching_box_fast(weighted_boxes, boxes[i], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[i])
            else:
                new_boxes.append([boxes[i].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[i].copy()))

        # Perform weighted fusion for box clusters
        for j in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[j])

            weighted_box = get_weighted_box(clustered_boxes, conf_type)
            weighted_boxes[j] = weighted_box

        # Mark the boxes that do not match the given requirements as rejected
        rejected_annotations = [0.0 for _ in range(len(new_boxes))]
        for j in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[j])
            unique_models = np.unique(clustered_boxes[:, 3])

            if len(unique_models) < min_votes:
                rejected_annotations[j] = 1.0
                continue

            nme_list = compute_cross_nme(clustered_boxes)
            if any([_ > max_nme for _ in nme_list]):
                rejected_annotations[j] = 1.0
                e = entropy(np.array(nme_list))
                kp_entr_list.append(e)
                mean_nme_list.append(np.mean(nme_list))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = new_boxes[i]

            if conf_type == 'box_and_model_avg':
                clustered_boxes = np.array(clustered_boxes)
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                clustered_boxes = np.array(clustered_boxes)
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()

        reject_status.extend(rejected_annotations)
        overall_boxes.append(weighted_boxes)

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    if len(overall_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 15)), np.zeros((0,)), []

    indices = overall_boxes[:, 1].argsort()[::-1]
    overall_boxes = overall_boxes[indices]
    reject_status = np.array(reject_status)[indices]

    boxes = overall_boxes[:, 4:8]
    kps = overall_boxes[:, 8:23]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, kps, labels, reject_status, kp_entr_list, mean_nme_list
