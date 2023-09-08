import cv2
import numpy as np
from typing import Tuple, List

from autoannotator.types.base import ImageColorFormat


__all__ = ['resize_image', 'np2onnx', 'normalize_image']


def _compute_new_shape(
        img0_shape: Tuple[int, int],
        img1_shape: Tuple[int, int],
        keep_ratio: bool = True
) -> Tuple[int, int]:
    """
    Compute resize HxW dimensions

    Arguments:
        img0_shape (Tuple[int, int]): The input image H0 x W0 shape
        img1_shape (Tuple[int, int]): The target H1 x W1 shape
        keep_ratio (bool): Whether to resize keeping ratio or not
    Returns:
        Tuple[int, int]: New width x New height to resize to
    """
    h0, w0 = img0_shape
    h1, w1 = img1_shape
    if keep_ratio:
        img_ratio = float(w0) / h0

        w_ratio = float(w1) / float(w0)
        h_ratio = float(h1) / float(h0)

        if w_ratio < h_ratio:
            new_w = w1
            new_h = int(new_w / img_ratio)
        else:
            new_h = h1
            new_w = int(new_h * img_ratio)

        return new_h, new_w
    else:
        return h1, w1


def resize_image(
        img: np.ndarray,
        size: Tuple[int, int],
        keep_ratio: bool = True,
        position: str = 'center',
        value: int = 0
) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    Resize image

    Arguments:
        img (np.ndarray): The input image H0 x W0 x 3
        size (Tuple[int, int]): The target H1 x W1 size
        keep_ratio (bool): Whether to resize keeping ratio or not
        position (str): Position to insert resized image (`topleft` and `center` options are available).
        value (int): Border value
    Returns:
        (np.ndarray): Resized image (H1 x W1 x 3)
        (Tuple[int, int]): xy translation shift
        (float): resize scale (resized to original scale)
    """
    h1, w1 = size
    h0, w0 = img.shape[:2]

    out_img = np.ones((h1, w1, 3), dtype=np.uint8) * value

    new_h, new_w = _compute_new_shape(img0_shape=(h0, w0), img1_shape=(h1, w1), keep_ratio=keep_ratio)

    scale = float(new_h) / h0
    resized_img = cv2.resize(img, (new_w, new_h))       # dsize in cv2.resize is in (w x h) format

    if position == 'center':
        x1 = (w1 - new_w) // 2
        y1 = (h1 - new_h) // 2
        x2 = (x1 + new_w)
        y2 = (y1 + new_h)
        assert 0 <= x1 <= w1 and 0 <= x2 <= w1
        assert 0 <= y1 <= h1 and 0 <= y2 <= h1
        out_img[y1:y2, x1:x2, :] = resized_img
        shift = (x1, y1)
    elif position == 'topleft':
        out_img[:new_h, :new_w, :] = resized_img
        shift = (0, 0)
    else:
        raise NotImplementedError(f'Only `center` and `topleft` positions are supported. Got: {position}')

    return out_img, shift, scale


def np2onnx(img: np.ndarray, color_mode: ImageColorFormat = ImageColorFormat.RGB) -> np.ndarray:
    """
    Convert numpy image to onnx-friendly input format

    Arguments:
        img (np.ndarray): The input image numpy array HxWx3 in rgb format
        color_mode (str): output color mode (rgb or bgr)
    Returns:
        (np.ndarray): ONNX-friendly input (BxCxHxW)
    """
    if color_mode == ImageColorFormat.BGR:
        img = img[:, :, ::-1]
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img


def normalize_image(img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    """
    Normalize numpy rgb image

    Arguments:
        img (np.ndarray): The input image numpy array HxWx3 in rgb format
        mean (List[float]): normalized pixels mean list for 3 rgb channels in the range [0, 1]
        std (List[float]): normalized pixels std pixels list for 3 rgb channels in the range [0, 1]
    Returns:
        (np.ndarray): Normalized image
    """
    mean = np.array(mean)
    std = np.array(std)

    if img.dtype == np.uint8:
        img = img / 255
    img = (img - mean) / std
    return img
