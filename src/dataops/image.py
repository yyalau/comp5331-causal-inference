from __future__ import annotations

from collections.abc import Callable
import cv2
import numpy as np


def cv2_loader(path: str) -> np.ndarray:
    return np.array(cv2.imread(path))


def image_loader(path: str) -> np.ndarray:
    if path.endswith(".jpg") or path.endswith(".png"):
        return cv2_loader(path)
    raise NotImplementedError("Image format not supported.")


def create_image_loader(path: str, lazy: bool) -> Callable[[], np.ndarray]:
    if lazy:
        return lambda: image_loader(path)
    return lambda: image_loader(path).copy()
