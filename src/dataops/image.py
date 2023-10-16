from __future__ import annotations

from collections.abc import Callable
import cv2
import numpy as np
import numpy.typing as npt


def cv2_loader(path: str) -> npt.NDArray[np.float32]:
    return np.array(cv2.imread(path))


def image_loader(path: str) -> npt.NDArray[np.float32]:
    if path.endswith(".jpg") or path.endswith(".png"):
        return cv2_loader(path)
    raise NotImplementedError("Image format not supported.")


def create_image_loader(path: str, lazy: bool) -> Callable[[], npt.NDArray[np.float32]]:
    if lazy:
        return lambda: image_loader(path)
    image = image_loader(path)
    return lambda: image
