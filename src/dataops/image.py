from __future__ import annotations

from collections.abc import Callable
import cv2
import numpy as np
import numpy.typing as npt

__all__ = ['create_image_loader']

def cv2_loader(path: str) -> npt.NDArray[np.float32]:
    """
    loads an image using the OpenCV library (cv2) from the specified
    path and returns it as a NumPy array with a data type of np.float32.
    """
    return np.array(cv2.imread(path))


def image_loader(path: str) -> npt.NDArray[np.float32]:
    if path.endswith(".jpg") or path.endswith(".png"):
        return cv2_loader(path)
    raise NotImplementedError("Image format not supported.")


def create_image_loader(path: str, lazy: bool) -> Callable[[], npt.NDArray[np.float32]]:
    """
    Returns a callable object that loads images.
    If `lazy` is True, the returned callable is a lambda function
    that, when called, will load and return the image using
    the image_loader() method. If `lazy` is False, the image
    is loaded immediately, and the returned callable will always
    return the same pre-loaded image.
    """
    if lazy:
        return lambda: image_loader(path)
    image = image_loader(path)
    return lambda: image
