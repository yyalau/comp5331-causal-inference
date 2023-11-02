from __future__ import annotations

from collections.abc import Callable
import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path

from typing_extensions import Optional, TypeAlias
from dataclasses import dataclass
from PIL import Image
import torch
import torchvision.transforms.v2 as T

from .augmentation import RandAugment

__all__ = ['ImageLoader', "RandAugmentParams", "ImageResizeParams", "PreprocessParams"]


Tensor: TypeAlias = torch.Tensor

@dataclass
class RandAugmentParams:
    """
    Parameters
    ----------
    alpha : int
        random augmentation parameter.
    beta : int
        random augmentation parameter.
    """
    alpha: int
    beta: int

@dataclass
class ImageResizeParams:
    """
    Parameters
    ----------
    resize_height : int
        Image height after resizing.
    resize_width : int
        Image width after resizing.
    interpolation_mode : T.InterpolationMode
        Interpolation mode for reshaping the image.
    """
    height: int
    width: int
    interpolation_mode: T.InterpolationMode

@dataclass
class PreprocessParams:
    """
    Parameters
    ----------
    image_resize_params : ImageResizeParams
        Parameters for resizing the image.
    rand_augment_params : RandAugmentParams
        Parameters for random augmentation.
    """
    image_resize_params: ImageResizeParams
    rand_augment_params: Optional[RandAugmentParams]

class ImageLoader:

    def __init__(self, path: Path, lazy: bool, preprocess_params: PreprocessParams):
        super().__init__()
        self.transform = self._get_transform(preprocess_params)
        self.rand_augment = self._get_rand_augment(preprocess_params)
        self.load_callback = self._load(path, lazy)

    def _get_transform(self, preprocess_params: PreprocessParams) -> T.Compose:
        resize_params = preprocess_params.image_resize_params
        return T.Compose([
            T.Resize(
                (resize_params.height, resize_params.width),
                interpolation=resize_params.interpolation_mode
            ),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
        ])

    def _get_rand_augment(self, preprocess_params: PreprocessParams) -> RandAugment:
        rand_augment_params = preprocess_params.rand_augment_params
        if rand_augment_params is None:
            return RandAugment(None, None)
        alpha = rand_augment_params.alpha
        beta = rand_augment_params.beta
        return RandAugment(alpha, beta)

    def __call__(self) -> Tensor:
        return self.load_callback()

    def _preprocess(
        self,
        image_array: npt.NDArray[np.float32],
    ) -> Tensor:
        image = Image.fromarray(image_array)
        augmented_img = self.rand_augment(image)
        resized_img = self.transform(augmented_img)
        assert isinstance(resized_img, Tensor)
        return resized_img

    def _load_from_file(self, path: Path) -> npt.NDArray[np.float32]:
        if path.suffix == ".jpg" or path.suffix == ".png":
            return self._cv2_loader(path)
        raise NotImplementedError("Image format not supported.")


    def _cv2_loader(self, path: Path) -> npt.NDArray[np.float32]:
        """
        loads an image using the OpenCV library (cv2) from the specified
        path and returns it as a NumPy array with a data type of np.float32.
        """
        return np.array(cv2.imread(str(path)))

    def _load(
        self,
        path: Path,
        lazy: bool
    ) -> Callable[[], Tensor]:
        """
        Returns a callable object that loads images.
        If `lazy` is True, the returned callable is a lambda function
        that, when called, will load and return the image using
        the image_loader() method. If `lazy` is False, the image
        is loaded immediately, and the returned callable will always
        return the same pre-loaded image.
        """
        if lazy:
            return lambda: self._preprocess(self._load_from_file(path))
        image = self._preprocess(self._load_from_file(path))
        return lambda: image
