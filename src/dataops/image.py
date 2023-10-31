from __future__ import annotations

from collections.abc import Callable
import cv2
import numpy as np
import numpy.typing as npt
from pathlib import Path

from typing_extensions import TypeAlias



from dataclasses import dataclass
from PIL import Image
import torch
import torchvision.transforms.v2 as T

from .augmentation import RandAugment

__all__ = ['ImageLoader']


Tensor: TypeAlias = torch.Tensor

@dataclass
class PreprocessParams:
    height: int
    width: int
    interpolation_mode: T.InterpolationMode
    augment: RandAugment

class ImageLoader:

    def __init__(self, path: Path, lazy: bool, preprocess_params: PreprocessParams):
        super().__init__()
        self.preprocess_params = preprocess_params
        self.load_callback = self._load(path, lazy)

    def __call__(self) -> Tensor:
        return self.load_callback()

    def _preprocess(
        self,
        image_array: npt.NDArray[np.float32],
    ) -> Tensor:
        params = self.preprocess_params
        image = Image.fromarray(image_array)
        transform = T.Compose([
            T.Resize((params.height, params.width), interpolation=params.interpolation_mode),
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float32),
        ])
        resized_img = transform(params.augment(image))
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
