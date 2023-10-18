from __future__ import annotations

from typing import Protocol, TypedDict

import torch

from ..base import ClassificationModel, Classification_Y

__all__ = ['FA_X', 'Classification_Y', 'FAModel']


class FA_X(TypedDict):
    content: torch.Tensor
    """
    A batch of images to classify.

    Shape: `(batch_size, num_channels, depth, height, width)`
    """

    styles: list[torch.Tensor]
    """
    For each style, a tensor containing a batch of images from which to extract the style.

    Shape: `(batch_size, num_channels, depth, height, width)`
    """

class FAModel(ClassificationModel[FA_X], Protocol):
    """
    Represents a FA (Front-door Adjustment via Neural Style Transfer) classifier for images.
    """
