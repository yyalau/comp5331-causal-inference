from __future__ import annotations

from typing import Protocol, TypedDict

import torch

from .base import ClassificationModel

__all__ = ['FAST_X', 'FASTModel']


class FAST_X(TypedDict):
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

class FASTModel(ClassificationModel[FAST_X], Protocol):
    """
    Represents a FAST (Front-door Adjustment via Neural Style Transfer) classifier for images.
    """
