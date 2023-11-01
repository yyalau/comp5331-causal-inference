from __future__ import annotations

from typing import Protocol, TypeAlias, TypedDict, runtime_checkable

import torch

from ..base import NNModule

__all__ = ['StyleTransfer_X', 'StyleTransfer_Y', 'StyleTransferModel']


class StyleTransfer_X(TypedDict):
    style: torch.Tensor
    """
    A batch of images from which to extract the style.

    Shape: `(batch_size, num_channels, height, width)`
    """

    content: torch.Tensor
    """
    A batch of images to which the style is applied.

    Shape: `(batch_size, num_channels, height, width)`
    """

StyleTransfer_Y: TypeAlias = torch.Tensor
"""
A batch of style-transferred images.

Shape: `(batch_size, num_channels, height, width)`
"""

@runtime_checkable
class StyleTransferModel(NNModule[StyleTransfer_X, StyleTransfer_Y], Protocol):
    """
    Represents a style transfer model for images.
    """
