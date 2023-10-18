from __future__ import annotations

from typing import TypeAlias, TypedDict

import torch

__all__ = ['StyleTransfer_X', 'StyleTransfer_Y']


class StyleTransfer_X(TypedDict):
    style: torch.Tensor
    """
    A batch of images from which to extract the style.

    Shape: `(batch_size, num_channels, depth, height, width)`
    """

    content: torch.Tensor
    """
    A batch of images to which the style is applied.

    Shape: `(batch_size, num_channels, depth, height, width)`
    """

StyleTransfer_Y: TypeAlias = torch.Tensor
"""
A batch of style-transferred images.

Shape: `(batch_size, num_channels, depth, height, width)`
"""
