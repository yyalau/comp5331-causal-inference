from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, TypeAlias, TypedDict, TypeVar

import torch

__all__ = [
    'StyleTransfer_X', 'StyleTransfer_Y', 'StyleTransferModel',
    'Classifier_X', 'Classifier_Y', 'ClassifierModel',
    'CausalFD_X', 'CausalFD_Y', 'CausalFDModel',
]


# Input
X = TypeVar('X')
X_contra = TypeVar('X_contra', contravariant=True)

# Output
Y = TypeVar('Y')
Y_co = TypeVar('Y_co', covariant=True)

class NNModule(Protocol[X_contra, Y_co]):
    """
    A partial interface for :class:`torch.nn.Module`.
    """
    def forward(self, x: X_contra) -> Y_co:
        """
        Same as :meth:`torch.nn.Module.forward`.
        """
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """
        Same as :meth:`torch.nn.Module.parameters`.
        """
        ...


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

Shape: `(batch_size, num_classes)`
"""

StyleTransferModel: TypeAlias = NNModule[StyleTransfer_X, StyleTransfer_Y]
"""
Represents a style transfer model for images.
"""


Classifier_X: TypeAlias = torch.Tensor
"""
A batch of images to classify.

Shape: `(batch_size, num_channels, depth, height, width)`
"""

Classifier_Y: TypeAlias = torch.Tensor
"""
A batch of class probabilities.

Shape: `(batch_size, num_classes)`
"""

ClassifierModel: TypeAlias = NNModule[Classifier_X, Classifier_Y]
"""
Represents a classifier model for images.
"""


class CausalFD_X(TypedDict):
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

CausalFD_Y: TypeAlias = torch.Tensor
"""
A batch of class probabilities.

Shape: `(batch_size, num_classes)`
"""

CausalFDModel: TypeAlias = NNModule[CausalFD_X, CausalFD_Y]
"""
Represents a causal front-door adjustment model for images.
"""
