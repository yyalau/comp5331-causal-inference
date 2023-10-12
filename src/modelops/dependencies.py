from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, TypeAlias, TypedDict, TypeVar

import torch

__all__ = [
    'StyleTransfer_X', 'StyleTransfer_Y',
    'AdaINEncoder', 'AdaINDecoder',
    'Classification_Y', 'ClassificationModel',
    'ERM_X', 'ERMModel',
    'FAST_X', 'FASTModel',
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
    def __call__(self, x: X_contra) -> Y_co:
        """
        Same as :meth:`torch.nn.Module.__call__`.
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

class AdaINEncoder(NNModule[torch.Tensor, torch.Tensor], Protocol):
    def get_states(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """
        Similar to :meth:`torch.nn.Module.__call__`, but returns the output of
        each intermediate layer; the last output is the same as the result of
        :meth:`torch.nn.Module.__call__`.
        """
        ...

class AdaINDecoder(NNModule[torch.Tensor, torch.Tensor], Protocol):
    pass


Classification_Y: TypeAlias = torch.Tensor
"""
A batch of class probabilities.

Shape: `(batch_size, num_classes)`
"""

ClassificationModel: TypeAlias = NNModule[X, Classification_Y]
"""
Represents a classification model.
"""


ERM_X: TypeAlias = torch.Tensor
"""
A batch of images to classify.

Shape: `(batch_size, num_channels, depth, height, width)`
"""

ERMModel: TypeAlias = ClassificationModel[ERM_X]
"""
Represents a standard ERM (Empirical Risk Minimization) classifier for images.
"""


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

FASTModel: TypeAlias = ClassificationModel[FAST_X]
"""
Represents a FAST (Front-door Adjustment via Neural Style Transfer) classifier for images.
"""
