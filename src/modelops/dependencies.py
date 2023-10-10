from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self, TypeAlias, TypedDict

import torch

__all__ = [
    'StyleTransferInputBatch', 'StyleTransferOutputBatch', 'StyleTransferModel',
    'ClassifierInputBatch', 'ClassifierOutputBatch', 'ClassifierModel',
    'CausalFDInputBatch', 'CausalFDOutputBatch', 'CausalFDModel',
]

torch.nn.Module().state_dict
class NNModule(Protocol):
    def train(self, mode: bool = ...) -> Self:
        """
        See :meth:`torch.nn.Module.train`.
        """
        ...

    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Mapping[str, object]:
        """
        See :meth:`torch.nn.Module.state_dict`.
        """
        ...


class StyleTransferInputBatch(TypedDict):
    style: torch.Tensor
    """Shape: `(batch_size, num_channels, depth, height, width)`"""

    content: torch.Tensor
    """Shape: `(batch_size, num_channels, depth, height, width)`"""

StyleTransferOutputBatch: TypeAlias = torch.Tensor
"""Shape: `(batch_size, num_classes)`"""

class StyleTransferModel(NNModule, Protocol):
    def forward(self, style: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        style : tensor with shape `(batch_size, num_channels, depth, height, width)`
            A batch of images from which to extract the style.
        content : tensor with shape `(batch_size, num_channels, depth, height, width)`
            A batch of images to which the style is applied.

        Returns
        -------
        tensor with shape `(batch_size, num_channels, depth, height, width)`
            A batch of style-transferred images.
        """
        ...


ClassifierInputBatch: TypeAlias = torch.Tensor
"""Shape: `(batch_size, num_channels, depth, height, width)`"""

ClassifierOutputBatch: TypeAlias = torch.Tensor
"""Shape: `(batch_size, num_classes)`"""

class ClassifierModel(NNModule, Protocol):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ : tensor with shape `(batch_size, num_channels, depth, height, width)`
            A batch of images to classify.

        Returns
        -------
        tensor with shape `(batch_size, num_classes)`
            A batch of class probabilities.
        """
        ...


class CausalFDInputBatch(TypedDict):
    content: torch.Tensor
    """Shape: `(batch_size, num_channels, depth, height, width)`"""

    styles: list[torch.Tensor]
    """
    A list of tensors, one for each style.

    Shape: `(batch_size, num_channels, depth, height, width)`
    """

CausalFDOutputBatch: TypeAlias = torch.Tensor
"""Shape: `(batch_size, num_classes)`"""

class CausalFDModel(NNModule, Protocol):
    def forward(self, input_: torch.Tensor, styles: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ : tensor with shape `(batch_size, num_channels, depth, height, width)`
            A batch of images to classify.
        styles : list of tensor, each with shape `(batch_size, num_channels, depth, height, width)`
            For each style, a batch of images from which to extract the style.

        Returns
        -------
        tensor with shape `(batch_size, num_classes)`
            A batch of class probabilities.
        """
        ...
