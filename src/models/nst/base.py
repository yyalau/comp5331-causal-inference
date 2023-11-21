from __future__ import annotations

from typing import Protocol, TypeAlias, TypedDict, runtime_checkable

import os
import torch
import torch.nn as nn

from ...dataops.utils import download_from_gdrive

from ..base import NNModule

__all__ = ['StyleTransfer_X', 'StyleTransfer_Y', 'StyleTransferModel', 'PretrainedNNModule']


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


class PretrainedNNModule(NNModule[torch.Tensor, torch.Tensor], Protocol):
    """
    A partial interface for :class:`torch.nn.Module` that can be loaded from a pre-trained
    """
    default_url: str
    default_wpath: str
    net: nn.Sequential

    def load_pretrain(self, *, pretrain: bool, net: nn.Module) -> None:
        """
        Loads the weights for the model from a given path.
        """
        if not pretrain:
            return

        if not os.path.exists(self.default_wpath):
            os.makedirs(os.path.dirname(self.default_wpath), exist_ok=True)
            if 'drive.google.com' in self.default_url:
                download_from_gdrive(self.default_url, self.default_wpath)
            else:
                os.system(
                    f'wget {self.default_url} -O '
                    + os.path.join(self.default_wpath)
                )

        net.load_state_dict(torch.load(self.default_wpath))
