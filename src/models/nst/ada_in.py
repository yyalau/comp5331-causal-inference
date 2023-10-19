from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from ..base import NNModule

from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel

__all__ = ['AdaINEncoder', 'AdaINDecoder', 'AdaINModel']


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


class AdaINModel(nn.Module, StyleTransferModel):
    """
    Represents an AdaIN (Adaptive Instance Normalization) [1]_ style transfer model for images.

    References
    ----------
    .. [1] Xun Huang and Serge Belongie. 2017. Arbitrary style transfer in real-time with adaptive
       instance normalization. In *CVPR*. 1501--1510. <https://doi.org/10.48550/arXiv.1703.06868>
    """
    def __init__(
        self,
        encoder: AdaINEncoder,
        decoder: AdaINDecoder,
        *,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.alpha = alpha

    def ada_in(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_std, content_mean = torch.std_mean(content, dim=(-2, -1))
        style_std, style_mean = torch.std_mean(style, dim=(-2, -1))

        return style_std * (content - content_mean) / content_std + style_mean

    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        enc_style = self.encoder(x['style'])
        enc_content = self.encoder(x['content'])
        alpha = self.alpha

        return alpha * self.ada_in(enc_content, enc_style) + (1 - alpha) * enc_content
