from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from ...nst import StyleTransferModel
from ..erm import ERMModel

from .base import FA_X, Classification_Y, FAModel

__all__ = ['FAST']


class FAST(nn.Module, FAModel):
    """
    Represents a FAST (Front-door Adjustment via Neural Style Transfer) [1]_ classifier for images.

    Parameters
    ----------
    nst : StyleTransferModel
        The style transfer model to use in the network.
    classifier : ERMModel
        The classifier model to use in the network.
    beta : float, default 0.2
        The interpolation coefficient between the original image and the stylized image.
    pixel_mean : tuple of float, default (0.5, 0.5, 0.5)
        For each channel, the mean value of pixels to be used for normalization.
    pixel_mean : tuple of float, default (0.5, 0.5, 0.5)
        For each channel, the standard deviation of pixels to be used for normalization.

    References
    ----------
    .. [1] Toan Nguyen, Kien Do, Duc Thanh Nguyen, Bao Duong, and Thin Nguyen. 2023.
       Causal Inference via Style Transfer for Out-of-distribution Generalisation.
       In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)*.
       Association for Computing Machinery, New York, NY, USA, 1746--1757.
       <https://doi.org/10.1145/3580305.3599270>
    """

    def __init__(
        self,
        nst: StyleTransferModel,
        classifier: ERMModel,
        beta: float = 0.2,
        pixel_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        pixel_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()

        self._nst = nst
        self._classifier = classifier
        self._beta = beta
        self._pixel_mean = pixel_mean
        self._pixel_std = pixel_std

        self.normalization = Normalize(mean=pixel_mean, std=pixel_std)

        # The NST model is considered frozen when training by FA
        for p in self.nst.parameters():
            p.requires_grad = False

    @property
    def nst(self) -> StyleTransferModel:
        return self._nst

    @property
    def classifier(self) -> ERMModel:
        return self._classifier

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def pixel_mean(self) -> tuple[float, float, float]:
        return self._pixel_mean

    @property
    def pixel_std(self) -> tuple[float, float, float]:
        return self._pixel_std

    def get_num_classes(self) -> int:
        return self.classifier.get_num_classes()

    def forward(self, input: FA_X) -> Classification_Y:
        content = input.get('content')
        styles = input.get('styles')

        fx_tildes = []
        for x_prime in styles:
            x_tilde = self.nst({'style': x_prime, 'content': content})
            fx_tilde = self.classifier(self.normalization(x_tilde))
            fx_tildes.append(fx_tilde)
        fx_tildes_avg = torch.stack(fx_tildes, dim=0).mean(dim=0)

        fx = self.classifier(self.normalization(content))
        weighted_output = fx * self.beta + (1 - self.beta) * fx_tildes_avg

        predictions: Classification_Y = weighted_output

        return predictions
