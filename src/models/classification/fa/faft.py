from __future__ import annotations

import torch
import torch.nn as nn
# from torchvision.models import resnet18
from torchvision.transforms import Normalize

from ..erm import ERMModel

from .base import FA_X, Classification_Y, FAModel
from .fourier import FourierMix

__all__ = ['FAFT']


class FAFT(nn.Module, FAModel):
    """
    Represents a FAFT (Front-door Adjustment via Fourier-based Style Transfer) [1]_ classifier for images.

    Parameters
    ----------
    nst : StyleTransferModel
        The style transfer model to use in the network.
    classifier : ERMModel
        The classifier model to use in the network.
    eta : float, default 2.0
        A hyperparameter controlling the maximum style mixing rate.
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
        classifer: ERMModel,
        eta: float = 2.0,
        beta: float = 0.2,
        pixel_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        pixel_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()

        self.style_transfer = FourierMix(eta)
        self.classifier = classifer
        self.beta = beta
        self.normalization = Normalize(mean=pixel_mean, std=pixel_std)

    def forward(self, input: FA_X) -> Classification_Y:
        content = input.get('content')
        styles = input.get('styles')

        # TODO: may need to downsize
        fx_hats = []
        for x_prime in styles:
            x_hat = self.style_transfer(content, x_prime)
            fx_hat = self.classifier(self.normalization(x_hat))
            fx_hats.append(fx_hat)
        fx_hats_avg = torch.stack(fx_hats, dim=0).mean(dim=0)

        fx = self.classifier(self.normalization(content))
        weighted_output = fx * self.beta + (1 - self.beta) * fx_hats_avg

        predictions: Classification_Y = weighted_output

        return predictions
