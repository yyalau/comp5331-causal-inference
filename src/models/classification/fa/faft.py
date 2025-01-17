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
    classifier : ERMModel
        The classifier model to use in the network.
    eta : float, default 1.0
        A hyperparameter controlling the maximum style mixing rate.
        This value should be between 0.0 and 1.0 inclusive.
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
        classifier: ERMModel,
        eta: float = 1.0,
        beta: float = 0.2,
        pixel_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        pixel_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()

        self._classifier = classifier
        self._eta = eta
        self._beta = beta
        self._pixel_mean = pixel_mean
        self._pixel_std = pixel_std

        self.fst = FourierMix(eta)
        self.normalization = Normalize(mean=pixel_mean, std=pixel_std)

    @property
    def classifier(self) -> ERMModel:
        return self._classifier

    @property
    def eta(self) -> float:
        return self._eta

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

    def get_x_hats(self, inputs: FA_X) -> list[torch.Tensor]:
        content = inputs.get('content')
        styles = inputs.get('styles')

        # TODO: may need to downsize
        return [self.fst(content, x_prime) for x_prime in styles]

    def forward(self, inputs: FA_X) -> Classification_Y:
        x = inputs.get('content')
        fx = self.classifier(self.normalization(x))

        fx_hats = [self.classifier(self.normalization(x_hat)) for x_hat in self.get_x_hats(inputs)]
        fx_hats_avg = torch.stack(fx_hats, dim=0).mean(dim=0)

        weighted_output = fx * self.beta + (1 - self.beta) * fx_hats_avg
        predictions: Classification_Y = weighted_output

        return predictions
