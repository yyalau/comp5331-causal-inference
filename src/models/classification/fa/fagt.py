from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from ...nst import StyleTransferModel
from ..erm import ERMModel

from .base import FA_X, Classification_Y, FAModel
from .fourier import FourierMix

__all__ = ['FAGT']


class FAGT(nn.Module, FAModel):
    """
    Represents a FAGT (Front-door Adjustment via General Transfer) [1]_ classifier for images.

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
        nst: StyleTransferModel,
        classifier: ERMModel,
        eta: float = 2.0,
        beta: float = 0.2,
        pixel_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        pixel_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()

        self._nst = nst
        self._classifier = classifier
        self._eta = eta
        self._beta = beta
        self._pixel_mean = pixel_mean
        self._pixel_std = pixel_std

        self.fst = FourierMix(eta)
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

    def get_hparams(self) -> dict[str, object]:
        return dict(
            nst=dict(
                name=type(self.nst).__name__,
                hparams=self.nst.get_hparams(),
            ),
            classifier=dict(
                name=type(self.classifier).__name__,
                hparams=self.classifier.get_hparams(),
            ),
            eta=self.eta,
            beta=self.beta,
            pixel_mean=self.pixel_mean,
            pixel_std=self.pixel_std,
        )

    def forward(self, input: FA_X) -> Classification_Y:
        content = input.get('content')
        styles = input.get('styles')

        # TODO: may need to downsize
        fx_hats = []
        fx_tildes = []
        for x_prime in styles:
            x_hat = self.fst(content, x_prime)
            fx_hat = self.classifier(self.normalization(x_hat))
            fx_hats.append(fx_hat)

            x_tilde = self.nst({'style': x_prime, 'content': content})
            fx_tilde = self.classifier(self.normalization(x_tilde))
            fx_tildes.append(fx_tilde)
        fx_hats_tildes_avg = torch.stack([*fx_hats, *fx_tildes], dim=0).mean(dim=0)

        fx = self.classifier(self.normalization(content))
        weighted_output = fx * self.beta + (1 - self.beta) * fx_hats_tildes_avg

        predictions: Classification_Y = weighted_output

        return predictions
