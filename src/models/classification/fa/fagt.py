from __future__ import annotations

import torch.nn as nn
from torchvision.transforms import Normalize

from ...nst import StyleTransferModel
from ..erm import ERMModel

from .base import FA_X, Classification_Y, FAModel
from .fourier import fourierMix

__all__ = ['FAGT']


class FAGT(nn.Module, FAModel):
    """
    Represents a FAGT (Front-door Adjustment via General Transfer) [1]_ classifier for images.

    References
    ----------
    .. [1] Toan Nguyen, Kien Do, Duc Thanh Nguyen, Bao Duong, and Thin Nguyen. 2023.
       Causal Inference via Style Transfer for Out-of-distribution Generalisation.
       In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)*.
       Association for Computing Machinery, New York, NY, USA, 1746--1757.
       <https://doi.org/10.1145/3580305.3599270>
    """
    def __init__(self,
                nst: StyleTransferModel,
                classifer: ERMModel,
                device: str = 'cpu', # "cpu" for cpu, "cuda" for gpu
                eta: float = 2.0, # namda ~ U(0,eta) eta controlling the maximum style mixing rate
                beta: float = 0.2, # interpolation coefficient
                pixel_mean: list[float] = [0.5, 0.5, 0.5], # mean for normolization
                pixel_std: list[float] = [0.5, 0.5, 0.5], # std for normolization
                gamma: float = 2.0, # Controls importance of StyleLoss vs ContentLoss, Loss = gamma*StyleLoss + ContentLoss
                training: bool = True, # Wether or not network is training
                scr_temperature: float = 0.1,
                ) -> None:
        super(FAGT).__init__()

        self.fst = fourierMix(eta).to(device)
        self.nst = nst(gamma, training, scr_temperature).to(device)
        self.classifier = classifer
        self.beta = beta
        self.normalization = Normalize(mean = pixel_mean, std = pixel_std)

    def forward(self, input: FA_X) -> Classification_Y:
        content = input.get('content')
        styles = input.get('styles')

        #TODO: may need to downsize
        fx_hats = []
        fx_tildes = []
        for x_prime in styles:
            x_hat = self.fst(content, x_prime)
            fx_hat = self.classifier(self.normalization(x_hat))
            fx_hats.append(fx_hat)

            x_tilde = self.nst(content, x_prime)
            fx_tilde = self.classifier(self.normalization(x_tilde))
            fx_tildes.append(fx_tilde)

        fx = self.classifier(self.normalization(content))
        weighted_output = fx*self.beta+(1-self.beta)*(sum(fx_hats)+sum(fx_tildes))/(len(fx_hats)+len(fx_tildes))

        predictions: Classification_Y = weighted_output

        return predictions
