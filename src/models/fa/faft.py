from __future__ import annotations

from typing import Generic, TypeVar

from torch import Tensor
import torch.nn as nn
# from torchvision.models import resnet18
from torchvision.transforms import Normalize

from modelops.dependencies import ClassificationModel, FAST_X, Classification_Y

from fourier import fourierMix

__all__ = ['FAFT']

Classification_Output = TypeVar('Classification_Output', bound=Classification_Y)

class FAST(nn.Module, Generic[Classification_Output]):
    def __init__(self,
                classifer: ClassificationModel[FAST_X],
                device: str = 'cpu', # "cpu" for cpu, "cuda" for gpu
                eta: float = 2.0, # namda ~ U(0,eta) eta controlling the maximum style mixing rate
                beta: float = 0.2, # interpolation coefficient
                pixel_mean: float[3] = [0.5, 0.5, 0.5], # mean for normolization
                pixel_std: float[3] = [0.5, 0.5, 0.5], # std for normolization
                training: bool =True, # Wether or not network is training
                ) -> None:
        super(FAST).__init__()


        self.style_transfer = fourierMix(eta).to(device)
        self.classifier = classifer
        self.beta = beta
        self.normalization = Normalize(mean = pixel_mean, std = pixel_std)


    def forward(self, input: FAST_X) -> Classification_Output:
        content = input.get('content')
        styles = input.get('styles')

        #TODO: may need to downsize
        fx_hats = []
        for x_prime in styles:
            x_hat = self.style_transfer(content, x_prime)
            fx_hat = self.classifier(self.normalization(x_hat))
            fx_hats.append(fx_hat)
        fx = self.classifier(self.normalization(content))
        weighted_output = fx*self.beta+(1-self.beta)*sum(fx_hats)/len(fx_hats)

        predictions: Classification_Output = weighted_output

        return predictions

