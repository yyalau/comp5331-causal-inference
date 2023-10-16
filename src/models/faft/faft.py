from __future__ import annotations

from typing import Generic, TypeVar

from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import Normalize

from modelops.dependencies import ClassificationModel, FAST_X, Classification_Y

from fourier import fourierMix

__all__ = ['FAFT']

Classification_Output = TypeVar('Classification_Output', bound=Classification_Y)

class FAST(nn.Module, Generic[Classification_Output]):
    def __init__(self,
                classifer: ClassificationModel[FAST_X],
                model = resnet18,
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
        self.model = model
        self.beta = beta
        self.normalization = Normalize(mean = pixel_mean, std = pixel_std)


    def forward(self, input: FAST_X) -> Classification_Output:
        content = input.get('content')
        styles = input.get('styles')

        #TODO: may need to downsize
        fx_primes = []
        for x_prime in styles:
            fx_prime = fourierMix(content, x_prime)
            fx_prime = self.model(self.normalization(fx_prime))
            fx_primes.append(fourierMix(content, x_prime))
        fx = self.model(self.normalization(content))
        weighted_output = fx*self.beta+(1-self.beta)*sum(fx_primes)/len(fx_primes)

        classifier_input: FAST_X = {'content': weighted_output, 'styles': styles}
        predictions: Classification_Output = self.classifier(classifier_input)

        return predictions

