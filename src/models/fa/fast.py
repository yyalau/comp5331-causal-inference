from __future__ import annotations

from typing import Generic, TypeVar

import torch.nn as nn
from torchvision.transforms import Normalize

from modelops.dependencies import ClassificationModel, FAST_X, Classification_Y

__all__ = ['FAST']

NST = TypeVar('NST', bound=nn.Module)

Classification_Output = TypeVar('Classification_Output', bound=Classification_Y)

class FAST(nn.Module, Generic[NST, Classification_Output]):
    def __init__(self,
                style_transfer: NST,
                classifer: ClassificationModel[FAST_X],
                device: str = 'cpu', # "cpu" for cpu, "cuda" for gpu
                pixel_mean: list[float] = [0.5, 0.5, 0.5], # mean for normolization
                pixel_std: list[float] = [0.5, 0.5, 0.5], # std for normolization
                gamma: float = 2.0, # Controls importance of StyleLoss vs ContentLoss, Loss = gamma*StyleLoss + ContentLoss
                training: bool =True, # Wether or not network is training
                scr_temperature: float = 0.1,
                ) -> None:
        super(FAST).__init__()


        self.style_transfer = style_transfer(gamma, training, scr_temperature).to(device)
        self.classifier = classifer
        self.normalization = Normalize(mean = pixel_mean, std = pixel_std)


    def forward(self, input: FAST_X) -> Classification_Output:
        content = input.get('content')
        styles = input.get('styles')

        # transferred_contents: Tensor = self.style_transfer(content, styles)
        fx_tildes = []
        for x_prime in styles:
            x_tilde = self.style_transfer(content, x_prime)
            fx_tilde = self.classifier(self.normalization(x_tilde))
            fx_tildes.append(fx_tilde)

        fx = self.classifier(self.normalization(content))
        weighted_output = fx*self.beta+(1-self.beta)*sum(fx_tildes)/len(fx_tildes)

        # classifier_input: FAST_X = {'content': weighted_output, 'styles': styles}
        # predictions: Classification_Output = self.classifier(classifier_input)
        predictions: Classification_Output = weighted_output

        return predictions

