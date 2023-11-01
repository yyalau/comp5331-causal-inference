import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lightning as L

from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel
from .modules import TransformerNet
from . import utils as ut

__all__ = ['ItnModel']


class ItnModel(nn.Module, StyleTransferModel):
    """
    Represents an Image Transfer Network [1]_ style transfer model for images.

    References
    ----------
    .. [1] Yijun Liu, Zuoteng Xu, Wujian Ye, Ziwen Zhang, Shaowei Weng, Chin-Chen Chang,
       and Huajin Tang. 2019. Image Neural Style Transfer With Preserving the Salient Regions.
       *IEEE Access 7* (2019), 40027--40037. <https://doi.org/10.1109/access.2019.2891576>
    """
    def __init__(self):
        super().__init__()

        # localization network
        self.features_blobs = []

        self.squeeze_net = models.squeezenet1_1() # pretrained=True
        self.squeeze_net.eval()
        self.squeeze_net._modules.get('features')[1].register_forward_hook(self.hook_feature)


        # perception network
        self.vgg16 = models.vgg16() # pretrained=True

        # main model: image transfer network
        self.transfer_model = TransformerNet()


    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        y = self.transfer_model(x)
        return y


if __name__ == "__main__":

    # TODO: test compatability with data loader
    # TODO: not yet tested if the code works / model is training properly
    model = ItnModel()
