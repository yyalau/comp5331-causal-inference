import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_1
import lightning as L

from ..base import NNModule
from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel
from .modules import TransferNet
from . import utils as ut

__all__ = ['ItnModel']


class Vgg16(nn.Module, NNModule[torch.Tensor, torch.Tensor]):
    def __init__(self, wpath: str = None):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.net = nn.Sequential(
            self.conv1_1,
            nn.ReLU(inplace=True), 
            self.conv1_2,
            nn.ReLU(inplace=True), # out

            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv2_1,
            nn.ReLU(inplace=True),
            self.conv2_2,
            nn.ReLU(inplace=True), # out

            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3_1,
            nn.ReLU(inplace=True),
            self.conv3_2,
            nn.ReLU(inplace=True),
            self.conv3_3,   
            nn.ReLU(inplace=True), # out

            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv4_1,
            nn.ReLU(inplace=True),
            self.conv4_2,
            nn.ReLU(inplace=True),
            self.conv4_3,
            nn.ReLU(inplace=True), # out                       
                                                                                  
        )

        self.load_weights(wpath)
    
    def get_states(self, batch: torch.Tensor) -> list[torch.Tensor]:
        states = []
        for i, layer in enumerate(self.net):
            batch = layer(batch)
            if i in [3, 8, 15, 22]:
                states.append(batch)
        
        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_states(x)[-1]


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

        self.features_blobs = []

        # localization network
        self.squeeze_net = squeezenet1_1().eval() # pretrained=True
        self.squeeze_net._modules.get('features').register_forward_hook(self.hook_feature)

        # perception network
        self.vgg16 = Vgg16() # pretrained=True

        # main model: image transfer network
        self.transfer_model = TransferNet()
        
    def hook_feature(self, module, input, output):
        # self.features_blobs.append(output.data.cpu().numpy())
        self.features_blobs.append(output)

    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        return self.transfer_model(x['content'])


if __name__ == "__main__":

    # TODO: test compatability with data loader
    # TODO: not yet tested if the code works / model is training properly
    model = ItnModel()
