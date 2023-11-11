import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import squeezenet1_1, vgg16
import lightning as L

from ..base import NNModule
from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel, PretrainedNNModule
from .modules import ConvLayer, InstanceNormalization, ResidualBlock, UpsampleConvLayer
from . import utils as ut

__all__ = ['ItnModel']

class TransferNet(nn.Module, NNModule[torch.Tensor, torch.Tensor]):
    def __init__(self):
        super(TransferNet, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = InstanceNormalization(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = InstanceNormalization(64)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = InstanceNormalization(128)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

        # Non-linearities
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, X: torch.Tensor):
        in_X = X
        y = self.relu(self.in1(self.conv1(in_X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        # y = (self.tanh(y)+1)*127.5
        return y


class Vgg16(nn.Module, PretrainedNNModule):
    def __init__(self, pretrain: bool = True):
        super().__init__()
        
        self.default_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        self.default_wpath = 'weights/itn/vgg16-397923af.pth'
        
        vgg = vgg16()
        self.load_pretrain(pretrain=pretrain, net = vgg)
        self.net = vgg.features[:23]
    
    def get_states(self, batch: torch.Tensor) -> list[torch.Tensor]:
        states = []
        for i, layer in enumerate(self.net):
            batch = layer(batch)
            if i in [3, 8, 15, 22]:
                states.append(batch)
        
        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_states(x)[-1]

class SqueezeNet(nn.Module, PretrainedNNModule):
    
    def __init__(self, pretrain: bool = True):
        super().__init__()

        self.default_url = 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth'
        self.default_wpath = 'weights/itn/squeezenet1_1-b8a52dc0.pth'   
             
        self.net = squeezenet1_1()
        self.load_pretrain(pretrain=pretrain, net = self.net)
        

class ItnModel(nn.Module, StyleTransferModel):
    """
    Represents an Image Transfer Network [1]_ style transfer model for images.

    References
    ----------
    .. [1] Yijun Liu, Zuoteng Xu, Wujian Ye, Ziwen Zhang, Shaowei Weng, Chin-Chen Chang,
       and Huajin Tang. 2019. Image Neural Style Transfer With Preserving the Salient Regions.
       *IEEE Access 7* (2019), 40027--40037. <https://doi.org/10.1109/access.2019.2891576>
    """

    def __init__(self, 
                 vgg16: Vgg16,
                 squeeze_net: SqueezeNet, 
                 transfer_net: TransferNet):
        super().__init__()

        self.features_blobs = []

        # localization network
        self.squeeze_net = squeeze_net.net.eval() # pretrained=True
        self.squeeze_net._modules.get('features').register_forward_hook(self.hook_feature)

        # perception network
        self.vgg16 = Vgg16() # pretrained=True

        # main model: image transfer network
        self.transfer_net = TransferNet()
        
    def hook_feature(self, module, input, output):
        # self.features_blobs.append(output.data.cpu().numpy())
        self.features_blobs.append(output)

    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        return self.transfer_net(x['content'])

