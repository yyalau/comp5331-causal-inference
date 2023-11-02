from __future__ import annotations
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataops")))

import torch
import torch.nn as nn

from utils import download_from_gdrive

from ..base import NNModule

from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel

__all__ = ['AdaINEncoder', 'AdaINDecoder', 'AdaINModel']


class AdaINEncoder(nn.Module, NNModule[torch.Tensor, torch.Tensor]):
    """
    Parameters
    ----------
    wpath : str, optional
        If given, loads the weights of the encoder from the provided path.
    """

    def __init__(self, *, pretrain: bool = True, wpath: str | None = None):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),  # First layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),  # Second layer from which Style Loss is calculated
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),  # Third layer from which Style Loss is calculated
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),  # This is Relu 4.1 The output layer of the encoder.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.load_weights(pretrain, wpath)

    def load_weights(self, pretrain: bool, path: str) -> None:
        """
        Loads the weights for the VGG19 model from a given path.
        """
        
        pretrain_path = "weights/vgg19/vgg_normalised.pth"
        model_url = "https://drive.google.com/u/0/uc?id=1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU&export=download"
        
        if pretrain and (path is not None and path != pretrain_path):
            raise ValueError(f"Pretrain is True but weights is not {pretrain_path}. Specify the path if you want to load your own weights with pretrain=False")
            
        if pretrain: 
            path = pretrain_path
            if not os.path.exists(path): 
                os.makedirs(os.path.dirname(path), exist_ok=True)
                download_from_gdrive(model_url, path)

        if not os.path.exists(path): 
            raise FileNotFoundError(f"Could not find weights at {path}")
        
        self.net.load_state_dict(torch.load(path))
        

    def get_states(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """
        Similar to :meth:`torch.nn.Module.__call__`, but returns the output of
        each intermediate layer; the last output is the same as the result of
        :meth:`torch.nn.Module.__call__`.
        """
        states = []
        for i, layer in enumerate(self.net):
            batch = layer(batch)

            if i in [3, 10, 17, 30]:
                states.append(batch)

        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_states(x)[-1]


class AdaINDecoder(nn.Module, NNModule[torch.Tensor, torch.Tensor]):
    """
    Parameters
    ----------
    wpath : str, optional
        If given, loads the weights of the decoder from the provided path.
    """
    def __init__(self, *, pretrain: bool = True, wpath: str | None = None):
        super().__init__()

        # 

        self.padding = nn.ReflectionPad2d(padding=1)  # Using reflection padding as described in vgg19
        self.UpSample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0)


        self.net = nn.Sequential(
            self.padding,
            self.conv4_1,
            nn.ReLU(inplace=True),
            self.UpSample,

            self.padding,
            self.conv3_1,
            nn.ReLU(inplace=True),

            self.padding,
            self.conv3_2,
            nn.ReLU(inplace=True),

            self.padding,
            self.conv3_3,
            nn.ReLU(inplace=True),

            self.padding,
            self.conv3_4,
            nn.ReLU(inplace=True),
            self.UpSample,

            self.padding,
            self.conv2_1,
            nn.ReLU(inplace=True),

            self.padding,
            self.conv2_2,
            nn.ReLU(inplace=True),
            self.UpSample,

            self.padding,
            self.conv1_1,
            nn.ReLU(inplace=True),

            self.padding,
            self.conv1_2,
        )

        self.load_weights(pretrain, wpath)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def load_weights(self, pretrain: bool, path: str) -> None:
        """
        Loads the weights for the model from a given path.
        """
        
        pretrain_path = "weights/decoder/decoder.pth"
        model_url = "https://drive.google.com/u/0/uc?id=1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr&export=download"
        
        if pretrain and (path is not None and path != pretrain_path):
            raise ValueError(f"Pretrain is True but weights is not {pretrain_path}. Specify the path if you want to load your own weights with pretrain=False")

        if pretrain: 
            path = pretrain_path
            if not os.path.exists(path): 
                os.makedirs(os.path.dirname(path), exist_ok=True)
                download_from_gdrive(model_url, path)

        if not os.path.exists(path): 
            raise FileNotFoundError(f"Could not find weights at {path}")
        
        self.net.load_state_dict(torch.load(path))
        

class AdaINModel(nn.Module, StyleTransferModel):
    """
    Represents an AdaIN (Adaptive Instance Normalization) [1]_ style transfer model for images.

    Parameters
    ----------
    encoder : AdaINEncoder
        The encoder model to use in the network.
    decoder : AdaINDecoder
        The decoder model to use in the network.
    alpha : float, default 1.0
        The interpolation coefficient between the content features and the style features.

    References
    ----------
    .. [1] Xun Huang and Serge Belongie. 2017. Arbitrary style transfer in real-time with adaptive
       instance normalization. In *CVPR*. 1501--1510. <https://doi.org/10.48550/arXiv.1703.06868>
    """
    def __init__(
        self,
        encoder: AdaINEncoder,
        decoder: AdaINDecoder,
        *,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.alpha = alpha

    def ada_in(self, content: torch.Tensor, style: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        content_std, content_mean = torch.std_mean(content, dim=(-2, -1), keepdim=True)
        style_std, style_mean = torch.std_mean(style, dim=(-2, -1), keepdim=True)

        return style_std * (content - content_mean) / (content_std + eps) + style_mean

    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        # forward for validation and testing
        enc_style = self.encoder(x['style'])
        enc_content = self.encoder(x['content'])
        alpha = self.alpha

        enc_applied = alpha * self.ada_in(enc_content, enc_style) + (1 - alpha) * enc_content
        return self.decoder(enc_applied)
