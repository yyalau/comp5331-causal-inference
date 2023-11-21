from __future__ import annotations

import os

import torch
import torch.nn as nn

from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel, PretrainedNNModule

__all__ = ['AdaINEncoder', 'AdaINDecoder', 'AdaINModel']


class AdaINEncoder(nn.Module, PretrainedNNModule):
    """
    Parameters
    ----------
    pretrain : bool, default True
        If `True`, loads the weights of the encoder from online.
    wpath : str, optional
        If given, loads the weights of the encoder from the provided path.
    """

    def __init__(self, *, pretrain: bool = False, freeze: bool = False):

        super().__init__()

        self.default_wpath: str = "weights/vgg19/vgg_normalised.pth"
        self.default_url: str = "https://drive.google.com/u/0/uc?id=1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU&export=download"

        self._pretrain = pretrain

        vgg = nn.Sequential(
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

        self.load_pretrain(pretrain=pretrain, net = vgg)
        self.net = vgg[:31]

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    @property
    def pretrain(self) -> bool:
        return self._pretrain

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


class AdaINDecoder(nn.Module, PretrainedNNModule):
    """
    Parameters
    ----------
    pretrain : bool, default True
        If `True`, loads the weights of the decoder from online.
    wpath : str, optional
        If given, loads the weights of the decoder from the provided path.
    """

    def __init__(self, *, pretrain: bool = False, wpath: str | None = None):
        super().__init__()

        self.default_wpath = "weights/decoder/decoder.pth"
        self.default_url = "https://drive.google.com/u/0/uc?id=1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr&export=download"

        self._pretrain = pretrain
        self._wpath = wpath

        self.net = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=0),
        )

        self.load_pretrain(pretrain=pretrain, net = self.net)

    @property
    def pretrain(self) -> bool:
        return self._pretrain

    def forward(self, x: torch.Tensor):
        return self.net(x)


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
        ckpt_path: str | None = None,
    ) -> None:
        super().__init__()

        self._encoder = encoder
        self._decoder = decoder

        self._alpha = alpha

        self.load_ckpt(ckpt_path)

    @property
    def encoder(self) -> AdaINEncoder:
        return self._encoder

    @property
    def decoder(self) -> AdaINDecoder:
        return self._decoder

    @property
    def alpha(self) -> float:
        return self._alpha

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

    def load_ckpt(self, ckpt_path: str | None) -> None:
        """
        Loads the weights for the model from a given path.
        """
        if ckpt_path is None:
            return
        if not os.path.exists(ckpt_path):
            raise ValueError(f'`ckpt_path` does not exist: {ckpt_path}')

        state_dict = torch.load(ckpt_path)['state_dict']
        # state_dict = {k.replace('network.', ''): v for k, v in state_dict.items()}

        if (self._encoder.pretrain or self._decoder.pretrain) and ckpt_path is not None:
            raise ValueError('Cannot load pre-trained weights and checkpoint weights at the same time.')

        self._encoder.load_state_dict({
            k.replace('network._encoder.', ''): v for k, v in state_dict.items() if k.startswith('network._encoder')
        })
        self._decoder.load_state_dict({
            k.replace('network._decoder.', ''): v for k, v in state_dict.items() if k.startswith('network._decoder')
        })
