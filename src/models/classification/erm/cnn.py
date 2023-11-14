from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any
from pathlib import Path

from .base import ERMModel, ERM_X, Classification_Y

__all__ = ['SmallConvNet']

class Convolution(nn.Module):

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x: ERM_X):
        return self.relu(self.conv(x))


class ConvNet(nn.Module):
    def __init__(self, c_hidden: int = 64):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 2**2 * c_hidden

    @property
    def out_features(self) -> int:
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return 0
        return self._out_features

    def _check_input(self, x: ERM_X):
        H, W = x.shape[2:]
        assert (
            H == 32 and W == 32
        ), 'Input to network must be 32x32, " "but got {}x{}'.format(H, W)

    def forward(self, x: ERM_X) -> ERM_X:
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)

class SmallConvNet(nn.Module, ERMModel):
    """
    Represents a small convolutional network to be used for as backbone
    of DigitsDG dataset [To be trained from scratch].

    References
    ----------
    .. [1] Kaiyang Zhou, Yongxin Yang, Timothy Hospedales, and Tao Xiang. 2020.
        Deep domain-adversarial image generation for domain generalisation.
        <https://arxiv.org/pdf/2003.06054.pdf>
    """

    def __init__(self, num_classes: int = 10, pretrained_path: Path | None = None) -> None:
        super().__init__()

        self._num_classes = num_classes
        self._pretrained_path = pretrained_path

        self.backbone = ConvNet()
        self.classifier = nn.Linear(in_features=self.backbone.out_features, out_features=num_classes)

        if pretrained_path is not None:
            self.load_ckpt(pretrained_path)

    def load_ckpt(self, ckpt_path: Path, is_fa: bool = False) -> None:
        if ckpt_path.exists() and ckpt_path.is_file():
            state_dict: dict[str, Any] = torch.load(ckpt_path)['state_dict']

            if is_fa:
                self.backbone.load_state_dict({
                    k.replace('classifier._classifier.backbone.', ''): v for k, v in state_dict.items() if k.startswith('classifier._classifier.backbone.')
                })
                self.classifier.load_state_dict({
                    k.replace('classifier._classifier.classifier.', ''): v for k, v in state_dict.items() if k.startswith('classifier._classifier.classifier.')
                })
            else:
                self.backbone.load_state_dict({
                    k.replace('classifier.backbone.', ''): v for k, v in state_dict.items() if k.startswith('classifier.backbone.')
                })
                self.classifier.load_state_dict({
                    k.replace('classifier.classifier.', ''): v for k, v in state_dict.items() if k.startswith('classifier.classifier.')
                })
        else:
            raise ValueError(f'provided path {ckpt_path} does not exist to load the pretrained params.')

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def pretrained_path(self) -> Path | None:
        return self._pretrained_path

    def get_num_classes(self) -> int:
        return self.num_classes

    def forward(self, x: ERM_X) -> Classification_Y:
        backbone = self.backbone(x)
        return self.classifier(backbone)
