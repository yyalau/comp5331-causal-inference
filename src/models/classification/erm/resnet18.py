from __future__ import annotations

from torch import nn
from torch.utils import model_zoo

from typing import Optional
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

from .base import ERMModel, ERM_X, Classification_Y

__all__ = ['ResNet18']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Sequential | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: ERM_X):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Represents a General ResNet Model.
    """
    def __init__(
        self,
        *,
        block: type[BasicBlock],
        layers: list[int],
        # ms_class=None,
        # ms_layers=[],
        # ms_p=0.5,
        # ms_a=0.1,
        # **kwargs
    ):
        self.inplanes = 64
        super().__init__()
        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self._init_params()

    @property
    def out_features(self) -> Optional[int]:
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x: ERM_X):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: ERM_X):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)


def init_pretrained_weights(model: ResNet, model_url: str):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)

def resnet18(pretrained_url: str):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    init_pretrained_weights(model, pretrained_url)

    return model

class ResNet18(nn.Module, ERMModel):
    """
    Represents a pre-trained ResNet18 [1]_ backbone for PACS and Office-Home.

    Parameters
    ----------
    num_classes : int
        The number of unique classes that can be outputted by the model.
    pretrained_url: str
        The url to download the pretrained resnet model from.

    References
    ----------
    .. [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.
       eep residual learning for image recognition. In *CVPR*. 770--778.
       <https://doi.org/10.48550/arXiv.1512.03385>
    """

    def __init__(self, *, num_classes: int, pretrained_url: str = model_urls['resnet18']):
        super().__init__()

        self._num_classes = num_classes
        self._pretrained_url = pretrained_url

        self.backbone = resnet18(pretrained_url)
        feature_dim = self.backbone.out_features

        if feature_dim is None:
            raise ValueError(f'Not able to load classifier while the feature dimesion is {feature_dim}')

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(feature_dim, num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def pretrained_url(self) -> str:
        return self._pretrained_url

    def get_num_classes(self) -> int:
        return self.num_classes

    def get_hparams(self) -> dict[str, object]:
        return dict(num_classes=self.num_classes, pretrained_url=self.pretrained_url)

    def forward(self, x: ERM_X) -> Classification_Y:
        f = self.backbone(x)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        return y
