from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torchvision.models.vision_transformer import vit_b_16, vit_b_32
from torchvision.models import ViT_B_16_Weights
from .base import ERM_X, ERMModel, Classification_Y
from typing import Optional, Any

__all__ = ['ViT_wrapper']

# helpers

# T = TypeVar('T', int, float)
# MaybePair: TypeAlias = T | tuple[T, T]

# def pair(t: MaybePair[T]) -> tuple[T, T]:
#     return t if isinstance(t, tuple) else (t, t)

# classes

class ViT_wrapper(nn.Module, ERMModel):

    def __init__(
            self,
            *,
            image_size: int,
            num_classes: int,
            # dim: int,
            # depth: int,
            # heads: int,
            # mlp_dim: int,
            # pool: Literal['cls', 'mean'] = 'cls',
            # channels: int = 3,
            # dim_head: int = 64,
            # dropout_rate: float = 0.,
            # emb_dropout_rate: float = 0.,
            # pretrained = True
        ):
        super().__init__()

        self._image_size = image_size
        if self._image_size != 224:
            self.proj_layer = nn.Linear(self._image_size, 224) 

        self._num_classes = num_classes
        self.backbone = vit_b_16(ViT_B_16_Weights)
        if num_classes > 0:
            self.classifier = nn.Linear(1000, num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    def get_num_classes(self) -> int:
        return self.num_classes
    
    def load_ckpt(self, ckpt_path: Path, is_fa: bool = False):
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

    def forward(self, img: ERM_X) -> Classification_Y:
        
        if self._image_size != 224:
            img = self.proj_layer(img) 
        
        x = self.backbone(img)
        return self.classifier(x)