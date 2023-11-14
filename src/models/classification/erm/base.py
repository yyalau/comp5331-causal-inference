from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

import torch

from pathlib import Path

from ..base import ClassificationModel, Classification_Y

__all__ = ['ERM_X', 'Classification_Y', 'ERMModel']


ERM_X: TypeAlias = torch.Tensor
"""
A batch of images to classify.

Shape: `(batch_size, num_channels, height, width)`
"""

@runtime_checkable
class ERMModel(ClassificationModel[ERM_X], Protocol):
    """
    Represents a standard ERM (Empirical Risk Minimization) classifier for images.
    """

    def load_ckpt(self, ckpt_path: Path, is_fa: bool = False): ...
