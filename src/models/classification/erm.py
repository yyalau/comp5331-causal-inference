from __future__ import annotations

from typing import Protocol, TypeAlias

import torch

from .base import ClassificationModel

__all__ = ['ERM_X', 'ERMModel']


ERM_X: TypeAlias = torch.Tensor
"""
A batch of images to classify.

Shape: `(batch_size, num_channels, depth, height, width)`
"""

class ERMModel(ClassificationModel[ERM_X], Protocol):
    """
    Represents a standard ERM (Empirical Risk Minimization) classifier for images.
    """
