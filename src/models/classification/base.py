from __future__ import annotations

from typing import Protocol, TypeAlias

import torch

from ..base import NNModule, X_contra

__all__ = ['Classification_Y', 'ClassificationModel']


Classification_Y: TypeAlias = torch.Tensor
"""
A batch of class probabilities.

Shape: `(batch_size, num_classes)`
"""


class ClassificationModel(NNModule[X_contra, Classification_Y], Protocol[X_contra]):
    """
    Represents a classification model.
    """
