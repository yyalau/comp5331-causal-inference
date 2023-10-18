from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, TypeVar

import torch

__all__ = ['NNModule']


# Input
X = TypeVar('X')
X_contra = TypeVar('X_contra', contravariant=True)

# Output
Y = TypeVar('Y')
Y_co = TypeVar('Y_co', covariant=True)

class NNModule(Protocol[X_contra, Y_co]):
    """
    A partial interface for :class:`torch.nn.Module`.
    """
    def __call__(self, x: X_contra) -> Y_co:
        """
        Same as :meth:`torch.nn.Module.__call__`.
        """
        ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """
        Same as :meth:`torch.nn.Module.parameters`.
        """
        ...
