from __future__ import annotations

from typing import Protocol

import torch

from ..base import NNModule

__all__ = ['AdaINEncoder', 'AdaINDecoder']


class AdaINEncoder(NNModule[torch.Tensor, torch.Tensor], Protocol):
    def get_states(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """
        Similar to :meth:`torch.nn.Module.__call__`, but returns the output of
        each intermediate layer; the last output is the same as the result of
        :meth:`torch.nn.Module.__call__`.
        """
        ...


class AdaINDecoder(NNModule[torch.Tensor, torch.Tensor], Protocol):
    pass
