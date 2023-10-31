from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger

from ...models.nst import StyleTransfer_X, StyleTransfer_Y, AdaINModel

from ..base import EvalOutput, BaseTask

__all__ = ['ItnTask', 'StyleTransfer_X', 'StyleTransfer_Y']


@dataclass(frozen=True)
class ItnEvalOutput(EvalOutput):
    x: StyleTransfer_X
    lazy_y_hat: Callable[[], StyleTransfer_Y]

class ItnTask(BaseTask[StyleTransfer_X, ItnEvalOutput, StyleTransfer_X, StyleTransfer_Y]):
    def __init__(
        self,
        network: ItnModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        gamma: float = 2.0,
        img_log_freq: int = 64,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.network = network

        self.gamma = gamma

        self.loss = self._combined_loss
        self.metrics = {
            'content_loss': self._content_loss,
            'style_loss': self._style_loss,
        }

        self.img_log_freq = img_log_freq

        # hyperparameters: TODO: move to config file
        # self.content_weight = 1 # default value
        # self.style_weight = 20 # default value
        # self.learning_rate = 1e-3 # default value

