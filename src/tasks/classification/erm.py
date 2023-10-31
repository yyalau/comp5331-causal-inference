from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...models.classification import ERM_X, ERMModel

from .base import ClassificationTask

__all__ = ['ERMTask', 'ERM_X']


class ERMTask(ClassificationTask[ERM_X]):
    __doc__ = ClassificationTask.__doc__

    def __init__(
        self,
        classifier: ERMModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        img_log_freq: int = 64,
    ) -> None:
        super().__init__(
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            img_log_freq=img_log_freq,
        )
