from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...models.classification.fa import FA_X, FAModel

from .base import ClassificationTask

__all__ = ['FATask', 'FA_X']


class FATask(ClassificationTask[FA_X]):
    __doc__ = ClassificationTask.__doc__

    def __init__(
        self,
        classifier: FAModel,
        *,
        num_classes: int,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__(
            classifier=classifier,
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
        )
