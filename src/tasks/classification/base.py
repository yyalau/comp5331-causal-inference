from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy, F1Score, Precision, Recall

from ...models.base import X
from ...models.classification import Classification_Y, ClassificationModel

from ..base import BaseTask

__all__ = ['ClassificationTask', 'X', 'Classification_Y']


class ClassificationTask(BaseTask[tuple[X, Classification_Y], X, Classification_Y], Generic[X]):
    """
    Defines the train/validation/test/predict loops for a classification model.

    Parameters
    ----------
    classifier : ClassificationModel
        The classification model to use.
    optimizer : callable
        A factory function that constructs a new :class:`Optimizer` instance for
        training the model.
    scheduler : callable
        A factory function that constructs a new :class:`LRScheduler` instance for
        training the model.
    """
    def __init__(
        self,
        classifier: ClassificationModel[X],
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.classifier = classifier

        self.loss = F.cross_entropy
        self.metrics = {
            'accuracy': Accuracy(task='multiclass'),
            'precision': Precision(task='multiclass'),
            'recall': Recall(task='multiclass'),
            'f1': F1Score(task='multiclass'),
        }

    def _eval_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.classifier(x)

        loss = self.loss(y_hat, y)
        metrics = {name: metric(y_hat, y) for name, metric in self.metrics.items()}

        return {'loss': loss, **metrics}

    def forward(self, batch: X) -> Classification_Y:
        return self.classifier(batch)
