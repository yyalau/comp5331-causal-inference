from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Generic

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy, F1Score, Precision, Recall
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .dependencies import NNModule, X, Y
from .dependencies import *

__all__ = ['ClassificationTask', 'FASTTask']


class BaseTask(pl.LightningModule, Generic[X, Y]):
    def __init__(
        self,
        model: NNModule[X, Y],
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        loss: Callable[[Y, Y], torch.Tensor],
        metrics: Mapping[str, Callable[[Y, Y], torch.Tensor]],
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss = loss
        self.metrics = dict(metrics)

    def forward(self, batch: X) -> Y:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.model.forward(batch)

    def training_step(self, batch: tuple[X, Y], batch_idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_metrics = self._eval_step(batch, batch_idx)

        metrics = eval_metrics
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch: tuple[X, Y], batch_idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_metrics = self._eval_step(batch, batch_idx)

        metrics = {f'val_{k}': v for k, v in eval_metrics.items()}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: tuple[X, Y], batch_idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_metrics = self._eval_step(batch, batch_idx)

        metrics = {f'test_{k}': v for k, v in eval_metrics.items()}
        self.log_dict(metrics)
        return metrics

    def _eval_step(self, batch: tuple[X, Y], batch_idx: int) -> dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        metrics = {name: metric(y_hat, y) for name, metric in self.metrics.items()}

        return {'loss': loss, **metrics}

    def predict_step(self, batch: X, batch_idx: int, dataloader_idx: int = 0) -> Y:
        return self.forward(batch)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)


class ClassificationTask(BaseTask[Classifier_X, Classifier_Y]):
    def __init__(
        self,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__(
            model=ClassifierModel(),    # TODO: Instantiate a concrete implementation
            optimizer=optimizer,
            scheduler=scheduler,
            loss=F.cross_entropy,
            metrics={
                'accuracy': Accuracy(task='multiclass'),
                'precision': Precision(task='multiclass'),
                'recall': Recall(task='multiclass'),
                'f1': F1Score(task='multiclass'),
            },
        )


class FASTTask(BaseTask[FAST_X, FAST_Y]):
    def __init__(
        self,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__(
            model=FASTModel(),    # TODO: Instantiate a concrete implementation
            optimizer=optimizer,
            scheduler=scheduler,
            loss=F.cross_entropy,
            metrics={
                'accuracy': Accuracy(task='multiclass'),
                'precision': Precision(task='multiclass'),
                'recall': Recall(task='multiclass'),
                'f1': F1Score(task='multiclass'),
            },
        )
