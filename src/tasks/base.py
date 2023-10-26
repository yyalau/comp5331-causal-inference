from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

__all__ = ['BaseTask']


Eval_X = TypeVar('Eval_X')                                  # For model evaluation
Infer_X, Infer_Y = TypeVar('Infer_X'), TypeVar('Infer_Y')   # For model inference

class BaseTask(pl.LightningModule, Generic[Eval_X, Infer_X, Infer_Y], ABC):
    """
    Base class to define the train/validation/test/predict loops for a model.

    Parameters
    ----------
    optimizer : callable
        A factory function that constructs a new :class:`Optimizer` instance for
        training the model.
    scheduler : callable
        A factory function that constructs a new :class:`LRScheduler` instance for
        training the model.
    """
    def __init__(
        self,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler

    @abstractmethod
    def _eval_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: Infer_X) -> Infer_Y:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError

    def training_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_metrics = self._eval_step(batch, batch_idx)

        metrics = eval_metrics
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_metrics = self._eval_step(batch, batch_idx)

        metrics = {f'val_{k}': v for k, v in eval_metrics.items()}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_metrics = self._eval_step(batch, batch_idx)

        metrics = {f'test_{k}': v for k, v in eval_metrics.items()}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch: Infer_X, batch_idx: int, dataloader_idx: int = 0) -> Infer_Y:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)
