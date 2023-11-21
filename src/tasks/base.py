from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, Any

import yaml

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Metric
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

__all__ = ['EvalOutput', 'BaseTask']


@dataclass(frozen=True)
class EvalOutput:
    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor | Metric]


Eval_X, Eval_Out = TypeVar('Eval_X'), TypeVar('Eval_Out', bound=EvalOutput)     # Model evaluation
Infer_X, Infer_Y = TypeVar('Infer_X'), TypeVar('Infer_Y')                       # Model inference


class BaseTask(pl.LightningModule, Generic[Eval_X, Eval_Out, Infer_X, Infer_Y], ABC):
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
    def _eval_step(self, batch: Eval_X, batch_idx: int) -> Eval_Out:
        raise NotImplementedError

    @abstractmethod
    def forward(self, batch: Infer_X) -> Infer_Y:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError

    def _process_eval_loss_metrics(self, eval_output: EvalOutput, *, prefix: str) -> dict[str, torch.Tensor | Metric]:
        loss_metrics = {'loss': eval_output.loss, **eval_output.metrics}
        prefixed_loss_metrics = {f'{prefix}{k}': v for k, v in loss_metrics.items()}

        self.log_dict(prefixed_loss_metrics, prog_bar=True)

        return prefixed_loss_metrics

    def _log_cli_config(self) -> None:
        logger = self.logger
        if logger is None:
            return

        log_dir = logger.log_dir
        if log_dir is None:
            return

        config_path = Path(log_dir) / 'config.yaml'
        with config_path.open() as f:
            config = yaml.load(f, yaml.SafeLoader)
            if config is not None:
                logger.log_hyperparams(config)

    def on_train_start(self) -> None:
        self._log_cli_config()

    def on_validation_start(self) -> None:
        self._log_cli_config()

    def on_test_start(self) -> None:
        self._log_cli_config()

    def on_predict_start(self) -> None:
        self._log_cli_config()

    def _update_dataloader_sampler_epoch(self, dataloader: DataLoader[Any]):
        if isinstance(dataloader.batch_sampler, DistributedSampler):
            dataloader.batch_sampler.set_epoch(self.current_epoch)

    def on_train_epoch_start(self) -> None:
        train_loader = self.trainer.train_dataloader
        if isinstance(train_loader, DataLoader):
            self._update_dataloader_sampler_epoch(train_loader)

    def on_validation_epoch_start(self) -> None:
        val_loader = self.trainer.val_dataloaders
        if isinstance(val_loader, DataLoader):
            self._update_dataloader_sampler_epoch(val_loader)

    def training_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor | Metric]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_output = self._eval_step(batch, batch_idx)
        return self._process_eval_loss_metrics(eval_output, prefix='')

    def validation_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor | Metric]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_output = self._eval_step(batch, batch_idx)
        return self._process_eval_loss_metrics(eval_output, prefix='val_')

    def test_step(self, batch: Eval_X, batch_idx: int) -> dict[str, torch.Tensor | Metric]:  # pyright: ignore[reportIncompatibleMethodOverride]
        eval_output = self._eval_step(batch, batch_idx)
        return self._process_eval_loss_metrics(eval_output, prefix='test_')

    def predict_step(self, batch: Infer_X, batch_idx: int, dataloader_idx: int = 0) -> Infer_Y:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)
