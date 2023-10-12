from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import Accuracy, F1Score, Precision, Recall
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .dependencies import (
    X, StyleTransfer_X, StyleTransfer_Y,
    AdaINEncoder, AdaINDecoder,
    ClassificationModel, Classification_Y,
    ERMModel, ERM_X,
    FASTModel, FAST_X,
)

__all__ = ['ERMTask', 'FASTTask']


Eval_X = TypeVar('Eval_X')                                  # For model evaluation
Infer_X, Infer_Y = TypeVar('Infer_X'), TypeVar('Infer_Y')   # For model inference

class BaseTask(pl.LightningModule, Generic[Eval_X, Infer_X, Infer_Y], ABC):
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


class AdaINTask(BaseTask[tuple[StyleTransfer_X, StyleTransfer_Y], StyleTransfer_X, StyleTransfer_Y]):
    def __init__(
        self,
        encoder: AdaINEncoder,
        decoder: AdaINDecoder,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        alpha: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.encoder = encoder
        self.decoder = decoder

        self.alpha = alpha
        self.gamma = gamma

        self.loss = self._combined_loss
        self.metrics = {
            'content_loss': self._content_loss,
            'style_loss': self._style_loss,
        }

    def _content_loss(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input_, target)

    def _style_loss(self, input_states: list[torch.Tensor], target_states: list[torch.Tensor]) -> torch.Tensor:
        std_loss, mean_loss = torch.tensor(0.0), torch.tensor(0.0)

        for input_state, target_state in zip(input_states, target_states):
            input_std, input_mean = torch.std_mean(input_state, dim=(-2, -1))
            target_std, target_mean = torch.std_mean(target_state, dim=(-2, -1))

            std_loss += F.mse_loss(input_std, target_std)
            mean_loss += F.mse_loss(input_mean, target_mean)

        return mean_loss + std_loss

    def _combined_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        content_loss = self._content_loss(enc_applied_states[-1], content)
        style_loss = self._style_loss(enc_applied_states, enc_style_states)
        gamma = self.gamma

        return content_loss + gamma * style_loss

    def _ada_in(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_std, content_mean = torch.std_mean(content, dim=(-2, -1))
        style_std, style_mean = torch.std_mean(style, dim=(-2, -1))

        return style_std * (content - content_mean) / content_std + style_mean

    def _eval_step(self, batch: tuple[StyleTransfer_X, StyleTransfer_Y], batch_idx: int) -> dict[str, torch.Tensor]:
        x, _ = batch

        enc_style_states = self.encoder.get_states(x['style'])
        enc_content = self.encoder(x['content'])

        enc_applied = self._ada_in(enc_content, enc_style_states[-1])
        applied = self.decoder(enc_applied)

        enc_applied_states = self.encoder.get_states(applied)

        loss = self.loss(enc_applied_states, enc_style_states, enc_content)
        metrics = {name: metric(enc_applied_states, enc_style_states, enc_content) for name, metric in self.metrics.items()}

        return {'loss': loss, **metrics}

    def forward(self, batch: StyleTransfer_X) -> StyleTransfer_Y:
        enc_style = self.encoder(batch['style'])
        enc_content = self.encoder(batch['content'])
        alpha = self.alpha

        return alpha * self._ada_in(enc_content, enc_style) + (1 - alpha) * enc_content


class ClassificationTask(BaseTask[tuple[X, Classification_Y], X, Classification_Y], Generic[X]):
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

class ERMTask(ClassificationTask[ERM_X]):
    def __init__(
        self,
        classifier: ERMModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__(
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )

class FASTTask(ClassificationTask[FAST_X]):
    def __init__(
        self,
        classifier: FASTModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
    ) -> None:
        super().__init__(
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
