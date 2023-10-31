from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy, F1Score, Precision, Recall
from lightning.pytorch.loggers import TensorBoardLogger

from ...models.base import X
from ...models.classification import Classification_Y, ClassificationModel

from ..base import EvalOutput, BaseTask

__all__ = ['ClassificationTask', 'X', 'Classification_Y']


@dataclass(frozen=True)
class ClassificationEvalOutput(EvalOutput, Generic[X]):
    x: X
    y: Classification_Y
    y_hat: Classification_Y


class ClassificationTask(BaseTask[tuple[X, Classification_Y], ClassificationEvalOutput[X], X, Classification_Y], Generic[X]):
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
    img_log_freq : int, default 64
        Controls how often the input (`x`), ground truth (`y`) and output (`y_hat`) images are logged to TensorBoard.
        Specifically, they are logged every `img_log_freq` batches during model evaluation.
    """
    def __init__(
        self,
        classifier: ClassificationModel[X],
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        img_log_freq: int = 64,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.classifier = classifier

        num_classes = self.classifier.get_num_classes()

        self.loss = F.cross_entropy
        self.metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=num_classes),
            'precision': Precision(task='multiclass', num_classes=num_classes),
            'recall': Recall(task='multiclass', num_classes=num_classes),
            'f1': F1Score(task='multiclass', num_classes=num_classes),
        }

        self.img_log_freq = img_log_freq

    def _eval_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> ClassificationEvalOutput[X]:
        x, y = batch
        y_hat = self.classifier(x)

        loss = self.loss(y_hat, y)
        metrics = {name: metric(y_hat, y) for name, metric in self.metrics.items()}

        return ClassificationEvalOutput(loss=loss, metrics=metrics, x=x, y=y, y_hat=y_hat)

    def forward(self, batch: X) -> Classification_Y:
        return self.classifier(batch)

    def _process_images(self, eval_output: ClassificationEvalOutput[X], *, prefix: str, batch_idx: int) -> None:
        if batch_idx % self.img_log_freq:
            return

        if isinstance(self.logger, TensorBoardLogger):
            writer = self.logger.experiment
            if isinstance(writer, SummaryWriter):
                writer.add_images(f'images/x/{prefix}{batch_idx}', eval_output.x, self.current_epoch)
                writer.add_images(f'images/y/{prefix}{batch_idx}', eval_output.y, self.current_epoch)
                writer.add_images(f'images/y_hat/{prefix}{batch_idx}', eval_output.y_hat, self.current_epoch)
            else:
                raise TypeError('Incorrect type of writer')
        else:
            raise TypeError('Incorrect type of logger')

    def validation_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='val_')
        return self._process_eval_loss_metrics(eval_output, prefix='val_')

    def test_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='test_')
        return self._process_eval_loss_metrics(eval_output, prefix='test_')
