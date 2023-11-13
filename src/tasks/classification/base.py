from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic


import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy, F1Score, Metric, Precision, Recall

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
    img_log_max_examples_per_batch : int, default 4
        Controls the maximum number of examples that are logged per batch, regardless of the batch size.
    """

    def __init__(
        self,
        classifier: ClassificationModel[X],
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        img_log_freq: int = 64,
        img_log_max_examples_per_batch: int = 4,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.classifier = classifier

        num_classes = self.classifier.get_num_classes()

        self.loss_fn = F.cross_entropy

        # PyTorch Lightning cannot detect and automatically set the device for each metric
        # unless they are directly set as an attribute of the LightningModule
        # (as opposed to placing them inside a container like a dictionary)
        self._accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self._precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self._recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self._f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

        self.metrics = {
            'acc': self._accuracy,
            'pre': self._precision,
            'rec': self._recall,
            'f1': self._f1,
        }

        self.img_log_freq = img_log_freq
        self.img_log_max_examples_per_batch = img_log_max_examples_per_batch

    def _eval_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> ClassificationEvalOutput[X]:
        x, y = batch
        y_hat = self.classifier(x)

        loss = self.loss_fn(y_hat, y)

        for metric in self.metrics.values():
            metric.update(y_hat, y.argmax(dim=-1))

        return ClassificationEvalOutput(loss=loss, metrics=self.metrics, x=x, y=y, y_hat=y_hat)

    def forward(self, batch: X) -> Classification_Y:
        return self.classifier(batch)

    def _process_images(self, eval_output: ClassificationEvalOutput[X], *, prefix: str, batch_idx: int) -> None:
        if batch_idx % self.img_log_freq:
            return

        if isinstance(self.logger, TensorBoardLogger):
            self._log_images(self.logger.experiment, eval_output, prefix=prefix)
        else:
            raise TypeError('Incorrect type of logger')

    @abstractmethod
    def _log_images(self, writer: SummaryWriter, eval_output: ClassificationEvalOutput[X], *, prefix: str) -> None:
        raise NotImplementedError

    def training_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> dict[str, torch.Tensor | Metric]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='train_')
        return self._process_eval_loss_metrics(eval_output, prefix='')

    def validation_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> dict[str, torch.Tensor | Metric]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='val_')
        return self._process_eval_loss_metrics(eval_output, prefix='val_')

    def test_step(self, batch: tuple[X, Classification_Y], batch_idx: int) -> dict[str, torch.Tensor | Metric]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='test_')
        return self._process_eval_loss_metrics(eval_output, prefix='test_')
