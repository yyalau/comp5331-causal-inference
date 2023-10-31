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

__all__ = ['AdaINTask', 'StyleTransfer_X', 'StyleTransfer_Y']


@dataclass(frozen=True)
class AdaINEvalOutput(EvalOutput):
    x: StyleTransfer_X
    lazy_y_hat: Callable[[], StyleTransfer_Y]


class AdaINTask(BaseTask[StyleTransfer_X, AdaINEvalOutput, StyleTransfer_X, StyleTransfer_Y]):
    """
    Defines the train/validation/test/predict loops for an AdaIN model.

    Parameters
    ----------
    network : AdaINModel
        The AdaIN model to use.
    optimizer : callable
        A factory function that constructs a new :class:`Optimizer` instance for
        training the model.
    scheduler : callable
        A factory function that constructs a new :class:`LRScheduler` instance for
        training the model.
    gamma : float, default 2.0
        The ratio of importance between the style loss and content loss.
        The overall loss is given by `content_loss + gamma * style_loss`.
    img_log_freq : int, default 64
        Controls how often the input style (`x_style`), input content (`x_content`) and output (`y_hat`) images are logged to TensorBoard.
        Specifically, they are logged every `img_log_freq` batches during model evaluation.
    """
    def __init__(
        self,
        network: AdaINModel,
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

    def _content_loss_fn(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input_, target)

    def _style_loss_fn(self, input_states: list[torch.Tensor], target_states: list[torch.Tensor]) -> torch.Tensor:
        device = input_states[0].device
        std_loss, mean_loss = torch.tensor(0.0, device = device), torch.tensor(0.0, device = device)

        for input_state, target_state in zip(input_states, target_states):
            input_std, input_mean = torch.std_mean(input_state, dim=(-2, -1))
            target_std, target_mean = torch.std_mean(target_state, dim=(-2, -1))

            std_loss += F.mse_loss(input_std, target_std)
            mean_loss += F.mse_loss(input_mean, target_mean)

        return mean_loss + std_loss

    def _content_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        return self._content_loss_fn(enc_applied_states[-1], content)

    def _style_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        return self._style_loss_fn(enc_applied_states, enc_style_states)

    def _combined_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        content_loss = self._content_loss(enc_applied_states, enc_style_states, content)
        style_loss = self._style_loss(enc_applied_states, enc_style_states, content)
        gamma = self.gamma

        return content_loss + gamma * style_loss

    def _eval_step(self, batch: StyleTransfer_X, batch_idx: int) -> AdaINEvalOutput:
        x = batch
        
        enc_style_states = self.network.encoder.get_states(x['style']) # enc_style_states[0] = 2,64,224,224; enc_style_states[3] = 2,512,28,28;  
        enc_content = self.network.encoder(x['content']) # 2, 512, 28, 28

        enc_applied = self.network.ada_in(enc_content, enc_style_states[-1]) # 2, 512, 28, 28
        applied = self.network.decoder(enc_applied) # 2, 3, 224, 224

        enc_applied_states = self.network.encoder.get_states(applied)

        loss = self.loss(enc_applied_states, enc_style_states, enc_content)
        metrics = {name: metric(enc_applied_states, enc_style_states, enc_content) for name, metric in self.metrics.items()}

        return AdaINEvalOutput(loss=loss, metrics=metrics, x=x, lazy_y_hat=lambda: self.network(batch))

    def forward(self, batch: StyleTransfer_X) -> StyleTransfer_Y:
        return self.network(batch)

    def _process_images(self, eval_output: AdaINEvalOutput, *, prefix: str, batch_idx: int) -> None:
        if batch_idx % self.img_log_freq:
            return

        if isinstance(self.logger, TensorBoardLogger):
            writer = self.logger.experiment
            if isinstance(writer, SummaryWriter):
                writer.add_images(f'images/x_style/{prefix}{batch_idx}', eval_output.x['style'], self.current_epoch)
                writer.add_images(f'images/x_content/{prefix}{batch_idx}', eval_output.x['content'], self.current_epoch)
                writer.add_images(f'images/y_hat/{prefix}{batch_idx}', eval_output.lazy_y_hat(), self.current_epoch)
            else:
                raise TypeError('Incorrect type of writer')
        else:
            raise TypeError('Incorrect type of logger')

    def validation_step(self, batch: StyleTransfer_X, batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='val_')
        return self._process_eval_loss_metrics(eval_output, prefix='val_')

    def test_step(self, batch: StyleTransfer_X, batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='test_')
        return self._process_eval_loss_metrics(eval_output, prefix='test_')
